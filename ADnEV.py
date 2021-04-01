from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import data_prep
import Config as conf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from torchvision.utils import save_image
import time
import torch.nn.functional as F


def padding_same(x):
    """
    PyTorch function that works as Keras’ padding=same for a given x, for a given filters
    credit to- https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/4
    :param x: input matrix with 4 dimensions (batch_size, 32, n, m)
    :return: a new padded x that will keep his m,n dimensions after convolutional layer
    """
    in_height, in_width = x.shape[2], x.shape[3]
    filter_height, filter_width = 4, 4
    strides = (None, 1, 1)
    out_height = np.ceil(float(in_height) / float(strides[1]))
    out_width = np.ceil(float(in_width) / float(strides[2]))

    # The total padding applied along the height and width is computed as:

    if (in_height % strides[1] == 0):
        pad_along_height = max(filter_height - strides[1], 0)
    else:
        pad_along_height = max(filter_height - (in_height % strides[1]), 0)
    if (in_width % strides[2] == 0):
        pad_along_width = max(filter_width - strides[2], 0)
    else:
        pad_along_width = max(filter_width - (in_width % strides[2]), 0)

    # Finally, the padding on the top, bottom, left and right are:

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    # new_x = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])
    # return new_x
    return pad_left, pad_right, pad_top, pad_bottom


class TimeDistributed(nn.Module):
    """
    PyTorch function that works as Keras’ Timedistributed
    credit to- https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    """

    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y


class CNN(nn.Module):
    """
    CNN class that attempt to improve a given matrix and predict its measure
    """

    def __init__(self):
        super(CNN, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4),
            nn.BatchNorm2d(32, momentum=0.8),
            nn.ReLU())
        self.up_sampling = nn.Upsample(scale_factor=2)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=4),
            nn.BatchNorm2d(32, momentum=0.8),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=4),
            nn.MaxPool2d(2))
        self.GRU1 = nn.GRU(input_size=1, hidden_size=64, num_layers=1,
                           batch_first=True)
        self.GRU2 = nn.GRU(input_size=1, hidden_size=64, num_layers=1,
                           batch_first=True)
        self.time_dis = TimeDistributed(nn.Linear(64, 1))
        self.dense = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    # @profile
    def forward(self, x):
        pad_left, pad_right, pad_top, pad_bottom = padding_same(x)
        out = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])
        out = self.layer1(out)  # [batch_size, 32, n, m]
        out = self.up_sampling(out)  # [batch_size, 32, 2n, 2m]
        pad_left, pad_right, pad_top, pad_bottom = padding_same(out)
        out = F.pad(out, [pad_left, pad_right, pad_top, pad_bottom])
        out = self.layer2(out)  # [batch_size, 32, 2n, 2m]
        pad_left, pad_right, pad_top, pad_bottom = padding_same(out)
        out = F.pad(out, [pad_left, pad_right, pad_top, pad_bottom])
        out = self.layer3(out)  # [batch_size, 1, n, m]
        flatten_out = out.reshape(out.shape[0], out.shape[2] * out.shape[3], out.shape[1])  # [batch_size, m*n, 1]

        # improve the matrix
        flatten_out_img, _ = self.GRU1(flatten_out)  # [batch_size, m*n, 64]
        flatten_out_img = self.tanh(flatten_out_img)
        flatten_out_img = torch.squeeze(self.time_dis(flatten_out_img))  # [batch_size, m*n]

        # calc measure (p/r/f/cos)
        flatten_out, _ = self.GRU2(flatten_out)
        flatten_out = self.tanh(flatten_out)
        measure = flatten_out.permute(1, 0, 2)[-1]
        measure = self.sigmoid(self.dense(measure)[0])
        return flatten_out_img.to(self.device), measure  # new mat, measure(p\r\f\cos)


def eval(model, test_loader, wanted_measure='p'):
    """
    the evalutation function- finds the improvment of the validation\test set for all measures
    attempt to improve the matrix as long as the measure increasing
    :param model: the trained model
    :param test_loader: test loader object
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval().float()
    new_mat_measure = dict()
    print('Start evaluate')
    with torch.no_grad():
        tot_loss = 0
        tot_measure = 0
        for idx, (X_mat, Y_mat, measure_values) in enumerate(test_loader):
            print(idx)
            origin_X = X_mat.to(device)
            prev_measure = 0
            improved = True
            while improved:
                origin_measure = torch.squeeze(measure_values[wanted_measure]).to(device)
                new_mat, measure = model(X_mat.float().to(device))
                new_mat = torch.unsqueeze(new_mat, dim=0).to(device)  # needed only if batch_size=1
                Y_mat_new = Y_mat.reshape(Y_mat.shape[0], Y_mat.shape[2]).float().to(device)
                # loss = 0.5 * loss_function_y(new_mat, Y_mat_new)
                # loss += 0.5 * loss_function_measure(measure.to(device), origin_measure)
                # cur_measures = calc_measures(new_mat, Y_mat_new)
                # cur_measure = cur_measures[wanted_measure]
                # X_mat_new = X_mat.reshape(X_mat.shape[1], X_mat.shape[2] * X_mat.shape[3])
                # prev_measures = calc_measures(X_mat_new, Y_mat_new)
                # prev_measure = prev_measures[wanted_measure]
                cur_measure = torch.squeeze(measure)
                new_mat = torch.unsqueeze(new_mat, dim=0).to(device)  # needed only if batch_size=1
                correct_new_measure = calc_measures(new_mat, Y_mat_new)[wanted_measure]
                if prev_measure >= cur_measure:
                    print('real new val:', correct_new_measure)
                    print('predicted new measure:', measure_values[wanted_measure])
                    print('real prev measure: ', correct_old_measure)
                    print('predicted prev measure: ', prev_measure)
                    new_mat_measure[X_mat] = prev_measure
                    # tot_loss += loss
                    tot_measure += prev_measure
                    improved = False

                    ##plot matrix
                    img = torch.stack(tensors=(Y_mat.reshape(1, X_mat.shape[2], X_mat.shape[3]).to(device),
                                               torch.ceil(origin_X.reshape(1, X_mat.shape[2], X_mat.shape[3])).to(
                                                   device),
                                               torch.round(X_mat.reshape(1, X_mat.shape[2], X_mat.shape[3]).double().to(
                                                   device))), dim=0)
                    img = 1 - img
                    save_image(img, f'matrix_img/{idx}.png')
                else:
                    X_mat = new_mat.reshape(X_mat.shape)
                    prev_measure = cur_measure
                    correct_old_measure = correct_new_measure
        tot_loss = tot_loss / len(test_loader)
        tot_measure = tot_measure / len(test_loader)
        print('Total test loss: ', tot_loss)
        print(f'Total test {wanted_measure}: ', tot_measure)


def calc_measures(new_mat, Y_mat):
    """
    find all measures (p,r,f1,cosine) for a given threshold
    :param new_mat: the predicted matrix
    :param Y_mat: the true matrix
    :param t: the threshold
    :return: dictionary of measures
    """
    Y_mat = torch.squeeze(Y_mat)
    Y_mat = torch.unsqueeze(Y_mat, dim=0)
    new_mat = new_mat.reshape(Y_mat.shape)
    measures = dict()
    measures['p'], measures['r'], measures['f'] = precision_recall_fscore_support(
        np.ceil(np.array(new_mat.reshape(new_mat.shape[0] * new_mat.shape[1], 1).detach().cpu().numpy())),
        np.array(Y_mat.reshape(Y_mat.shape[0] * Y_mat.shape[1], 1).detach().cpu().numpy()), average='binary')[:3]
    measures['cos'] = \
        cosine_similarity(new_mat.reshape(1, -1).detach().cpu().numpy(), Y_mat.reshape(1, -1).detach().cpu().numpy())[
            0][0]
    return measures


def train(train_loader, epochs, wanted_measure='p'):
    """
    the train function
    :param train_loader: train loader object
    :param epochs: number of epochs to train
    :param wanted_measure: the measure we attempt to predict
    :return: the trained model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optimizer_y = torch.optim.Adam(model.parameters())
    optimizer_measure = torch.optim.Adam(model.parameters())
    loss_function_y = nn.BCEWithLogitsLoss().float()
    loss_function_measure = nn.MSELoss().float()
    print("Start training")
    model.train()
    for epoch in range(epochs):
        T = time.time()
        tot_loss = 0
        tot_loss1, tot_loss2 = 0, 0
        tot_measure = 0
        for idx, (X_mat, Y_mat, measure_values) in enumerate(train_loader):
            print(idx)
            new_mat, new_measure = model(X_mat.float().to(device))
            new_mat = torch.unsqueeze(new_mat, dim=0).to(device)  # needed only if batch_size=1
            Y_mat = Y_mat.reshape(Y_mat.shape[0], Y_mat.shape[2]).float().to(device)
            correct_new_measure = calc_measures(new_mat, Y_mat)[wanted_measure]
            loss = 0.5 * loss_function_y(new_mat, Y_mat)
            loss1 = loss_function_y(new_mat, Y_mat)
            loss += 0.5 * loss_function_measure(torch.squeeze(new_measure.float().to(device)),
                                                torch.tensor(correct_new_measure).float().to(device))
            loss2 = loss_function_measure(torch.squeeze(new_measure.float().to(device)),
                                          torch.squeeze(measure_values[wanted_measure]).float().to(device))
            optimizer_y.zero_grad()
            optimizer_measure.zero_grad()
            loss.backward()
            optimizer_y.step()
            optimizer_measure.step()
            tot_loss += loss
            tot_loss1 += loss1
            tot_loss2 += loss2
            tot_measure += new_measure
        print(
            f'epoch {epoch + 1}/{epochs}, train loss: {tot_loss / len(train_loader)}, {wanted_measure}: {tot_measure / ((idx) * 2)}\n Time: {time.time() - T} seconds')
        print(
            f'epoch {epoch + 1}/{epochs}, BCE loss: {tot_loss1 / len(train_loader)}, MSE loss: {tot_loss2 / ((idx) * 2)}\n')
    return model


def main():
    epochs = 1
    batch_size = 1

    kfold = KFold(5, True, 1)
    keys = conf.datafile
    for train_keys, test_keys in kfold.split(keys):
        train_files = [conf.datafile[index] for index in train_keys]
        dh_train = data_prep.DataHandler(train_files)
        dh_train.build_eval()
        dh_train.update()
        train_loader = DataLoader(dh_train, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

        test_files = [conf.datafile[index] for index in test_keys]
        dh_test = data_prep.DataHandler(test_files)
        dh_test.build_eval()
        test_loader = DataLoader(dh_test, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        model = train(train_loader, epochs, wanted_measure='f')
        eval(model, test_loader, wanted_measure='f')


if __name__ == '__main__':
    main()
