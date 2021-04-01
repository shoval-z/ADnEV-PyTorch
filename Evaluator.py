from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import data_prep
import Config as conf
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
import math
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
    CNN class that attempt to predict the measure of a given matrix
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
        self.GRU = nn.GRU(input_size=1, hidden_size=64, num_layers=1,
                          batch_first=True)
        self.time_dis = TimeDistributed(nn.Linear(64, 1))
        self.dense = nn.Linear(64, 1)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

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
        flatten_out, _ = self.GRU(flatten_out)  # [batch_size, m*n, 64]
        # calc measure (p/r/f/cos)
        measure = flatten_out.permute(1, 0, 2)[-1]
        measure = self.sigmoid(self.dense(measure)[0])

        return measure.to(self.device)  # measure(p,r,f,cos)


def eval(model, test_loader, wanted_measure='p'):
    """
    the evalutation function- finds the improvment of the validation\test set for all measures
    :param model: the trained model
    :param test_loader: test loader object
    :param wanted_measure: the measure we want to predict
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_function_measure = nn.MSELoss().float()
    model.eval().float()
    print('Start evaluate')
    with torch.no_grad():
        tot_loss = 0
        for idx, (X_mat, Y_mat, measure_values) in enumerate(test_loader):
            origin_measure = torch.squeeze(measure_values[wanted_measure]).to(device)
            measure = model(X_mat.float().to(device))
            tot_loss += loss_function_measure(measure.to(device), origin_measure)
            print('real val:', measure_values[wanted_measure])
            print('predicted measure:', measure)
        tot_loss = tot_loss / len(test_loader)
        print('Total test loss: ', tot_loss)


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
    :param wanted_measure: the measure we want to predict
    :return: the trained model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_function_measure = nn.MSELoss().float()
    print("Start training")
    model.train()
    prev_loss = math.inf
    for epoch in range(epochs):
        T = time.time()
        tot_loss = 0
        tot_measure = 0
        for idx, (X_mat, Y_mat, measure_values) in enumerate(train_loader):
            tmp_train_loss = 0
            new_measure = model(X_mat.float().to(device))
            correct_new_measure = measure_values[wanted_measure]
            loss = loss_function_measure(new_measure.float().to(device),
                                         torch.tensor(correct_new_measure).reshape(new_measure.shape).float().to(
                                             device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tmp_train_loss += loss.item()
            tot_loss += loss
            tot_measure += new_measure
        if prev_loss <= tot_loss:
            break
        else:
            prev_loss = tot_loss
        print(
            f'epoch {epoch + 1}/{epochs}, train loss: {tot_loss / len(train_loader)}, {wanted_measure}: {tot_measure / len(train_loader)}\n Time: {time.time() - T} seconds')
    return model


def main():
    epochs = 5
    batch_size = 1

    kfold = KFold(5, True, 1)
    keys = conf.datafile
    for train_keys, test_keys in kfold.split(keys):
        train_files = [conf.datafile[index] for index in train_keys]
        dh_train = data_prep.DataHandler(train_files)
        dh_train.build_eval()
        dh_train.update()
        train_loader = DataLoader(dh_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

        test_files = [conf.datafile[index] for index in test_keys]
        dh_test = data_prep.DataHandler(test_files)
        dh_test.build_eval()
        dh_test.update()
        test_loader = DataLoader(dh_test, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        model = train(train_loader, epochs, wanted_measure='p')
        eval(model, test_loader, wanted_measure='p')
        break


if __name__ == '__main__':
    main()
