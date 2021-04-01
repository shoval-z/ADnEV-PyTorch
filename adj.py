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
import math
import time
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def padding_same(x):
    """
    PyTorch function that works as Keras’ padding=same for a given x, for a given filters
    credit to- https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/4
    :param x: input matrix with 4 dimensions (batch_size, 32, n, m)
    :return: a new padded x that will keep his m,n dimensions after convolutional layer
    """
    in_height, in_width = x.shape[2], x.shape[3]
    filter_height, filter_width = 4, 4  # hyper parameters- can be change according to the parameters of the CNN
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
    CNN class that attempt to improve a given matrix
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
        """
        :param x: a matrix with dimensions [batch_size, 1, n, m]
        :return: (hopefully) and improve matrix wit dimensions [batch_size, m*n]
        """
        # padding to keep is the same size
        pad_left, pad_right, pad_top, pad_bottom = padding_same(x)
        out = F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])
        out = self.layer1(out)  # [batch_size, 32, n, m]
        out = self.up_sampling(out)  # [batch_size, 32, 2n, 2m]
        # padding to keep is the same size
        pad_left, pad_right, pad_top, pad_bottom = padding_same(out)
        out = F.pad(out, [pad_left, pad_right, pad_top, pad_bottom])
        out = self.layer2(out)  # [batch_size, 32, 2n, 2m]
        # padding to keep is the same size
        pad_left, pad_right, pad_top, pad_bottom = padding_same(out)
        out = F.pad(out, [pad_left, pad_right, pad_top, pad_bottom])
        out = self.layer3(out)  # [batch_size, 1, n, m]
        flatten_out = out.reshape(out.shape[0], out.shape[2] * out.shape[3], out.shape[1])  # [batch_size, m*n, 1]
        flatten_out_img, _ = self.GRU(flatten_out)  # [batch_size, m*n, 64]
        flatten_out_img = self.tanh(flatten_out_img)
        flatten_out_img = torch.squeeze(self.time_dis(flatten_out_img))  # [batch_size, m*n]
        # flatten_out_img = self.sigmoid(flatten_out_img)  # no need if we use BCEWithLogitsLoss
        return flatten_out_img.to(self.device)  # new mat [batch_size, m*n]


def eval_(model, test_loader, dh_test, extra_idx):
    """
    the evalutation function- finds the improvment of the validation\test set for all measures
    :param model: the trained model
    :param test_loader: test loader object
    :param dh_test: test dataset object
    :param extra_idx: the idx of the CV
    :return: dictionary with the measures values for each threshold and sample- the previous measure and the new measure
            the format is {t:{'old':{'p':[],'r':[],'f':[]}, 'new':{..},...}
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval().float()
    # which thresholds we want yo check
    thresholds = list(np.arange(0, 1, 0.05))
    measure_dict = dict()
    print('Start evaluate')
    with torch.no_grad():

        for idx, (X_mat, Y_mat, measure_values) in enumerate(test_loader):
            print(idx)

            true_list_old, false_list = list(), list()
            Y_mat_new = Y_mat.reshape(Y_mat.shape[2])
            X_mat_new = X_mat.reshape(Y_mat_new.shape)
            fig, ax = plt.subplots(2, 4, figsize=(40, 30))
            ax[0][0].set_xlim(0, 1)
            ax[0][1].set_xlim(0, 1)
            ax[0][2].set_xlim(0, 1)
            ax[0][3].set_xlim(0, 1)
            ax[1][0].tick_params(left=False, bottom=False)
            ax[1][1].tick_params(left=False, bottom=False)
            ax[1][2].tick_params(left=False, bottom=False)

            for x, y in zip(X_mat_new, Y_mat_new):
                if int(y) == 1:
                    true_list_old.append(float(x))
                else:
                    false_list.append(float(x))
            true_list = true_list_old  # use if you want non normalize plot

            # use if you want normalize plot:
            # maximum = max(true_list_old)
            # if maximum != 0:
            #     true_list = [proba / maximum for proba in true_list_old]
            # else:
            #     true_list = true_list_old

            sns.distplot(false_list, ax=ax[0][0], color='royalblue').set_title("false")

            sns.distplot(true_list, ax=ax[0][1], color='royalblue').set_title("true")

            sns.heatmap((Y_mat.reshape(X_mat.shape[2], X_mat.shape[3])), ax=ax[1][1], vmin=0, vmax=1,
                        cmap=ListedColormap(['white', 'black']), cbar=False).set(xticklabels=[], yticklabels=[],
                                                                                 title="real")
            sns.heatmap((X_mat.reshape(X_mat.shape[2], X_mat.shape[3])), ax=ax[1][0], vmin=0, vmax=1,
                        cmap=ListedColormap(['white', 'black']), cbar=False).set(xticklabels=[], yticklabels=[],
                                                                                 title="predicted")

            prev_p = torch.squeeze(measure_values['p']).to(device)
            prev_r = torch.squeeze(measure_values['r']).to(device)
            prev_f = torch.squeeze(measure_values['f']).to(device)
            new_mat = model(X_mat.float().to(device))
            new_mat = torch.sigmoid(new_mat)
            new_mat_new = new_mat.reshape(Y_mat_new.shape)
            true_list_old, false_list = list(), list()
            for x, y in zip(new_mat_new, Y_mat_new):
                if int(y) == 1:
                    true_list_old.append(float(x))
                else:
                    false_list.append(float(x))
            true_list = true_list_old  # use if you want non normalize plot

            # use if you want normalize plot:
            # maximum = max(true_list_old)
            # if maximum != 0:
            #     true_list = [proba / maximum for proba in true_list_old]
            # else:
            #     true_list = true_list_old

            sns.distplot(false_list, ax=ax[0][2], color='royalblue').set_title("false_new")

            sns.distplot(true_list, ax=ax[0][3], color='royalblue').set_title("true_new")

            sns.heatmap((new_mat.reshape(X_mat.shape[2], X_mat.shape[3])).cpu(), ax=ax[1][2], vmin=0, vmax=1,
                        cmap=ListedColormap(['white', 'black']), cbar=False).set(xticklabels=[], yticklabels=[],
                                                                                 title="predicted")
            fig.suptitle(f'{dh_test.alg_names[idx]}\n')
            fig.savefig(f"matrix_img_adj/{extra_idx}_{idx}.png")
            # fig.show()

            new_mat = torch.unsqueeze(new_mat, dim=0).to(device)  # needed only if batch_size=1
            Y_mat_new = Y_mat.reshape(Y_mat.shape[0], Y_mat.shape[2]).float().to(device)
            new_mat = torch.unsqueeze(new_mat, dim=0).to(device)  # needed only if batch_size=1
            for t in thresholds:
                new_p = calc_measures(new_mat, Y_mat_new, t)['p']
                new_r = calc_measures(new_mat, Y_mat_new, t)['r']
                new_f = calc_measures(new_mat, Y_mat_new, t)['f']
                # tot_new_measure = list()
                # tot_old_measure = list()
                if t not in measure_dict:
                    measure_dict[t] = dict()
                    measure_dict[t]['old'] = {'p': list(), 'r': list(), 'f': list()}
                    measure_dict[t]['new'] = {'p': list(), 'r': list(), 'f': list()}
                measure_dict[t]['old']['p'].append(float(prev_p))
                measure_dict[t]['new']['p'].append(float(new_p))
                measure_dict[t]['old']['r'].append(float(prev_r))
                measure_dict[t]['new']['r'].append(float(new_r))
                measure_dict[t]['old']['f'].append(float(prev_f))
                measure_dict[t]['new']['f'].append(float(new_f))
                # if 'old' not in measure_dict[t]:
                #     measure_dict[t]['old'] = list()
                #     measure_dict[t]['new'] = list()
                # measure_dict[t]['old'].append(float(prev_measure))
                # measure_dict[t]['new'].append(float(new_measure))
                # measure_dict[t] = dict() {'old': tot_old_measure, 'new': tot_new_measure}
                # tot_new_measure.append(float(prev_measure))
                # tot_old_measure.append(new_measure)
    return measure_dict


def calc_measures(new_mat, Y_mat, t):
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
    new_mat = (new_mat >= t).float()
    measures = dict()
    measures['p'], measures['r'], measures['f'] = precision_recall_fscore_support(
        np.array(Y_mat.reshape(Y_mat.shape[0] * Y_mat.shape[1], 1).detach().cpu().numpy()),
        np.array(new_mat.reshape(new_mat.shape[0] * new_mat.shape[1], 1).detach().cpu().numpy()),
        average='binary')[:3]
    measures['cos'] = \
        cosine_similarity(new_mat.reshape(1, -1).detach().cpu().numpy(), Y_mat.reshape(1, -1).detach().cpu().numpy())[
            0][0]
    return measures


def train(train_loader, epochs):
    """
    the train function
    :param train_loader: train loader object
    :param epochs: number of epochs to train
    :return: the trained model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    optimizer_y = torch.optim.Adam(model.parameters())
    loss_function_y = nn.BCEWithLogitsLoss().float()
    print("Start training")
    model.train()
    prev_loss = math.inf
    for epoch in range(epochs):
        T = time.time()
        tot_loss = 0
        for idx, (X_mat, Y_mat, measure_values) in enumerate(train_loader):
            new_mat = model(X_mat.float().to(device))
            new_mat = torch.unsqueeze(new_mat, dim=0).to(device)  # needed only if batch_size=1
            Y_mat = Y_mat.reshape(Y_mat.shape[0], Y_mat.shape[2]).float().to(device)
            loss = loss_function_y(new_mat, Y_mat)
            optimizer_y.zero_grad()
            loss.backward()
            optimizer_y.step()
            tot_loss += loss
        # stop if the loss is not improving between epochs
        if prev_loss <= tot_loss:
            break
        else:
            prev_loss = tot_loss
        print(
            f'epoch {epoch + 1}/{epochs}, train loss: {tot_loss / len(train_loader)}, Time: {time.time() - T} seconds')
    return model


def main():
    epochs = 5
    batch_size = 1

    kfold = KFold(5, True, 1)
    keys = conf.datafile
    extra_idx = 0
    final_t_dict = dict()
    for cv_idx, (train_keys, test_keys) in enumerate(kfold.split(keys)):
        # divide the data to train and test
        train_files = [conf.datafile[index] for index in train_keys]
        dh_train = data_prep.DataHandler(train_files)
        dh_train.build_eval()
        dh_train.update()
        train_loader = DataLoader(dh_train, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

        test_files = [conf.datafile[index] for index in test_keys]
        dh_test = data_prep.DataHandler(test_files)
        dh_test.build_eval()
        # dh_test.update()
        test_loader = DataLoader(dh_test, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

        model = train(train_loader, epochs)
        measure_dict = eval_(model, test_loader, dh_test, extra_idx)
        final_t_dict[cv_idx] = measure_dict
        print('-----------------------------------------')
        print(measure_dict)
        extra_idx += 1
    print('-----------------------------------------')
    print(final_t_dict)


if __name__ == '__main__':
    main()
