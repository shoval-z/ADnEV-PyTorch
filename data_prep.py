import Config as conf
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from random import randint


class DataHandler:
    """
    define the dataset out of the csv files and it's properties
    """

    def __init__(self, files):
        """
        go over the different filed and different algorithms and create the different dictionaries and arrays
        :param files: a list of csv files with the following columns:
        instance - the schema pair id
        alg - the algorithm provide the matching between the schemas.
        candName - the name of the attribute of the first schema (e.g table_name.attribute_name).
        targName -the same as above.
        conf -the confidence level (value between 0 to 1) of the algorithm in the attribute matching.
        realConf - 1 if the attributes are really matched together and 0 otherwise.
        """
        for idx, reg_file in enumerate(files):
            self.reg_flie = reg_file
            self.reg_df = None
            self.create_reg()
            # create the empty dictionaries only one time
            if idx == 0:
                self.conf_dict = {}
                self.instances = {}
                self.conf_dict_mat = {}
                self.conf_dict_seq = {}
                self.realConf_dict = {}
                self.realConf_dict_mat = {}
                self.trans = {}
                self.matN = {}
                self.matM = {}
                self.k = 0
                self.i = 0
                self.orig_mats_mat = {}
                self.orig_mats_seq = {}
                self.fullMat_dict = {}
                self.feat_dict = {}
                self.alg_names = {}
                self.orig_results = pd.DataFrame(columns=['instance', 'P', 'R', 'F', 'COS'])
            # for each file, we will calculate the wanted matrix for each algorithm separately
            for alg in self.reg_df['alg'].unique():
                self.build_dataset_reg(alg)

    def create_reg(self):
        """
        create a DataFrame object that contains the wanted values from a single given file
        """
        mylist = []
        for chunk in pd.read_csv(self.reg_flie, chunksize=10 ** 6):  # low_memory=False,
            mylist.append(chunk)
        self.reg_df = pd.concat(mylist, axis=0)
        self.reg_df['pair'] = self.reg_df['candName'] + '<->' + self.reg_df['targName']
        del mylist

    def build_dataset_reg(self, alg):
        """
        for the current file and for a given algorithm, we create the real matrix (the ground truth) and the predicted
        matrix. we also save the matrix sizes and a dictionary to map them to the input schema pair
        :param alg: the algorithm we are now going throw from out input file
        """
        new_df = self.reg_df[self.reg_df['alg'] == alg]
        for name, group in new_df.groupby(by=["instance"]):
            name = str(name)
            clean_name = name.replace(",", " ").replace("  ", " ")
            if clean_name not in self.instances:
                self.instances[clean_name] = self.k
                self.k += 1
            clean_name = clean_name + ' ' + alg
            if self.i not in self.conf_dict:
                self.trans[clean_name] = self.i
                self.alg_names[self.i] = clean_name
                self.i += 1
                self.conf_dict[self.trans[clean_name]] = np.array([])
                self.matN[self.trans[clean_name]] = group['targName'].value_counts()[-1]
                self.matM[self.trans[clean_name]] = group['candName'].value_counts()[-1]
            if self.i not in self.realConf_dict:
                self.realConf_dict[self.trans[clean_name]] = np.array(group['realConf'])
            self.conf_dict[self.trans[clean_name]] = np.append(self.conf_dict[self.trans[clean_name]],
                                                               np.array(group['conf']))

    def build_eval(self):
        """
        for every schema pair and for every algorithm, we calculate the it's scores-
        precision, recall, F1, and cosine similarity
        we also update some of the object dictionaries
        """
        for k in self.conf_dict:
            self.fullMat_dict[k] = {}
            self.fullMat_dict[k]['p'], self.fullMat_dict[k]['r'], self.fullMat_dict[k][
                'f'] = precision_recall_fscore_support(
                np.ceil(np.array(self.conf_dict[k].reshape(len(self.conf_dict[k]), 1))),
                np.array(self.realConf_dict[k].reshape(len(self.realConf_dict[k]), 1)), average='binary')[:3]
            for e in self.fullMat_dict[k]:
                self.fullMat_dict[k][e] = np.array(self.fullMat_dict[k][e]).reshape(1, 1)
            self.fullMat_dict[k]['cos'] = cosine_similarity(self.conf_dict[k].reshape(1, -1),
                                                            self.realConf_dict[k].reshape(1, -1))
            self.conf_dict_mat[k] = self.conf_dict[k].reshape(1, self.matN[k], self.matM[k])
            self.conf_dict_seq[k] = self.conf_dict[k].reshape(1, len(self.conf_dict[k]), 1)
            self.orig_mats_mat[k] = self.conf_dict[k].reshape(self.matN[k], self.matM[k])
            self.orig_mats_seq[k] = self.conf_dict[k].reshape(len(self.conf_dict[k]))
            if np.isnan(np.min(self.conf_dict[k])):
                print("**********")
            self.realConf_dict[k] = np.array(self.realConf_dict[k].reshape(1, len(self.realConf_dict[k]), 1))
            self.realConf_dict_mat[k] = np.array(self.realConf_dict[k].reshape(1, self.matN[k], self.matM[k]))

    def swap_rows_and_cols(self, X_mat, Y_mat):
        """
        We expand out dataset by swapping rows and columns for each pair of predicted matrix and ground truth matrix
        :param X_mat: the predicted matrix
        :param Y_mat: the ground truth matrix
        :return: the predicted & ground truth matrices after swapping 2 rows and 2 columns
        """
        X_mat_new = X_mat[0]
        Y_mat_new = Y_mat[0].reshape(X_mat_new.shape)
        r1_row = randint(0, len(X_mat_new) - 1)
        r2_row = randint(0, len(X_mat_new) - 1)
        r1_col = randint(0, len(X_mat_new[0]) - 1)
        r2_col = randint(0, len(X_mat_new[0]) - 1)
        X_mat_new[[r1_row, r2_row]] = X_mat_new[[r2_row, r1_row]]
        Y_mat_new[[r1_row, r2_row]] = Y_mat_new[[r2_row, r1_row]]
        X_mat_new = X_mat_new.reshape(X_mat_new.shape[1], X_mat_new.shape[0])
        Y_mat_new = Y_mat_new.reshape(Y_mat_new.shape[1], Y_mat_new.shape[0])
        X_mat_new[[r1_col, r2_col]] = X_mat_new[[r2_col, r1_col]]
        Y_mat_new[[r1_col, r2_col]] = Y_mat_new[[r2_col, r1_col]]
        return X_mat_new.reshape(X_mat_new.shape[1], X_mat_new.shape[0]), Y_mat_new.reshape(Y_mat_new.shape[1],
                                                                                            Y_mat_new.shape[0])

    def update(self):
        """
        calling the 'swap_rows_and_cols' function for every couple of matrices and update our dataset with the
        additional matrices.
        :return:
        """
        dict1 = self.conf_dict_mat.copy()
        dict2 = self.realConf_dict.copy()
        dict3 = self.fullMat_dict.copy()
        for item1, item2, item3 in zip(dict1.items(), dict2.items(), dict3.items()):
            for i in range(3):  # define how many times we make a new permutations for each input couple
                last_index = len(self.conf_dict_mat)
                X, Y = self.swap_rows_and_cols(item1[1], item2[1])
                self.conf_dict_mat[last_index] = X.reshape(item1[1].shape)
                self.realConf_dict[last_index] = Y.reshape(item2[1].shape)
                self.fullMat_dict[last_index] = item3[1]

    def __getitem__(self, index):
        """
        the function that is called while looping over the DataLoader object.
        you can choose to return the matrix in it's original size or expand it according to a given value
        (e.g the max height and width in the dataset) by using the padding part instead- you have to use it if the
        batch size is not 1
        :param index: the index of the wanted element
        :return: the predicted matrix, the ground truth matrix,
            the scores of the predicted matrix {'p':, 'r':, 'f':,'cos':}
        """
        return self.conf_dict_mat[index], self.realConf_dict[index], self.fullMat_dict[index]

        ### padding part ###

        # pad_x = 241 - self.conf_dict_mat[index].shape[1]
        # pad_y = 219 - self.conf_dict_mat[index].shape[2]
        # pad_real = 241 * 219 - self.realConf_dict[index].shape[1]
        # round_x, round_y, round_real = int(pad_x % 2), int(pad_y % 2), int(pad_real % 2)
        # return np.pad(self.conf_dict_mat[index],
        #               ((0, 0), (int(np.ceil(pad_x / 2)), int(np.ceil(pad_x / 2) - round_x)),
        #                (int(np.ceil(pad_y / 2)), int(np.ceil(pad_y / 2) - round_y)))), \
        #        np.pad(self.realConf_dict[index],
        #               ((0, 0), (int(np.ceil(pad_real / 2)), int(np.ceil(pad_real / 2) - round_real)),
        #                (0, 0))), \
        #        self.fullMat_dict[index]

    def __len__(self):
        return len(self.conf_dict_mat)


if __name__ == '__main__':
    # kfold = KFold(5, True, 1)
    # keys = conf.extra_data
    dh = DataHandler(conf.datafile)
    # dh.build_eval(False)
    # for data in conf.extra_data:
    #     dh.add_more_files(data)
    dh.build_eval()
    # dh.build_feat_dataset()
    # kfold = KFold(5, True, 1)
    # keys = np.array(list(dh.instances.keys()))[:]
    # for train, test in kfold.split(keys):
    #     pass
