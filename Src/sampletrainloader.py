from __future__ import print_function
import torch
import torch
import numpy as np
import readmnistdata as rm

class nonlinearSE(object):
    def __init__(self, root = "../data"):
        self.train_len = 50000
        self.test_len = 10000

        data = list(rm.read())
        self.train_data = np.zeros([self.train_len, 28, 28])
        self.train_label = np.zeros([self.train_len], dtype = 'int')

        self.test_data = np.zeros([self.test_len, 28, 28])
        self.test_label = np.zeros([self.test_len], dtype = 'int')

        for i in range(self.train_len):
            self.train_data[i], self.train_label[i] = data[i]

        for i in range(self.test_len):
            self.test_data[i], self.test_label[i] = data[i + self.train_len]

        self.train_data_tensor = torch.from_numpy(self.train_data).float()
        self.train_label_tensor = torch.from_numpy(self.train_label).long()
        self.train_data_tensor = self.train_data_tensor.unsqueeze(1)

        self.test_data_tensor = torch.from_numpy(self.test_data).float()
        self.test_label_tensor = torch.from_numpy(self.test_label).long()
        self.test_data_tensor = self.test_data_tensor.unsqueeze(1)


    def get_data(self, train = True):
        if train:
            return (self.train_data_tensor, self.train_label_tensor)
        else:
            return (self.test_data_tensor, self.test_label_tensor)
