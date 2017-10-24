from __future__ import print_function
import torch
import torch
import numpy as np
import readmnistdata as rm
import readdata as rd

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

class nonlinear1D__(object):
    def __init__(self, root = "../data", label_vectorize = False):
        self.train_len = 50000
        self.test_len = 10000

        data = list(rm.read(matrix = False))

        self.train_data = np.zeros([self.train_len, 784])
        self.test_data = np.zeros([self.test_len, 784])

        if label_vectorize:
            self.train_label = np.zeros([self.train_len, 10], dtype = 'int')
            self.test_label = np.zeros([self.test_len, 10], dtype = 'int')
        else:
            self.train_label = np.zeros([self.train_len], dtype = 'int')
            self.test_label = np.zeros([self.test_len], dtype = 'int')


        for i in range(self.train_len):
            x, y = data[i]
            self.train_data[i], self.train_label[i] = x, get_label(y, label_vectorize)

        for i in range(self.test_len):
            x, y = data[i + self.train_len]
            self.test_data[i], self.test_label[i] = x, get_label(y, label_vectorize)

        self.train_data_tensor = torch.from_numpy(self.train_data).float()
        self.train_label_tensor = torch.from_numpy(self.train_label).long()
#        self.train_data_tensor = self.train_data_tensor.unsqueeze(1)

        self.test_data_tensor = torch.from_numpy(self.test_data).float()
        self.test_label_tensor = torch.from_numpy(self.test_label).long()
#        self.test_data_tensor = self.test_data_tensor.unsqueeze(1)


    def get_data(self, train = True):
        if train:
            return (self.train_data_tensor, self.train_label_tensor)
        else:
            return (self.test_data_tensor, self.test_label_tensor)

class nonlinear1D(object):
    def __init__(self, train_len, test_len, root = "../data", label_vectorize = False):
        self.train_len = train_len
        self.test_len = test_len

        self.train_data, self.train_label = rd.get_data(train=True)
        self.test_data, self.test_label = rd.get_data(train=False)

        self.train_data_tensor = torch.from_numpy(self.train_data).float()
        self.train_label_tensor = torch.from_numpy(self.train_label).float()
#        self.train_data_tensor = self.train_data_tensor.unsqueeze(1)

        self.test_data_tensor = torch.from_numpy(self.test_data).float()
        self.test_label_tensor = torch.from_numpy(self.test_label).float()
#        self.test_data_tensor = self.test_data_tensor.unsqueeze(1)

    def get_data(self, train = True):
        if train:
            return (self.train_data_tensor, self.train_label_tensor)
        else:
            return (self.test_data_tensor, self.test_label_tensor)

def get_label(j, label_vectorize):
    if not label_vectorize:
        return j
    e = np.zeros((10))
    e[j] = 1.0
    return e
