from __future__ import print_function
import torch
import torch.utils.data as data_utils
import numpy as np
import nlse_readdata as rd
import nlse_datadisthandler as ddh
from sklearn.preprocessing import StandardScaler

class Dataloader(object):
    def __init__(self, data_filename, 
                       label_filename, 
                       train_len, 
                       test_len, 
                       read_func = rd.read_data2, 
                       label_vectorize = False, 
                       unsqueeze = False):
                       
        self.train_len = train_len
        self.test_len = test_len
        scaler = StandardScaler()

        self.train_data, self.train_label, self.test_data, self.test_label = read_func(data_filename, label_filename, self.train_len, self.test_len)

        #self.train_label = self.train_label.reshape((len(self.train_label), 1))
        #self.test_label = self.test_label.reshape((len(self.test_label), 1))
        #self.train_data = scaler.fit_transform(self.train_data)
        #self.test_data = scaler.fit_transform(self.test_data)
        #self.train_label = scaler.fit_transform(self.train_label)
        #self.test_label = scaler.fit_transform(self.test_label)

        self.train_data_tensor = torch.from_numpy(self.train_data).float()
        self.train_label_tensor = torch.from_numpy(self.train_label).float()
        if unsqueeze == True: self.train_data_tensor = self.train_data_tensor.unsqueeze(1)

        self.test_data_tensor = torch.from_numpy(self.test_data).float()
        self.test_label_tensor = torch.from_numpy(self.test_label).float()
        if unsqueeze == True: self.test_data_tensor = self.test_data_tensor.unsqueeze(1)

    def get_data(self, train = True):
        if train:
            return (self.train_data_tensor, self.train_label_tensor)
        else:
            return (self.test_data_tensor, self.test_label_tensor)

    def init_tensor_dataset(self):
        train_dataset = data_utils.TensorDataset(*self.get_data())
        test_dataset = data_utils.TensorDataset(*self.get_data(train = False))
        return train_dataset, test_dataset
        
class DataloaderTransformed(object):
    def __init__(self, data_filename,
                       label_filename, 
                       train_len, 
                       test_len, 
                       transform_func = ddh.make_balanced,
                       both = False, 
                       read_func = rd.read_data2, 
                       label_vectorize = False, 
                       unsqueeze = False):
                       
        self.train_len = train_len
        self.test_len = test_len
        scaler = StandardScaler()

        self.train_data, self.train_label, self.test_data, self.test_label = read_func(data_filename, label_filename, self.train_len, self.test_len)
        self.train_data, self.train_label = transform_func(self.train_data, self.train_label)
        self.train_data = np.random.choice(self.train_data, train_len)
        self.train_label = np.random.choice(self.train_label, train_len)
        

        #self.train_label = self.train_label.reshape((len(self.train_label), 1))
        #self.test_label = self.test_label.reshape((len(self.test_label), 1))
        #self.train_data = scaler.fit_transform(self.train_data)
        #self.test_data = scaler.fit_transform(self.test_data)
        #self.train_label = scaler.fit_transform(self.train_label)
        #self.test_label = scaler.fit_transform(self.test_label)

        self.train_data_tensor = torch.from_numpy(self.train_data).float()
        self.train_label_tensor = torch.from_numpy(self.train_label).float()
        if unsqueeze == True: self.train_data_tensor = self.train_data_tensor.unsqueeze(1)

        self.test_data_tensor = torch.from_numpy(self.test_data).float()
        self.test_label_tensor = torch.from_numpy(self.test_label).float()
        if unsqueeze == True: self.test_data_tensor = self.test_data_tensor.unsqueeze(1)

    def get_data(self, train = True):
        if train:
            return (self.train_data_tensor, self.train_label_tensor)
        else:
            return (self.test_data_tensor, self.test_label_tensor)

    def init_tensor_dataset(self):
        train_dataset = data_utils.TensorDataset(*self.get_data())
        test_dataset = data_utils.TensorDataset(*self.get_data(train = False))
        return train_dataset, test_dataset




def get_label(j, label_vectorize):
    if not label_vectorize:
        return j
    e = np.zeros((10))
    e[j] = 1.0
    return e
