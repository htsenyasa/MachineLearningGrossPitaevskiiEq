import os
import numpy as np


def read_data(data_file, label_file, train_len=800, test_len=200):


    data_file = os.path.join("../data/nonlinearSE/", data_file)
    label_file = os.path.join("../data/nonlinearSE/", label_file)

    #data = np.load(data_file)
    data = np.loadtxt(data_file)
    label = np.loadtxt(label_file)

    total_len = train_len + test_len
    train_data = data[0:train_len]
    train_label = label[0:train_len]

    test_data = data[train_len:total_len]
    test_label = label[train_len:total_len]

    return train_data, train_label, test_data, test_label


def read_data2(data_file, label_file, train_len=800, test_len=200):

    data_file = os.path.join("../data/nonlinearSE/", data_file)
    label_file = os.path.join("../data/nonlinearSE/", label_file)

    data = np.load(data_file)
    label = np.loadtxt(label_file)

    total_len = train_len + test_len
    train_data = data[0:train_len]
    train_label = label[0:train_len]

    test_data = data[train_len:total_len]
    test_label = label[train_len:total_len]

    return train_data, train_label, test_data, test_label
