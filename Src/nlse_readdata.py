import os
import numpy as np
import sys


def read_data(data_filename, label_filename, train_len=900, test_len=300, folders = None):
    path = "../data/nonlinearSE/generic_dataset/"

    if folders == None:
        folders = [name for name in os.listdir(path) if os.path.isdir(path + name)]
    length = len(folders)

    train_len //=length
    test_len //=length

    data = np.array([np.load(os.path.join(path + folders[i], data_filename)) for i in range(len(folders))])
    label = np.array([np.load(os.path.join(path + folders[i], label_filename)) for i in range(len(folders))])

    total_len = train_len + test_len
    train_data = np.array([data[i][0:train_len] for i in range(length)])
    train_label = np.array([label[i][0:train_len] for i in range(length)])

    test_data = np.array([data[i][train_len:total_len] for i in range(length)])
    test_label = np.array([label[i][train_len:total_len] for i in range(length)])
    
    train_data = np.reshape(train_data, (length * train_len, 2, data.shape[-1]))
    train_label = np.reshape(train_label, (length * train_len, 1))
    test_data = np.reshape(test_data, (length * test_len, 2, data.shape[-1]))
    test_label = np.reshape(test_label, (length * test_len, 1))
    
    if (test_label.shape[0] < test_len):
        print("TEST LENGTH ERROR!\n")
        print("Number of test examples is lower than desired!\n")
        print(test_label.shape[0])
        sys.exit(1)

    return train_data, train_label, test_data, test_label


def read_data2(data_filename, label_filename, train_len=800, test_len=200):

    data_filename = os.path.join("../data/nonlinearSE/", data_filename)
    label_filename = os.path.join("../data/nonlinearSE/", label_filename)

    data = np.load(data_filename)
    label = np.loadtxt(label_filename)

    total_len = train_len + test_len
    train_data = data[0:train_len]
    train_label = label[0:train_len]

    test_data = data[train_len:total_len]
    test_label = label[train_len:total_len]

    return train_data, train_label, test_data, test_label
