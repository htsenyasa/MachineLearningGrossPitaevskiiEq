import os
import numpy as np
import sys


def read_data(data_filename, label_filename, train_len=800, test_len=200):
    data_filename = os.path.join("../data/nonlinearSE/", data_filename)
    label_filename = os.path.join("../data/nonlinearSE/", label_filename)

    #data = np.load(data_filename)
    data = np.loadtxt(data_filename)
    label = np.loadtxt(label_filename)

    total_len = train_len + test_len
    train_data = data[0:train_len]
    train_label = label[0:train_len]

    test_data = data[train_len:total_len]
    test_label = label[train_len:total_len]

    if (len(test_label) < test_len):
        print("TEST LENGTH ERROR!\n")
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
