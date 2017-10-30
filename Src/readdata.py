import os
import numpy as np


def read_data(path = "../data/nonlinearSE/", train_len=800, test_len=200):

    data_file = os.path.join(path, "potential-g-1.dat")
    label_file = os.path.join(path, "energy-g-1.dat")

    f_data = open(data_file, "r")
    f_label = open(label_file, "r")

    data = np.loadtxt(f_data)
    label = np.loadtxt(f_label)

    total_len = train_len + test_len
    train_data = data[0:train_len]
    train_label = label[0:train_len]

    test_data = data[train_len:total_len]
    test_label = label[train_len:total_len]

    return train_data, train_label, test_data, test_label


#train_data, train_label = get_data(train=False)
#print(train_data.shape)
#print(train_label.shape)
