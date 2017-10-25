import os
import numpy as np


def get_data(train = True, path = "./xmds2/", train_len=800):

    data_file = os.path.join(path, "potential.dat")
    label_file = os.path.join(path, "energy.dat")

    f_data = open(data_file, "r")
    f_label = open(label_file, "r")

    data = np.loadtxt(f_data)
    label = np.loadtxt(f_label)

    train_data = data[0:train_len]
    train_label = label[0:train_len]

    test_data = data[train_len:len(data)]
    test_label = label[train_len:len(data)]

    if train:
        return train_data, train_label
    else:
        return test_data, test_label


#train_data, train_label = get_data(train=False)
#print(train_data.shape)
#print(train_label.shape)