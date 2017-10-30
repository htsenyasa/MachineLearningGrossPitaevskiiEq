import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

class analyze(object):
    def __init__(self, testdataset, predicted, args):
        self.testdataset = testdataset
        self.predicted = predicted
        self.training_len = args.training_len
        self.test_len = len(testdataset)
        self.error = np.array([abs((self.predicted[i] - self.testdataset[i])**2) for i in range(self.test_len)])
        self.error = float(sum(self.error/self.test_len))
        self.learning_rate = args.lr
        self.arch = args.network_arch
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.runtime_count = args.runtime_count

    def display_plot(self):
        plt.plot(self.testdataset, self.testdataset, "--r", label="real data")
        plt.plot(self.testdataset, self.predicted, ".")
        plt.xlabel("Real")
        plt.ylabel("Predicted")
        plt.legend()
        plt.grid()
        plt.figtext(0.6, 0.2, "{}\nlr={}\nepoch={}\ntrain_len={}\ntest_len={}\nerror={}".format(self.arch, self.learning_rate, self.epochs, self.training_len, self.test_len, self.error))
        plt.show()


def save_info(obj, file_name):
    f = open(file_name, "wb")
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
def load_info(file_name):
    f = open(file_name, "rb")
    return pickle.load(f)
