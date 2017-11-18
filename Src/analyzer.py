import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import torch

class analyzer(object):
    testdataset = None
    predicted = None
    error = None
    def __init__(self, args):
        self.learning_rate = args.lr
        self.arch = args.network_arch
        self.training_len = args.training_len
        self.test_len = args.test_len
        self.seed = args.seed
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.cur_epoch = 1
        self.runtime_count = args.runtime_count
        self.loss = np.empty(0)

    def calc_error(self, testdataset, predicted):
        self.testdataset = testdataset
        self.predicted = predicted
        #self.error = np.array([(self.predicted[i] - self.testdataset[i])**2 for i in range(self.test_len)])
        self.error = sum((self.predicted - self.testdataset)**2)
        self.error = (self.error/self.test_len)

    def display_plot(self, file_name = None):
        plt.plot(self.testdataset, self.testdataset, "--r", label="real data")
        plt.plot(self.testdataset, self.predicted, ".")
        plt.xlabel("Real")
        plt.ylabel("Predicted")
        plt.legend()
        plt.grid()
        plt.figtext(0.6, 0.2, "{}\nlr={}\nepoch={}\ntrain_len={}\ntest_len={}\nerror={}".format(self.arch, self.learning_rate, self.cur_epoch, self.training_len, self.test_len, self.error))
        if file_name == None:
            plt.show()
        else:
            plt.savefig(file_name + ".svg", format = "svg")

    def display_plot2(self, file_name = None):
        features = ['E_int', 'E_kin', 'E_pot', 'E_Total']
        for i in range(len(features)):
            plt.figure(features[i])
            plt.plot(self.testdataset[:,i], self.testdataset[:,i], "--r", label="real data")
            plt.plot(self.testdataset[:,i], self.predicted[:,i], ".")
            plt.xlabel(features[i] + " Real")
            plt.ylabel(features[i] + " Predicted")
            plt.legend()
            plt.grid()
            plt.figtext(0.6, 0.2, "{}\nlr={}\nepoch={}\ntrain_len={}\ntest_len={}\nerror={}".format(self.arch, self.learning_rate, self.cur_epoch, self.training_len, self.test_len, self.error[i]))
        if file_name == None:
            plt.show()
        else:
            plt.savefig(file_name + ".svg", format = "svg")


    def plot_error(self):
        error = self.testdataset - self.predicted
        plt.hist(error)
        plt.show()

    def step(self, loss_val):
        #self.cur_epoch += 1
        self.loss = np.append(self.loss, loss_val)


def save_info(obj, file_name):
    f = open(file_name, "wb")
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_info(file_name):
    f = open(file_name, "rb")
    return pickle.load(f)

def save_state(model, optimizer, epoch, file_name):
    state = {'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optim_state_dict': optimizer.state_dict() }
    torch.save(state, file_name)

def load_state(file_name):
    state = torch.load(file_name)
    epoch = state['epoch']
    mst = (state['model_state_dict'])
    ost = (state['optim_state_dict'])

    return epoch, mst, ost
