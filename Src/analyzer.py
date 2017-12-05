import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle
import torch
from scipy.optimize import curve_fit

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
        self.relative_error = sum(np.abs(self.predicted - self.testdataset)/sum(self.testdataset))


#    def display_plot(self, file_name = None):
#        plt.plot(self.testdataset, self.testdataset, "--r", label="real data")
#        plt.plot(self.testdataset, self.predicted, ".", label = "predicted")
#        plt.xlabel("Real")
#        plt.ylabel("Predicted")
#        plt.legend()
#        plt.grid()
#        #plt.figtext(0.6, 0.2, "{}\nlr={}\nepoch={}\ntrain_len={}\ntest_len={}\nerror={}".format(self.arch, self.learning_rate, self.cur_epoch, self.training_len, self.test_len, self.error))
#        pos = [0.65, 0.15, .2, .2]
#        inset = plt.axes(pos)
#        err = self.testdataset - self.predicted
#        plt.hist(err)
#        plt.title("Error")
#        plt.xticks([])
#        plt.yticks([])
#
#        if file_name == None:
#            plt.show()
#        else:
#            plt.savefig(file_name + ".svg", format = "svg")

    def display_plot(self, file_name = None):
        fig, ax1 = plt.subplots()
        left, bottom, width, height = [0.65, 0.20, .2, .2]
        inset = fig.add_axes([left, bottom, width, height])
        ax1.plot(self.testdataset, self.testdataset, "--r", label="real data")
        ax1.plot(self.testdataset, self.predicted, ".", label = "predicted")
        ax1.set_title("FNN{}".format(self.arch))
        ax1.set_xlabel("Real")
        ax1.set_ylabel("Predicted")
        ax1.legend()
        ax1.grid()
        #plt.figtext(0.6, 0.2, "{}\nlr={}\nepoch={}\ntrain_len={}\ntest_len={}\nerror={}".format(self.arch, self.learning_rate, self.cur_epoch, self.training_len, self.test_len, self.error))
        err = self.testdataset - self.predicted

        inset.hist(err, range=[-np.amax(np.abs(err)), np.amax(np.abs(err))], bins=20)
        inset.set_title("Error")

        if file_name == None:
            plt.show()
        else:
            figure = plt.gcf()
            figure.set_size_inches(8,6)
            plt.savefig(file_name + ".svg", format = "svg", dpi=1200)


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
            plt.figtext(0.6, 0.2, "{}\nlr={}\nepoch={}\ntrain_len={}\ntest_len={}\nerror={}".format(self.arch, self.learning_rate, self.cur_epoch, self.training_len, self.test_len, self.relative_error[i]))
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

#    def gauss(x, a, x0, sigma):
#        return a * np.exp(-(x-x0)**2 / (2*sigma**2))
#
#    def gaussian_dist():
#        err = self.testdataset - self.predicted)
#        x_r = np.amax(np.abs(err))
#        x = np.linspace(-x_r, x_r, self.test_len)
#        mean = np.mean(err)
#        sigma = np.std(err)
#        popt,pcov = curve_fit(gauss, x, err, p0 = [1,mean,sigma])
#        plt.hist(err)
#        plt.plot(x, gauss(x, *popt))
#        plt.show()


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
