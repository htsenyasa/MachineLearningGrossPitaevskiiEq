import numpy as np
import matplotlib
#matplotlib.use("pgf")
import matplotlib.pyplot as plt
import argparse
import pickle
import torch
from scipy.optimize import curve_fit
from decimal import Decimal

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
        self.relative_error = sum(np.abs(self.predicted - self.testdataset) / sum(self.testdataset))

    def display_plot(self, file_name = None, inter = False):
        err = self.testdataset - self.predicted
        relative_err = ((self.testdataset - self.predicted) / (self.testdataset)) * 100
        fig, ax1 = plt.subplots()
        left, bottom, width, height = [0.65, 0.20, .2, .2]
        inset = fig.add_axes([left, bottom, width, height])
        left2, bottom2, width2, height2 = [0.19, 0.60, .2, .2]
        inset2 = fig.add_axes([left2, bottom2, width2, height2])
        ax1.plot(self.testdataset, self.testdataset, "--r", label=None, linewidth = 1, alpha=0.5)
        #ax1.scatter(self.testdataset, self.predicted, c = np.abs(err), s = 4)
        ax1.plot(self.testdataset, self.predicted, ".", label = None, markersize = 4)
        #ax1.set_title("FNN{}".format(self.arch))
        if inter == True:
            x_label = "True Interaction Parameter"
            y_label = "Predicted Interaction Parameter"
        else:
            x_label = "True Energy"
            y_label = "Predicted Energy"

        ax1.set_xlabel(x_label, fontsize = 20)
        ax1.set_ylabel(y_label, fontsize = 20)
        ax1.tick_params(labelsize = 18)
        ax1.legend()
        ax1.grid()
        #props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        #textstr = "MSE:{:.4E}\nTraining Length:{}\nEpoch:{}\nBatch:{}".format(Decimal(float(self.error)), self.training_len, self.cur_epoch, self.batch_size)
        #ax1.text(0.03, 0.85, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)
        #ax1.text(, , "{}\nlr={}\nepoch={}\ntrain_len={}\ntest_len={}\nerror={}".format(self.arch, self.learning_rate, self.cur_epoch, self.training_len, self.test_len, self.error))

        inset.hist(err, range=[-np.amax(np.abs(err)), np.amax(np.abs(err))], bins=20)
        inset2.hist(relative_err, range=[-np.amax(np.abs(relative_err)), np.amax(np.abs(relative_err))], bins=20)
        inset.set_title("Error", fontsize = 18)
        inset.tick_params(labelsize=12)
        inset2.set_title("Rel.Error (%)", fontsize = 18)
        inset2.tick_params(labelsize=12)

        figure = plt.gcf()
        figure.set_size_inches(8,6)
        plt.show()        
        
        #if file_name == None:
        #    figure = plt.gcf()
        #    figure.set_size_inches(8,6)
        #    plt.show()
        #else:
        #    figure = plt.gcf()
        #    figure.set_size_inches(8,6)
        #    plt.savefig(file_name + ".svg", format = "svg", dpi=1200)
        #    plt.clf()




    def display_plot2(self, file_name = None):
        features = ['E_int', 'E_kin', 'E_pot', 'E_Total']
        features_l = ['$E_{int}$', '$E_{kin}$', '$E_{pot}$', '$E_{tot}$']
        err = []
        relative_err = []

        for i in range(len(features)):
            err.append(self.testdataset[:,i] - self.predicted[:,i])
            relative_err.append(((self.testdataset[:,i] - self.predicted[:,i]) / (self.testdataset[:,i])) * 100)
            fig, ax1 = plt.subplots(num=features[i])
            left, bottom, width, height = [0.65, 0.20, .2, .2]
            inset = fig.add_axes([left, bottom, width, height])
            left2, bottom2, width2, height2 = [0.19, 0.60, .2, .2]
            inset2 = fig.add_axes([left2, bottom2, width2, height2])
            ax1.plot(self.testdataset[:,i], self.testdataset[:,i], "--r", linewidth = 3)
            ax1.plot(self.testdataset[:,i], self.predicted[:,i], ".", markersize = 2)
            #ax1.set_title("FNN{}".format(self.arch))
            ax1.set_xlabel("True " + features_l[i], fontsize = 20)
            ax1.set_ylabel("Predicted " + features_l[i], fontsize = 20)
            ax1.tick_params(labelsize = 18)
            ax1.legend()
            ax1.grid()
            #props = dict(boxstyle='square', facecolor='white', alpha=0.5)
            #textstr = "MSE:{:.4E}\nTraining Length:{}\nEpoch:{}\nBatch:{}".format(Decimal(float(self.error[i])), self.training_len, self.cur_epoch, self.batch_size)
            #ax1.text(0.03, 0.85, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)


            inset.hist(err[i], range=[-np.amax(np.abs(err[i])), np.amax(np.abs(err[i]))], bins=20)
            inset2.hist(relative_err[i], range=[-np.amax(np.abs(relative_err[i])), np.amax(np.abs(relative_err[i]))], bins=20)
            inset.set_title("Error")
            inset.tick_params(labelsize=12)
            inset2.set_title("Rel.Error (%)", fontsize = 18)
            inset2.tick_params(labelsize=12)


            if file_name != None:
                fig.savefig(file_name + "{}-".format(features[i]) + ".svg", format = "svg")
                fig.clf()

        if file_name == None:
            plt.show()
            plt.clf()

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


def combine_plots(datum, file_name = None):
    fig, axarr = plt.subplots(1, len(datum), sharey=True)
    #left, bottom, width, height = [0.65, 0.20, .2, .2]

    err = []
    for i, data in enumerate(datum):
        err.append(data.testdataset - data.predicted)
        #inset = fig.add_axes([left, bottom, width, height])
        axarr[i].plot(data.testdataset, data.testdataset, "--r", label="real data", linewidth = 3)
        axarr[i].plot(data.testdataset, data.predicted, ".", label = "predicted", markersize = 2)
        axarr[i].set_title("FNN{}".format(data.arch))
        axarr[i].set_xlabel("Real $\\mu$", fontsize = 18)
        axarr[i].set_ylabel("Predicted", fontsize = 18)
        axarr[i].tick_params(labelsize = 12)
        axarr[i].legend()
        axarr[i].grid()
        props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        textstr = "MSE:{:.4E}\nTraining Length:{}\nEpoch:{}\nBatch:{}".format(Decimal(float(data.error)), data.training_len, data.cur_epoch, data.batch_size)
        axarr[i].text(0.03, 0.85, textstr, transform=axarr[i].transAxes, fontsize=11, verticalalignment='top', bbox=props)

    plt.show()
