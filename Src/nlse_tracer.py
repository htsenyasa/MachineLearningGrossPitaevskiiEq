import numpy as np
import matplotlib
import argparse
import pickle
import torch
from scipy.optimize import curve_fit
from decimal import Decimal
import time as time

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rc

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params) 

class Tracer(object):
    test_dataset = None
    predicted = None
    mse_error = None
    chrono_points = {}
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
        self.chrono_points["init"] = time.time()


    def load_data_to_analyze(self, predicted):
        self.predicted = predicted


    def chrono_point(self, label):
        self.chrono_points[label] = time.time()


    def analyze(self, test_dataset, predicted):
        self.predicted = predicted
        self.test_dataset = test_dataset
        self.mse_error = sum((self.predicted - self.test_dataset)**2) / self.test_len
        self.relative_error = ((self.test_dataset - self.predicted) / (self.test_dataset)) * 100
        self.diff_error = self.test_dataset - self.predicted


    def plot_figure(self, file_name = None, inter = False):
        err = self.test_dataset - self.predicted
        relative_err = (np.abs(self.test_dataset - self.predicted) / (self.test_dataset)) * 100

        fig, ax1 = plt.subplots()
        #ax1.scatter(self.testdataset, self.predicted, c = np.abs(err), s = 4)
        ax1.plot(self.test_dataset, self.predicted, ".", label = None, markersize = 4)
        ax1.plot(self.test_dataset, self.test_dataset, "--r", label=None, linewidth = 1, alpha=0.5)
        #yticks = ax1.get_yticks()
        #ax1.set_xticks(yticks)
        #ax1.set_title("FNN{}".format(self.arch))
        if inter == True:
            x_label = "True Interaction Parameter"
            y_label = "Predicted Interaction Parameter"
        else:
            x_label = "True Energy"
            y_label = "Predicted Energy"

        ax1.set_xlabel(x_label, fontsize = 26)
        ax1.set_ylabel(y_label, fontsize = 26)
        #fig.tight_layout()
        ax1.tick_params(labelsize = 18)
        ax1.legend()

        left, bottom, width, height = [0.635, 0.135, .25, .25]
        inset = fig.add_axes([left, bottom, width, height])
        left2, bottom2, width2, height2 = [0.135, 0.61, .25, .25]
        inset2 = fig.add_axes([left2, bottom2, width2, height2])

        #ax1.grid()
        #props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        #textstr = "MSE:{:.4E}\nTraining Length:{}\nEpoch:{}\nBatch:{}".format(Decimal(float(self.error)), self.training_len, self.cur_epoch, self.batch_size)
        #ax1.text(0.03, 0.85, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)
        #ax1.text(, , "{}\nlr={}\nepoch={}\ntrain_len={}\ntest_len={}\nerror={}".format(self.arch, self.learning_rate, self.cur_epoch, self.training_len, self.test_len, self.error))

        #inset.hist(err, range=[-np.amax(np.abs(err)), np.amax(np.abs(err))], bins=20)
        #inset2.hist(relative_err, range=[-np.amax(np.abs(relative_err)), np.amax(np.abs(relative_err))], bins=20)
        inset_fontsize = 16
        inset_ticksize = 14
        epochs = np.arange(1, self.epochs + 1, 1)
        inset.plot(epochs, self.loss)
        inset.set_xlabel("Epochs", fontsize=20)
        inset.set_ylabel("Loss", fontsize=20)
        inset.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        inset.legend()
        inset.set_yscale("log")
        inset.xaxis.tick_top()
        inset.xaxis.set_label_position('top')
        inset.set_xlabel("Epoch", fontsize = inset_fontsize)
        inset.set_ylabel("Loss", fontsize = inset_fontsize)
        inset.tick_params(labelsize=inset_ticksize)

        #inset2.set_title("Rel.Error (\%)", fontsize = 18)
        #inset2.hist(err, range=[-np.amax(np.abs(err)), np.amax(np.abs(err))], bins=20)
        inset2_legend = "{:.2F}".format(float(sum(np.abs(relative_err))/self.test_len)) # FIX # Generalize
        inset2.hist(relative_err, range=[0, 100], bins=20, label = inset2_legend)
        #inset2.hist(relative_err, range=[0, max(np.abs(relative_err))], bins=20, label = inset2_legend)
        #inset2.hist(relative_err, bins=20)
        #inset2.set_xlabel("Error $E_T - E_P$", fontsize = inset_fontsize)
        inset2.set_xlabel("Rel.Error (\%)", fontsize = inset_fontsize)
        inset2.set_ylabel("\# of Examples", fontsize = inset_fontsize)
        inset2.tick_params(labelsize=inset_ticksize)
        inset2.yaxis.tick_right()
        inset2.yaxis.set_label_position("right")
        inset2.legend(fontsize = 16)
        #inset2.legend("{}".format(sum(np.abs(relative_err))/self.test_len)) # FIX # Generalize
        
        figure = plt.gcf()
        figure.set_size_inches(8,6)
        
        if file_name == None:
            plt.show()
        else:
            plt.savefig(file_name + ".png", format = "png")#, dpi=1200)
            plt.clf()


    def plot_figure2(self, file_name = None):
        features = ['E_int', 'E_kin', 'E_pot', 'E_Total']
        features_l = ['$E_{int}$', '$E_{kin}$', '$E_{pot}$', '$E_{tot}$']
        err = []
        relative_err = []

        for i in range(len(features)):
            err.append(self.test_dataset[:,i] - self.predicted[:,i])
            relative_err.append(((self.test_dataset[:,i] - self.predicted[:,i]) / (self.test_dataset[:,i])) * 100)
            fig, ax1 = plt.subplots(num=features[i])
            left, bottom, width, height = [0.65, 0.20, .2, .2]
            inset = fig.add_axes([left, bottom, width, height])
            left2, bottom2, width2, height2 = [0.19, 0.60, .2, .2]
            inset2 = fig.add_axes([left2, bottom2, width2, height2])
            ax1.plot(self.test_dataset[:,i], self.test_dataset[:,i], "--r", linewidth = 3)
            ax1.plot(self.test_dataset[:,i], self.predicted[:,i], ".", markersize = 2)
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

    def plot_loss(self, file_name = None):
        #fig, ax1 = plt.subplots()
        #ax1.plot(self.loss, "--")
        #ax1.set_xlabel("Epoch", fontsize = 20)
        #ax1.set_ylabel("Loss", fontsize = 20)
        #ax1.set_yscale("log")
        #ax1.legend()
        epochs = np.arange(1, self.epochs + 1, 1)

        fig, ax1 = plt.subplots()
        ax1.plot(epochs, self.loss, "--b", label = "Loss")
        plt.locator_params(axis='x', nbins=len(epochs)/5)
        #ax1.set_xticks(epochs)
        ax1.set_xlabel("Epochs", fontsize=20)
        ax1.set_ylabel("Loss", fontsize=20)
        ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        #ax1.grid()
        ax1.legend()
        ax1.set_yscale("log")
        ax1.tick_params(labelsize=18)
        figure = plt.gcf()
        figure.set_size_inches(8,6)

        #props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        #textstr = "MSE(20):{:.2E}\nMSE(40):{:.2E}\nMSE(60):{:.2E}".format(Decimal(float(info.loss[19])), Decimal(float(info.loss[39])), Decimal(float(info.loss[-1])))
        #ax1.text(0.718, 0.9, textstr, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)

        figure = plt.gcf()
        figure.set_size_inches(8,6)
        
        if file_name == None:
            plt.show()
        else:
            plt.savefig(file_name + ".png", format = "png")#, dpi=1200)
            plt.clf()

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


def get_state(file_name):
    state = torch.load(file_name)
    epoch = state['epoch']
    mst = (state['model_state_dict'])
    ost = (state['optim_state_dict'])
    return epoch, mst, ost

def update_state(model, optimizer, inf_s, inf_d, epoch, mst, ost):
    inf_d.loss = inf_s.loss
    inf_d.cur_epoch = inf_s.cur_epoch
    inf_d.epochs += epoch - 1
    model.load_state_dict(mst)
    optimizer.load_state_dict(ost)




