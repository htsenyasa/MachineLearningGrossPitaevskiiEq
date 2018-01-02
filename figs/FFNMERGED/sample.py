import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append("../../Src/")
from decimal import Decimal

import analyzer as an

def display_plot2(info, file_name = None):
    features = ['E_int', 'E_kin', 'E_pot', 'E_Total']
    features_l = ['$E_{int}$', '$E_{kin}$', '$E_{pot}$', '$E_{tot}$']
    err = []
    relative_err = []
    for i in range(len(features)):
        err.append(info.testdataset[:,i] - info.predicted[:,i])
        relative_err.append(((info.testdataset[:,i] - info.predicted[:,i]) / (info.testdataset[:,i])) * 100)
        fig, ax1 = plt.subplots(num=features[i])
        left, bottom, width, height = [0.65, 0.20, .2, .2]
        inset = fig.add_axes([left, bottom, width, height])
        left2, bottom2, width2, height2 = [0.19, 0.60, .2, .2]
        if i != 0: inset2 = fig.add_axes([left2, bottom2, width2, height2])
        if i == 0: ax1.set_ylim(-3e-15, -3e-15)
        ax1.plot(info.testdataset[:,i], info.testdataset[:,i], "--r", linewidth = 3)
        ax1.plot(info.testdataset[:,i], info.predicted[:,i], ".", markersize = 2)
        #ax1.set_title("FNN{}".format(info.arch))
        ax1.set_xlabel("True " + features_l[i], fontsize = 20)
        ax1.set_ylabel("Predicted " + features_l[i], fontsize = 20)
        ax1.tick_params(labelsize = 18)
        ax1.legend()
        ax1.grid()
        #props = dict(boxstyle='square', facecolor='white', alpha=0.5)
        #textstr = "MSE:{:.4E}\nTraining Length:{}\nEpoch:{}\nBatch:{}".format(Decimal(float(info.error[i])), info.training_len, info.cur_epoch, info.batch_size)
        #ax1.text(0.03, 0.85, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)
        inset.hist(err[i], range=[-np.amax(np.abs(err[i])), np.amax(np.abs(err[i]))], bins=20)
        inset.set_title("Error")
        inset.tick_params(labelsize=12)
        if i != 0:
            inset2.hist(relative_err[i], range=[-np.amax(np.abs(relative_err[i])), np.amax(np.abs(relative_err[i]))], bins=20)
            inset2.set_title("Rel.Error (%)", fontsize = 18)
            inset2.tick_params(labelsize=12)
        plt.show()
#        plt.clf()
#    plt.savefig(file_name + ".svg", format = "svg", dpi=1200)
#    plt.clf()

files = ["potential-g-0-epoch-60-"]

for file in files:
    info = an.load_info(file + ".testinf")
    display_plot2(info, file)
    #if(file.find("epoch-60") != -1):
    #    plot_loss(info, file)
