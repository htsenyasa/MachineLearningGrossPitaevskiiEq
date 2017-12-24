import glob
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../../Src/")
from decimal import Decimal

import analyzer as an
epochs = [i+1 for i in range(60)]

def plot_loss(info, file):
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, info.loss, "--b", label = "Loss")
    ax1.set_xlabel("Epochs", fontsize=18)
    ax1.set_ylabel("Loss", fontsize=18)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.grid()
    ax1.legend()
    #plt.ylim(1e-3, 1e-1)
    ax1.tick_params(labelsize=18)
    figure = plt.gcf()
    figure.set_size_inches(8,6)

    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    textstr = "MSE(20):{:.4E}\nMSE(40):{:.4E}\nMSE(60):{:.4E}".format(Decimal(float(info.loss[19])), Decimal(float(info.loss[39])), Decimal(float(info.loss[-1])))
    ax1.text(0.75, 0.9, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)

    plt.show()
    #plt.savefig(file + "LOSS-" +".svg", format = "svg", dpi=1200)
    plt.clf()



files = []
file_ex = "*.inf"

for file in glob.glob(file_ex):
    files.append(os.path.splitext(file)[0])
files.sort()

for file in files:
    info = an.load_info(file + ".inf")
    info.display_plot()
    #if(file.find("epoch-60") != -1):
    #    plot_loss(info, file)
