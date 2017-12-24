import glob
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../../Src/")
from decimal import Decimal

import analyzer as an
epochs = [i+1 for i in range(30)]

c = ["b", "r", "g"]
markers = [
    r'$\lambda$',
    r'$\bowtie$',
    r'$\circlearrowleft$',
    r'$\clubsuit$',
    r'$\checkmark$']

labels = ["FPFS", "VPFS", "VPVS"]

def plot_loss(infos, files):

    fig, ax1 = plt.subplots()

    for i, info in enumerate(infos):
        ax1.plot(epochs, info.loss, "--" + c[i], label = labels[i])
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
    textstr = "MSE(FPFS):{:.4E}\nMSE(VPFS):{:.4E}\nMSE(VPVS):{:.4E}".format(Decimal(float(infos[0].loss[-1])), Decimal(float(infos[1].loss[-1])), Decimal(float(infos[2].loss[-1])))
    ax1.text(0.713, 0.81, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)

    #plt.show()
    file = "FPFS-VPFS-VPVS-"
    plt.savefig(file + "LOSS-" +".svg", format = "svg", dpi=1200)
    plt.clf()



files = []
file_ex = "*.inf"

for file in glob.glob(file_ex):
    files.append(os.path.splitext(file)[0])
files.sort()


#info = [an.load_info(file + ".inf") for file in files]
#plot_loss(info, files)

for file in files:
    info = an.load_info(file + ".inf")
    info.display_plot(file)
