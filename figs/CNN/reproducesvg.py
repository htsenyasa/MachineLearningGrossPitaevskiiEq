import glob
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("../../Src/")

import analyzer as an
epochs = [i+1 for i in range(60)]

def plot_loss(info, file):
    plt.plot(epochs, np.log(info.loss), "--b", label = "$Log$(Loss)")
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("$Log(Loss)$", fontsize=18)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid()
    plt.legend()
    #plt.ylim(1e-3, 1e-1)
    plt.tick_params(labelsize=12)
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    plt.show()
    #plt.savefig(file + "LOSS-" +".svg", format = "svg", dpi=1200)
    #plt.clf()



files = []
file_ex = "*.inf"

for file in glob.glob(file_ex):
    files.append(os.path.splitext(file)[0])
files.sort()

for file in files:
    info = an.load_info(file + ".inf")
    #info.display_plot(file)
    if(file.find("epoch-60") != -1):
        plot_loss(info, file)
