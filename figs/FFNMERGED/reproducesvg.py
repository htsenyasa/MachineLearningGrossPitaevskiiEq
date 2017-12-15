import glob
import sys
import os
import matplotlib.pyplot as plt
sys.path.append("../../Src/")

import analyzer as an
epochs = [i+1 for i in range(60)]

def plot_loss(info, file):
    plt.plot(epochs, info.loss, "--b", label = "Loss")
    plt.xlabel("Epochs", fontsize=18)
    plt.ylabel("Loss", fontsize=18)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plt.grid()
    plt.legend()
    #plt.ylim(1e-3, 1e-1)
    #plt.show()
    plt.tick_params(labelsize=12)
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    plt.savefig(file + "LOSS-" +".svg", format = "svg", dpi=1200)
    plt.clf()



files = []
file_ex = "*.inf"
features = ['E_int', 'E_kin', 'E_pot', 'E_Total']


for file in glob.glob(file_ex):
    files.append(os.path.splitext(file)[0])
files.sort()

for file in files:
    info = an.load_info(file + ".inf")
    #info.display_plot2(file)
    if(file.find("epoch-60") != -1):
        plot_loss(info, file)
