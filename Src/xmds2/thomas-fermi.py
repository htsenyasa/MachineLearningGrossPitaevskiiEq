import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

#matplotlib.rcParams.update({'axes': 16})
matplotlib.rc('axes', titlesize=16)
matplotlib.rc('axes', labelsize=16)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=14)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=14)    # fontsize of the tick labels
#matplotlib.rc('font.family', "Times New Roman")    # fontsize of the tick labels
matplotlib.rcParams["font.family"] = "Times New Roman"


os.chdir("../../data/nonlinearSE/generic_dataset/" + "harmonic/")
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
pot = np.loadtxt("potential-generic.dat")
en = np.loadtxt("energy-generic.dat")
inter = np.loadtxt("/home/user/Study/Src/APPL/Src/xmds2/inter.txt")
mu = ((9/32) * inter**2)**(1/3)

x = np.arange(-10, 10, 20/128)   
dens = np.loadtxt("dens-generic.dat")
for i in range(len(en)):
    dens2 = np.zeros(128)
    zz = mu[i] - pot[i]
    ind = np.where(zz > 0)
    dens2[ind] = zz[ind]
    dens2 /= inter[i]
    plt.plot(x, dens[i], label = "XMDS $g = {}$".format(inter[i]))
    plt.plot(x, dens2, label="TF $g = {}$".format(inter[i]))
    #plt.plot(metot, "+b", label = "MATLAB")
    plt.xlabel(r"$x$")
    plt.ylabel(r"$n(z) = |\psi(z)|^2$") #, fontsize = 16)
    plt.legend()
    file_path = "../../../../figs/numericanalyze/"
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    if i == 1 or i == 5 or i == 10 or i == 15:
        plt.savefig(file_path + "thomas-fermi-{}".format(inter[i]) + ".png", format = "png", bbox_inches='tight') #, dpi=800)    
#        plt.show()
    plt.clf()