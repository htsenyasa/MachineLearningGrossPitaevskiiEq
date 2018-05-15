import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import newton

#matplotlib.rcParams.update({'axes': 16})
matplotlib.rc('axes', titlesize=16)
matplotlib.rc('axes', labelsize=16)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=14)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=14)    # fontsize of the tick labels

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params) 

pot_type = "gaussian"
os.chdir("../../data/nonlinearSE/generic_dataset/" + pot_type + "/")

N = -1
V = np.loadtxt("potential-generic.dat")
inter = np.loadtxt("/home/user/Study/Src/APPL/Src/xmds2/inter.txt")
pot = V[N]
x = np.arange(-10, 10, 20/256)
dens = np.loadtxt("dens-generic.dat")
dx = np.abs(x[0] - x[1])

def ptl_num(mu, pot, gg):
    return np.clip((mu - pot) / gg, a_min=0, a_max=None)


def fmu(mu):
    from scipy.integrate import trapz
    gg = inter[N]
    return trapz(ptl_num(mu, pot, gg)) * dx - 1
    #return mu**2-mu


def calc_fmu(val):
    mu = newton(fmu, val)
    return mu


def display_tf_mu():
    os.chdir("../harmonic/")
    pot = np.loadtxt("potential-generic.dat")
    en = np.loadtxt("energy-generic.dat")
    inter = np.loadtxt("/home/user/Study/Src/APPL/Src/xmds2/inter.txt")
    mu = ((9/32) * inter**2)**(1/3)

    Np = 256
    x = np.arange(-10, 10, 20/Np)
    dens = np.loadtxt("dens-generic.dat")
    index = [0, -1]
    fig, ax1 = plt.subplots()
    ax1.plot(x, pot[0])
    ax1.set_xlabel('$x$', fontsize = 16)
    ax1.set_ylabel('Potential', fontsize = 16)
    ax1.set_ylim(0, None)
    ax1.tick_params('$y$')
    ax1.tick_params(labelsize = 18)
    ax2 = ax1.twinx()

    for i in index:
        dens2 = np.zeros(Np)
        zz = mu[i] - pot[i]
        ind = np.where(zz > 0)
        dens2[ind] = zz[ind]
        dens2 /= inter[i]
        ax2.plot(x, dens[i], label = "XMDS $g = {}$".format(inter[i]))
        ax2.plot(x, dens2, label = "TF $g = {}$".format(inter[i]))
    
    ax2.set_ylabel('$|\psi(x)|^2$', fontsize = 16)
    ax2.tick_params('$y$')
    ax2.set_ylim(0, None)
    ax2.tick_params(labelsize = 18)
    fig.tight_layout()
    plt.legend()
    file_path = "../../../../figs/numericanalyze/"
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    plt.savefig(file_path + "thomas-fermi-harmonic-gen" + ".png", format = "png", bbox_inches='tight') #, dpi=800)    
    #plt.show()
    plt.clf()

def calc_tf_ntf():
    mu = calc_fmu((min(pot) + max(pot))/2)
    n_tf = ptl_num(mu, pot, inter[N])

    fig, ax1 = plt.subplots()
    ax1.plot(x, pot)
    ax1.set_xlabel('$x$', fontsize = 16)
    ax1.set_ylabel('Potential', fontsize = 16)
    ax1.set_ylim(0, None)
    ax1.tick_params('$y$')
    ax1.tick_params(labelsize = 18)

    ax2 = ax1.twinx()
    ax2.plot(x, dens[N], label = "XMDS")
    ax2.plot(x, n_tf, label = "TF")
    ax2.set_ylabel('$|\psi(x)|^2$', fontsize = 16)
    ax2.tick_params('$y$')
    ax2.set_ylim(0, None)
    ax2.tick_params(labelsize = 18)
    fig.tight_layout()
    plt.legend()

    file_path = "../../../../figs/numericanalyze/"
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    plt.savefig(file_path + "thomas-fermi-{}-{}".format(pot_type, inter[N]) + ".png", format = "png", bbox_inches='tight') #, dpi=800)    
    #plt.show()
    plt.clf()

#calc_tf_ntf()