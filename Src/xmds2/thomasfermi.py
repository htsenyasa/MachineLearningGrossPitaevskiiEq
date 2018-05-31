import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.optimize import newton
import scipy.optimize as spo

#matplotlib.rcParams.update({'axes': 16})
import matplotlib
matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=22)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=20)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=20)    # fontsize of the tick labels

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params) 

x = np.arange(-10, 10, 20/256)
dx = np.abs(x[0] - x[1])

def ptl_num(mu, pot, gg):
    return np.clip((mu - pot) / gg, a_min=0, a_max=None)


def fmu(mu, pot, gg):
    from scipy.integrate import trapz
    return trapz(ptl_num(mu, pot, gg)) * dx - 1
    #return mu**2-mu


def calc_fmu(val, pot, gg):
    mu = newton(fmu, val, args=(pot, gg))
    return mu


def thom_vs_comp_mu():
    pot = np.loadtxt("potential-generic.dat")
    en = np.loadtxt("energy-generic.dat")
    eint = np.loadtxt("eint-generic.dat")
    gg = np.loadtxt("gg-generic.dat")
    mu = np.zeros(len(en))

    for i in range(len(en)):
        mu[i] = calc_fmu((min(pot[i]) + max(pot[i]))/2, pot[i], gg[i])

    mu_num = en + eint # FROM DIRECT INTEGRATION

    return mu, mu_num



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

def calc_tf_ntf(pot_type):
    if pot_type == None:
        pot_type = "random"
    
    cwd = os.getcwd()
    os.chdir("../../data/nonlinearSE/generic_dataset_TF/" + pot_type + "/")
    pot = np.loadtxt("potential-generic.dat")
    en = np.loadtxt("energy-generic.dat")
    gg = np.loadtxt("gg-generic.dat")
    dens = np.loadtxt("dens-generic.dat")
    mu = calc_fmu((min(pot) + max(pot))/2, pot, gg)
    #mu = np.array([calc_fmu((min(pot[i]) + max(pot[i]))/2, pot[i], gg[i]) for i in range(len(pot))])
    n_tf = ptl_num(mu, pot, gg)
    #n_tf = np.array([ptl_num(mu[i], pot[i], gg[i]) for i in range(len(pot))])

    fig, ax1 = plt.subplots()
    ax1.plot(x, pot, "black")
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('Potential')
    ax1.set_ylim(0, None)
    ax1.tick_params('$y$')
    #ax1.tick_params(labelsize = 18)

    ax2 = ax1.twinx()
    ax2.plot(x, dens, label = "XMDS")
    ax2.plot(x, n_tf, label = "TF")
    ax2.set_ylabel('$|\psi(x)|^2$')
    ax2.tick_params('$y$')
    ax2.yaxis.set_label_position("right")

    ax2.set_ylim(0, None)
    #ax2.tick_params(labelsize = 18)
    fig.tight_layout()
    plt.legend()

    file_path = "../../../../figs/numericanalyze/"  
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    plt.savefig(file_path + "thomas-fermi-{}-{}".format(pot_type, gg) + ".png", format = "png", bbox_inches='tight') #, dpi=800)    
    #plt.show()
    plt.clf()
    os.chdir(cwd)

def variation(g):
    cf = 1 / (2 * np.sqrt(2 * np.pi))
    f = lambda a, g: (a**2) + (cf * g * a**(3/2)) - 1/2
    en_f = lambda a, g: a/2 + 1/(2*a) + cf * g * a**(1/2)
    sols = np.array([float(spo.fsolve(f, 1.1, args=(arg))) for arg in g])
    en_var = en_f(sols, g)
    return en_var

def thomas_fermi_en(mu, g):
    cf = (20 * np.sqrt(2)) / (21 * g)
    return cf * mu**(5/2)



#calc_tf_ntf()