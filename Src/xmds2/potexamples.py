import random as rnd

import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import rc
import matplotlib

plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
          'font.size' : 11,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams.update(params) 

matplotlib.rc('axes', titlesize=20)
matplotlib.rc('axes', labelsize=24)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=22)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=22)    # fontsize of the tick labels

import potentialgenerator as pg

x = np.arange(-10, 10, 20/256)
potgen = pg.PotentialGenerator(seed = 99)

def env_example():
    env, a1a2 = potgen.create_envolope_pot(1)
    plt.plot(x, env, label = r"$\beta = 1$")
    env = potgen.create_envolope_pot(5)
    plt.plot(x, env, label = r"$\beta = 5$")
    env = potgen.create_envolope_pot(30)
    plt.plot(x, env, label = r"$\beta = 30$")
    plt.xlabel("$x$", fontsize = 20)
    plt.ylabel("Amplitude", fontsize = 20)
    plt.legend()
    plt.tick_params(labelsize = 18)
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    pltname = "../../figs/potentials/" + "env" + ".png"
    #plt.savefig(pltname, format = "png", bbox_inches='tight')
    plt.show()

def env_example_2():
    env, a1a2 = potgen.create_envolope_pot(4)
    plt.plot(x, a1a2, label = r"$\text{Env}_{LR}$, $\beta$ = 4$")
    #env, a1a2 = potgen.create_envolope_pot(5)
    plt.plot(x, env, label = r"$\text{Env}_{M}$, $\beta$ = 4$")
    #env, a1a2 = potgen.create_envolope_pot(30)
    #plt.plot(x, a1a2, label = r"$\beta = 30$")
    plt.xlabel("$x$", fontsize = 20)
    plt.ylabel("Amplitude", fontsize = 20)
    plt.legend()
    plt.tick_params(labelsize = 18)
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    pltname = "../../figs/potentials/" + "env" + ".png"
    plt.savefig(pltname, format = "png", bbox_inches='tight')
    #plt.show()

def raw_pot_and_proc():
    Np = 256
    points = np.array([rnd.gauss(0, 1) for i in range(Np)])
    pot = np.zeros(Np)
    pot[0] = points[0]

    for i in range(Np - 1):
        pot[i+1] = pot[i] + points[i]

    sigma = 3
    procs = [scipy.ndimage.filters.gaussian_filter1d]
    args = [(pot, sigma)]

    tpot = potgen.potential_process(pot, procs, args)
    a3, a1a2 = potgen.create_envolope_pot(4)

    file_path = "../../figs/potentials/"
    fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(x, pot, label = "Before Process")
    #ax1.plot(x, tpot)
    #ax1.set_ylim(0, 50)
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('Potential')
    ax1.set_ylim(0, None)
    ax1.tick_params('$y$')
    #ax1.tick_params(labelsize = 18)

    ax2 = ax1.twinx()
    ax2.plot(x, a3, 'r')
    ax2.plot(x, a1a2, 'r')
    #ax2.set_ylabel('Amplitude', fontsize = 16)
    ax2.tick_params('$y$')
    ax2.set_ylim(0, None)
    #ax2.tick_params(labelsize = 18)

    #ax3 = plt.subplot(122)
    ax3.plot(x, tpot, label = "After Process")
    #ax3.plot(x, tpot)
    ax3.set_xlabel('$x$')
    #ax3.set_ylabel('Potential', fontsize = 16)
    ax3.set_ylim(0, None)
    ax3.tick_params('$y$')
    #ax3.tick_params(labelsize = 18)
    plt.legend()
    figure = plt.gcf()
    figure.set_size_inches(16,6)
    fig.tight_layout()
    #plt.show()
    plt.savefig(file_path + "randombeforeafterproc" + ".png", format = "png", bbox_inches='tight') #, dpi=800)    


def before_after_pot(x, bpot, apot, type, posfix):
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(x, bpot, label = "Before Process")
    ax1.set_xlabel('$x$')
    ax1.set_ylabel('Potential')
    #ax1.set_ylim(0, None)
    ax1.tick_params('$y$')
    #ax1.tick_params(labelsize = 18)

    ax2.plot(x, apot, label = "After Process")
    ax2.set_xlabel('$x$')
    #ax2.set_ylabel('Potential')
    #ax2.set_ylim(0, None)
    ax2.tick_params('$y$')
    #ax2.tick_params(labelsize = 18)

    plt.legend()
#    plt.tick_params(labelsize = 18)
    figure = plt.gcf()
    figure.set_size_inches(16,6)
    fig.tight_layout()
    pltname = "../../figs/potentials/" + "random-{}-".format(type) + posfix + ".png"
    #plt.show()
    plt.savefig(pltname, format = "png", bbox_inches='tight')
    plt.clf()

dirs = ["random1", "random2", "random3"]

pot_types = [1, 2, 3]    # 0:Harmonic, 1:Infinite Well 2:Double Inverted Gaussian 3:Random 4:Random2 5:Random5


random_pot_generators = [potgen.generate_random_pot, 
                         potgen.generate_random_pot_2, 
                         potgen.generate_random_pot_3]    

#rpots = []
#prpots = []
#for i, pot in enumerate(random_pot_generators):
#    rpot, prpot = pot()
#    rpots.append(rpot)
#    prpots.append(prpot)
#    before_after_pot(x, rpot, prpot, pot_types[i], "noprocandproc")

raw_pot_and_proc()