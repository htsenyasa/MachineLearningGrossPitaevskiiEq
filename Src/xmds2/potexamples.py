import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


import potentialgenerator as pg

x = np.arange(-10, 10, 20/128)
potgen = pg.PotentialGenerator(seed = 10)

def env_example():
    env = potgen.create_envolope_pot(1)
    plt.plot(x, env, label = r"$\beta = 1$")
    env = potgen.create_envolope_pot(5)
    plt.plot(x, env, label = r"$\beta = 5$")
    env = potgen.create_envolope_pot(30)
    plt.plot(x, env, label = r"$\beta = 30$")
    plt.xlabel("x", fontsize = 20)
    plt.ylabel("Amplitude", fontsize = 20)
    plt.legend()
    plt.tick_params(labelsize = 18)
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    pltname = "../../figs/potentials/" + "env" + ".png"
    plt.savefig(pltname, format = "png", bbox_inches='tight')
    #plt.show()

def save_plot(x, pot, type, posfix):
    plt.plot(x, pot)
    plt.xlabel("x", fontsize = 20)
    plt.ylabel("Amplitude", fontsize = 20)
    plt.legend()
    plt.tick_params(labelsize = 18)
    figure = plt.gcf()
    figure.set_size_inches(8,6)
    pltname = "../../figs/potentials/" + "random-{}-".format(type) + posfix + ".png"
    plt.savefig(pltname, format = "png", bbox_inches='tight')
    plt.clf()
    #plt.show()

dirs = ["random1", "random2", "random3"]

pot_types = [1, 2, 3]    # 0:Harmonic, 1:Infinite Well 2:Double Inverted Gaussian 3:Random 4:Random2 5:Random5


random_pot_generators = [potgen.generate_random_pot, 
                         potgen.generate_random_pot_2, 
                         potgen.generate_random_pot_3]    

rpots = []
prpots = []
for i, pot in enumerate(random_pot_generators):
    rpot, prpot = pot()
    rpots.append(rpot)
    prpots.append(prpot)
    save_plot(x, rpot, pot_types[i], "noproc")
    save_plot(x, prpot, pot_types[i], "proc")

