import shlex, subprocess, time
import random as rnd
import time as time
import argparse

import numpy as np
import matplotlib.pyplot as plt

import potentialgenerator as pg

parser = argparse.ArgumentParser(description='XMDS initializer')
parser.add_argument('--examples',            type=int,             default=10,      help = 'Number of examples to solve (Default: 10)')
parser.add_argument('--pot-type',            type=int,             default=-1,      help = 'Type of potential 0:Harmonic, 1:Well, 2: DI Gaussian 3 & 4 & 5: Random (Default : -1:all)')

args = parser.parse_args()

pot_type = args.pot_type
N_of_ex = args.examples

dirs = ["harmonic", "well", "gaussian", "random", "random2", "random3"]

pot_types = [0, 1, 2, 3, 4, 5]    # 0:Harmonic, 1:Infinite Well 2:Double Inverted Gaussian 3:Random 4:Random2 5:Random5
if pot_type != -1:
    pot_types = [pot_type]
#pot_types = [2]

start = time.time()

pot_gen = pg.PotentialGenerator(g_exec_func = pg.save_as_h5)
#pot_gen = pg.PotentialGenerator(g_exec_func = pg.display_pot)
pot_generators = [pot_gen.generate_harmonic_pot,
                  pot_gen.generate_well_pot, 
                  pot_gen.generate_gaussian_pot, 
                  pot_gen.generate_random_pot, 
                  pot_gen.generate_random_pot_2, 
                  pot_gen.generate_random_pot_3]

alpha = 0.5
#pot_generators[pot_types[0]]()
for pot_type in pot_types:
    inter_params = np.array([i for i in range(N_of_ex)])
    for inter_param in inter_params:

        pot_generators[pot_type]()

        cmdline = "./xgp1d --interaction_param={} --alpha={}".format(inter_param, alpha)
        args = shlex.split(cmdline)
        p = subprocess.Popen(args)
        p.wait()

        cmdline = "python gp1d_auto.py --pos-file-ex=-generic.dat --dir={}".format(dirs[pot_type])
        args = shlex.split(cmdline)
        p = subprocess.Popen(args)
        p.wait()

np.savetxt("inter.txt", inter_params)

end = time.time()

print("Total Time = {}".format(end - start))

