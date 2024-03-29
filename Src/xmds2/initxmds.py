import shlex, subprocess, time
import random as rnd
import time as time
import numpy as np

import genrandompot as grp

def get_gaussian_params():
  l_1 = rnd.uniform(1, 10)
  mu_1 = rnd.uniform(-5, 5)
  s_1 = rnd.uniform(low + inc, high - (high / 7 * np.abs(mu_1)))
  l_2 = rnd.uniform(1, 10)
  mu_2 = rnd.uniform(-5, 5)
  s_2 = rnd.uniform(low + inc, high - (high / 7 * np.abs(mu_2)))
  return l_1, l_2, mu_1, mu_2, s_1, s_2 #use this order

def get_infwell_params():
  lc = rnd.uniform(-5, 5) # left_well
  lw = rnd.uniform(1, 8) # left_well
  return lc-lw/2, lc+lw/2

low = 0
inc = 0.5
high = 4

num_particles = 1
freq = 1.  # corresponds to omega in xmds file and it will change between [0.5, 2]
shift = 0   # shift [-10, 10]

N_of_ex = 10
N_of_ex_g = 1
rnd.seed()

dirs = ["harmonic", "well", "gaussian", "random"]

rnd.seed(34)
inter_params = np.array([rnd.uniform(0, 10) for i in range(N_of_ex)])
inter_params = np.array([0, 0.1, 1, 10, 20, 100])
pot_types = [0, 1, 2, 3] # 0:Harmonic, 1:Infinite Well 2:Double Inverted Gaussian 3:Random
pot_types = [3]

start = time.time()

for pot in pot_types:
    for inter_param in inter_params:

        shift = rnd.uniform(-5, 5)
        freq = rnd.uniform(0.1, 2)
        
        cmdline = "./xgp1d --interaction_param={} --num_particles={} --freq={} --shift={} --pot_type={} ".format(inter_param, num_particles, freq, shift, pot)

        if (pot == 1):
            cmdline += "--lw={} --rw={} ".format(*get_infwell_params())
        elif (pot == 2):
            cmdline += "--lam1={} --lam2={} --mu1={} --mu2={} --s1={} --s2={} ".format(*get_gaussian_params())
        elif (pot == 3):
            cmdline += ""
            grp.generate_random_pot(exec_func=grp.save_as_h5)            

        args = shlex.split(cmdline)
        p = subprocess.Popen(args)
        p.wait()

        cmdline = "python gp1d_auto.py --pos-file-ex=-generic.dat --dir={}".format(dirs[pot])
        args = shlex.split(cmdline)
        p = subprocess.Popen(args)
        p.wait()


print("Total Time = {}".format(time.time() - start))


