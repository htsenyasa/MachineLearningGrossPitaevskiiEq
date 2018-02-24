import shlex, subprocess, time
import random as rnd
import time as time
import numpy as np
from multiprocessing import Pool


## Without Multiprocessing ##
#
#    real	1m22,375s
#    user	1m7,182s
#    sys	0m11,365s
#    100 Examples for three pot type
#
#############################

### With Multiprocessing ##
#
#    real	0m43,691s
#    user	1m28,001s
#    sys	0m12,482s
#    100 Examples for three pot type
#
############################

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


def generate_potential(pot_type):
  for inter_param in inter_params:
      shift = np.random.uniform(-5,5)
      freq = np.random.uniform(0.5,2)

      dirs = ["harmonic", "well", "gaussian"]

      cmdline = "./xgp1d --interaction_param={} --num_particles={} --freq={} --shift={} --pot_type={} ".format(inter_param, num_particles, freq, shift, pot_type)

      if (pot_type == 1):
        cmdline += "--lw={} --rw={} ".format(*get_infwell_params())
      elif (pot_type == 2):
        cmdline += "--lam1={} --lam2={} --mu1={} --mu2={} --s1={} --s2={} ".format(*get_gaussian_params())

      args = shlex.split(cmdline)
      p = subprocess.Popen(args)
      p.wait()

      cmdline = "python2.7 gp1d_auto.py --pos-file-ex=-generic.dat --dir={} --pot={}".format(dirs[pot_type], pot_type)
      args = shlex.split(cmdline)
      p = subprocess.Popen(args)
      p.wait()

if __name__ == '__main__':

    low = 0
    inc = 0.5
    high = 4

    num_particles = 1
    freq = 1.  # corresponds to omega in xmds file and it will change between [0.5, 2]
    shift = 0   # shift [-10, 10]

    N_of_ex = 2
    N_of_ex_g = 1
    rnd.seed()

    np.random.seed(34)
    inter_params = np.random.uniform(0.,10.,N_of_ex)
    pot_types = [0, 1, 2] # 0:Harmonic, 1:Infinite Well 2:Double Inverted Gaussian
    #pot_types = [1]

    start = time.time()
    with Pool(3) as p:
        print(p.map(generate_potential, pot_types))

    #for pot in pot_types:
    #    generate_potential(pot)

    print("Total Time = {}".format(time.time() - start))


