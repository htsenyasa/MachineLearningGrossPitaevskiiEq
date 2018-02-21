import shlex, subprocess, time
import random as rnd
import time as time
import numpy as np

def get_gaussian_params():
  l_1 = rnd.uniform(1, 10)
  mu_1 = rnd.uniform(-5, 5)
  s_1 = rnd.uniform(low + inc, high - (high / 7 * np.abs(mu_1)))

  l_2 = rnd.uniform(1, 10)
  mu_2 = rnd.uniform(-5, 5)
  s_2 = rnd.uniform(low + inc, high - (high / 7 * np.abs(mu_2)))
  
  return l_1, mu_1, s_1, l_2, mu_2, s_2 #use this order

def get_infwell_params():
  lw = rnd.uniform(-5, -0.5) # left_well
  return lw, -lw

low = 0
inc = 0.5
high = 4

num_particles = 1
freq = 1.  # corresponds to omega in xmds file and it will change between [0.5, 2]
shift = 0   # shift [-10, 10]

N_of_ex = 10
N_of_ex_g = 1
rnd.seed()

np.random.seed(1234)
inter_params = np.random.uniform(0.,10.,N_of_ex)
pot_types = [0, 1, 2] # 0:Harmonic, 1:Infinite Well 2:Double Inverted Gaussian
pot_types = [2]

start = time.time()
for pot in pot_types:
  for inter_param in inter_params:
    #for i in range(N_of_ex_g):

      shift = np.random.uniform(-5,5)
      freq = np.random.uniform(0.5,2)

      cmdline = "./xgp1d --interaction_param={} --num_particles={} --freq={} --shift={} --pot_type={}".format(inter_param, num_particles, freq, shift, pot)

      if (pot == 1):
        cmdline += "--lw={} --rw={}".format(*get_infwell_params())
      elif (pot == 2):
        cmdline += "--lam1={} --lam2={} --mu1={} --mu2={} --s1={} --s2={}".format(*get_gaussian_params())

      args = shlex.split(cmdline)
      p = subprocess.Popen(args)
      p.wait()

      #cmdline = "python2.7 gp1d_auto.py --pos-file-ex=-g_{}_.dat".format(inter_param)
      cmdline = "python2.7 gp1d_auto.py --pos-file-ex=-generic.dat"
      args = shlex.split(cmdline)
      p = subprocess.Popen(args)
      p.wait()


print("Total Time = {}".format(time.time() - start))


#cmdline = "./xgp1d"
#cmdline = "./xgp1d --interaction_param=0 --num_particles=1 --freq=0.5 --shift=10"
