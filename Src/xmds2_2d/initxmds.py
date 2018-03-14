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

    return l_1, l_2, mu_1, mu_2, s_1, s_2 #use this order

def get_infwell_params():

    lc = rnd.uniform(-5, 5) # left_well
    lw = rnd.uniform(1, 8) # left_well
    uc = rnd.uniform(-5, 5) # upper_well
    uw = rnd.uniform(1, 8)  # upper_well

    return lc-lw/2, lc+lw/2, uc-uw/2, uc+uw/2

low = 0
inc = 0.5
high = 4

num_particles = 1
freq = 1.  # corresponds to omega in xmds file and it will change between [0.5, 2]
shift = 0   # shift [-10, 10]

N_of_ex = 1
N_of_ex_g = 1
rnd.seed()

dirs = ["harmonic", "well", "gaussian"]

rnd.seed(1)
inter_params = [rnd.uniform(0., 10.) for i in range(N_of_ex)]
pot_types = [0, 1, 2] # 0:Harmonic, 1:Infinite Well 2:Double Inverted Gaussian
pot_types = [0]

start = time.time()

for pot in pot_types:
    for inter_param in inter_params:
        #for i in range(N_of_ex_g):

        xshift = rnd.uniform(-5, 5)
        yshift = rnd.uniform(-5, 5)
        xfreq = rnd.uniform(0.1, 2)
        yfreq = rnd.uniform(0.1, 2)

        cmdline = "./xgp2d --interaction={} --pottype={} ".format(inter_param, pot)

        if (pot == 0):
            cmdline += "--wx={} --wy={} --xshift={} --yshift={} ".format(xfreq, yfreq, xshift, yshift)
        elif (pot == 1):
          cmdline += "--lw={} --rw={} --bw={} --uw={} ".format(*get_infwell_params())
        elif (pot == 2):
          cmdline += "--xlam1={} --xlam2={} --xmu1={} --xmu2={} --xs1={} --xs2={} ".format(*get_gaussian_params())
          cmdline += "--ylam1={} --ylam2={} --ymu1={} --ymu2={} --ys1={} --ys2={} ".format(*get_gaussian_params())

        print(cmdline)

        args = shlex.split(cmdline)
        p = subprocess.Popen(args)
        p.wait()

        cmdline = "python gp2d__.py --dir={}".format(dirs[pot])
        args = shlex.split(cmdline)
        p = subprocess.Popen(args)
        p.wait()


print("Total Time = {}".format(time.time() - start))


