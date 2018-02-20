import shlex, subprocess, time
import random as rnd
import time as time
import numpy as np

#parser = argparse.ArgumentParser(description='Fully Connected FeedForwardNetwork for nonlinearSE')
#
#parser.add_argument('--interaction-param',      type=float,           default=0.0,        help = 'input batch size for training (default: 64)')
#parser.add_argument('--num_particles',          type=int,             default=1,          help = 'Number of Particles (default: 1)')
#parser.add_argument('--frequency',              type=float,           default=1.0,        help = 'display progress (default:False)')
#parser.add_argument('--shift',                  type=float,           default=0.0,        help = 'display progress (default:False)')
#
#
#args = parser.parse_args()

inter_param = 20
inter_params = [1, 20]
num_particles = 1
freq = 1.  # corresponds to omega in xmds file and it will change between [0.5, 2]
shift = 0   # shift [-10, 10]

N_of_examples = 200000  #ten thousand

rnd.seed()

np.random.seed(1234)
inter_params=np.array([0.5])
inter_params=np.random.uniform(0.,10.,N_of_examples)


#print(freq)
#print(shift)

start = time.time()

for inter_param in inter_params:
  #for i in range(N_of_examples):

    #shift = rnd.random() * rnd.randint(-5, 5)
    #freq = rnd.uniform(0.5, 2)
    shift = np.random.uniform(-5,5)
    #freq = np.random.uniform(0.5,2)

    cmdline = "./xgp1d --interaction_param={} --num_particles={} --freq={} --shift={}".format(inter_param, num_particles, freq, shift)
    args = shlex.split(cmdline)
    p = subprocess.Popen(args)
    p.wait()

    #cmdline = "python2.7 gp1d_auto.py --pos-file-ex=-g_{}_.dat".format(inter_param)
    cmdline = "python2.7 gp1d_auto.py --pos-file-ex=-var_g_var_freq_.dat"
    args = shlex.split(cmdline)
    p = subprocess.Popen(args)
    p.wait()

print("Total Time = {}".format(time.time() - start))


#cmdline = "./xgp1d"
#cmdline = "./xgp1d --interaction_param=0 --num_particles=1 --freq=0.5 --shift=10"
