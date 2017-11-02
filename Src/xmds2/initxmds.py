import shlex, subprocess, time
import random as rnd
import time as time

#parser = argparse.ArgumentParser(description='Fully Connected FeedForwardNetwork for nonlinearSE')
#
#parser.add_argument('--interaction-param',      type=float,           default=0.0,        help = 'input batch size for training (default: 64)')
#parser.add_argument('--num_particles',          type=int,             default=1,          help = 'Number of Particles (default: 1)')
#parser.add_argument('--frequency',              type=float,           default=1.0,        help = 'display progress (default:False)')
#parser.add_argument('--shift',                  type=float,           default=0.0,        help = 'display progress (default:False)')
#
#
#args = parser.parse_args()

inter_param = 0
num_particles = 1
freq = 1.5  # corresponds to omega in xmds file and it will change between [0.5, 2]
shift = 5   # shift [-10, 10]

N_of_examples = 4000

rnd.seed()

#print(freq)
#print(shift)

start = time.time()

for i in range(N_of_examples):

    shift = rnd.random() * rnd.randint(-shift, shift)
    freq = rnd.uniform(0.5, 2)

    cmdline = "./xgp1d --interaction_param={} --num_particles={} --freq={} --shift={}".format(inter_param, num_particles, freq, shift)
    args = shlex.split(cmdline)
    p = subprocess.Popen(args)
    p.wait()

    cmdline = "python2.7 gp1d.py"
    args = shlex.split(cmdline)
    p = subprocess.Popen(args)
    p.wait()

print("Total Time = {}".format(time.time() - start))


#cmdline = "./xgp1d"
#cmdline = "./xgp1d --interaction_param=0 --num_particles=1 --freq=0.5 --shift=10"
