import shlex, subprocess, time
import random as rnd
import time as time

inter_param = 0
num_particles = 1
freq = 1.5  # corresponds to omega in xmds file and it will change between [0.5, 2]
shift = 0   # shift [-10, 10]

rnd.seed()

#print(freq)
#print(shift)

start = time.time()

for i in range(1000):

    shift = rnd.random() * rnd.randint(-10, 10)
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
