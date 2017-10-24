import shlex, subprocess, time
import random as rnd


inter_param = 0
num_particles = 1
freq = 1.5  # corresponds to omega in xmds file and it will change between [0, 1.7]
shift = 0   # shift [-10, 10]

#rnd.seed(10)
shift = rnd.triangular(-10, 10)
freq = rnd.triangular(0, 1.7)

cmdline = "./xgp1d --interaction_param={} --num_particles={} --freq={} --shift={}".format(inter_param, num_particles, freq, shift)
args = shlex.split(cmdline)
p = subprocess.Popen(args)
p.wait()

cmdline = "python2.7 gp1d.py"
args = shlex.split(cmdline)
p = subprocess.Popen(args)
p.wait()


#cmdline = "./xgp1d"
#cmdline = "./xgp1d --interaction_param=0 --num_particles=1 --freq=0.5 --shift=10"
