import shlex, subprocess, time
import random as rnd
import time as time


inter_params = [0, 0.1, 1, 10, 20]
epoch = 60

net_cnn = "cnnnetwork1d.py"
net_ffn = "feedforwardnetwork2.py"


start = time.time()
for i in range(len(inter_params)):

    arguments = "--inter-param={} --epochs={}".format(inter_params[i], epoch)

    cmdline = "python {} {}".format(net_ffn, arguments)
    args = shlex.split(cmdline)
    p = subprocess.Popen(args)
    p.wait()

#    cmdline = "python {} {}".format(net_cnn, arguments)
#    args = shlex.split(cmdline)
#    p = subprocess.Popen(args)
#    p.wait()

print("Total Time = {}".format(time.time() - start))
