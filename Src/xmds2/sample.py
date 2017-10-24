import shlex, subprocess, time


#cmdline = "./xgp1d"
cmdline = "./xgp1d --interaction_param=0 --num_particles=1 --freq=-0.5 --shift=10"
args = shlex.split(cmdline)
print(args)
p = subprocess.Popen(args)
p.wait()

cmdline = "python2.7 gp1d.py"
args = shlex.split(cmdline)
p = subprocess.Popen(args)
p.wait()
