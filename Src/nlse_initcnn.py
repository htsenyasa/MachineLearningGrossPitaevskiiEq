import shlex, subprocess, time
import time

dirs = ["harmonic", "well", "gaussian", "random", "random2", "random3"]

dirs = ["gaussian", "random"]
dirs = ["gaussian"]

root =  "generic_dataset_MAIN/"
data = "pot_inter-comb.dat.npy"
label = "en-generic-comb.dat"

def inter_init():
    dir = "inter"
    inter_data_filename   = root + "pot_dens-inter.dat.npy"
    inter_label_filename  = root + "gg-generic-inter.dat"
    cmdline = "python nlse_cnnnetwork1d-inter.py  --display-progress --data-filename={} --label-filename={} --info-file={}".format (inter_data_filename, inter_label_filename, dir)
    args = shlex.split(cmdline)
    p = subprocess.Popen(args)
    p.wait()    


def normal_init():
    for dir in dirs:
        print(dir)
        data_filename   = root + dir    + "/" + data
        label_filename  = root + dir    + "/" + label
        cmdline = "python nlse_cnnnetwork1d.py --data-filename={} --label-filename={} --info-file={}".format (data_filename, label_filename, dir)
        args = shlex.split(cmdline)
        p = subprocess.Popen(args)
        p.wait()

def comb_init():
    dir = "main"
    data_filename   = root + data
    label_filename  = root + label
    cmdline = "python nlse_cnnnetwork1d.py --display-progress --data-filename={} --label-filename={} --info-file={}".format (data_filename, label_filename, dir)
    args = shlex.split(cmdline)
    p = subprocess.Popen(args)
    p.wait()


def cross_init():
    for dir in dirs:
        for cr_dir in dirs:
            print(dir)
            print(cr_dir)
            if dir == cr_dir:
                continue
            data_filename   = root + dir    + "/" + data
            data_filename2  = root + cr_dir + "/" + data
            label_filename  = root + dir    + "/" + label
            label_filename2 = root + cr_dir + "/" + label

            cmdline = "python nlse_cnnnetwork1d.py --cross-test --data-filename={} --label-filename={} --data-filename2={} --label-filename2={} --info-file={}".format  (data_filename, label_filename, data_filename2, label_filename2, dir + "-" + cr_dir)
            args = shlex.split(cmdline)
            p = subprocess.Popen(args)
            p.wait()
start = time.time()
#comb_init()
#cross_init()
inter_init()
end = time.time()

print("Total Time = {}".format(end - start))

