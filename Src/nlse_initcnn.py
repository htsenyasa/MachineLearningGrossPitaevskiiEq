import shlex, subprocess, time

dirs = ["harmonic", "well", "gaussian", "random", "random2", "random3"]

dirs = ["gaussian", "random"]

root =  "generic_dataset_MAIN/"
data = "pot_inter.dat.npy"
label = "energy-generic.dat"


def normal_init():
    for dir in dirs:
        print(dir)
        data_filename   = root + dir    + "/" + data
        label_filename  = root + dir    + "/" + label
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

            cmdline = "python nlse_cnnnetwork1d.py --display-progress --cross-test --data-filename={} --label-filename={} --data-filename2={} --label-filename2={} --info-file={}".format  (data_filename, label_filename, data_filename2, label_filename2, dir + "-" + cr_dir)
            args = shlex.split(cmdline)
            p = subprocess.Popen(args)
            p.wait()

cross_init()