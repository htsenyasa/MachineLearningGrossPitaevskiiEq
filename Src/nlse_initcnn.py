import shlex, subprocess, time
import time

dirs = ["harmonic", "well", "gaussian", "random", "random2", "random3"]

dirs = ["gaussian", "random"]
dirs = ["gaussian"]

root =  "generic_dataset_MAIN/"
data = "pot_inter-comb.dat.npy"
label = "en-generic-comb.dat"

def inter_init(load_state = False):
    info_file_name = "../figs/training/interaction/" + "internew2.inf"
    inter_data_filename   = root + "pot_dens-inter.dat.npy"
    inter_label_filename  = root + "gg-generic-inter.dat"
    save_state_file = "/home/user/Study/Src/APPL/network/states/inter-state-new.sf"
    cmdline = "python nlse_cnnnetwork1d-inter.py  --display-progress --data-filename={} --label-filename={} --info-file={} --save-state-file={}".format(inter_data_filename, inter_label_filename, info_file_name, save_state_file)
    if load_state == True:
        load_state_file = "/home/user/Study/Src/APPL/network/states/inter-state-new.sf"
        cmdline += " --load-state --load-state-file={}".format(load_state_file)
    args = shlex.split(cmdline)
    p = subprocess.Popen(args)
    p.wait()    


def normal_init():
    for dir in dirs:
        print(dir)
        data_filename   = root + dir    + "/" + data
        label_filename  = root + dir    + "/" + label
        info_file_name = "../figs/training/" + "{}.inf".format(dir)
        cmdline = "python nlse_cnnnetwork1d.py --data-filename={} --label-filename={} --info-file={}".format (data_filename, label_filename, info_file_name)
        args = shlex.split(cmdline)
        p = subprocess.Popen(args)
        p.wait()

def comb_init():
    data_filename   = root + data
    label_filename  = root + label
    info_file_name = "../figs/training/combined/" + "combined.inf"
    cmdline = "python nlse_cnnnetwork1d.py --display-progress --data-filename={} --label-filename={} --info-file={}".format (data_filename, label_filename, info_file_name)
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
            info_file_name = dir + "-" + cr_dir + ".inf"

            cmdline = "python nlse_cnnnetwork1d.py --cross-test --data-filename={} --label-filename={} --data-filename2={} --label-filename2={} --info-file={}".format  (data_filename, label_filename, data_filename2, label_filename2, info_file_name)
            args = shlex.split(cmdline)
            p = subprocess.Popen(args)
            p.wait()
start = time.time()
#comb_init()
#cross_init()
inter_init()
end = time.time()

print("Total Time = {}".format(end - start))

