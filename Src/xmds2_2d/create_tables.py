import numpy as np
import argparse
import tables as tb

parser = argparse.ArgumentParser(description='xmds data')
parser.add_argument('--pos-file-ex',         type=str,             default=".h5",                    help = 'target file ex')
parser.add_argument('--dir',                 type=str,             default="",                             help = 'target file ex')

args = parser.parse_args()
file_ex = args.pos_file_ex
dire = args.dir

root = "../../data/nonlinearSE/generic_dataset_2d/harmonic" + dire + "/"
pot_file_name = root + "potential" + file_ex
en_file_name = root + "energy" + file_ex
ekin_file_name = root + "ekin" + file_ex
epot_file_name = root + "epot" + file_ex
eint_file_name = root + "eint" + file_ex
dens_file_name = root + "dens" + file_ex
gg_file_name = root + "gg" + file_ex

row_s = 128
col_s = 128
features = 5


f = tb.open_file(pot_file_name, "a")
f2 = tb.open_file(dens_file_name, "a")
f3 = tb.open_file(root + "features.h5", "a")

atom = tb.Float64Atom()

# features = [GG, Etot, Ekin, Epot, Eint]

f.create_earray(f.root, 'data', atom, (0, row_s, col_s))
f2.create_earray(f2.root, 'data', atom, (0, row_s, col_s))
f3.create_earray(f3.root, 'data', atom, (0, features))


f.close()
f2.close()
f3.close()