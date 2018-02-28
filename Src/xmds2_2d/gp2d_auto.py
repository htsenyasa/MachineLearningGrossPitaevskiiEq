#!/usr/bin/env python
from xpdeint.XSILFile import XSILFile
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cPickle as pickle

parser = argparse.ArgumentParser(description='xmds data')
parser.add_argument('--pos-file-ex',         type=str,             default="-test.dat",                    help = 'target file ex')
parser.add_argument('--dir',                 type=str,             default="",                             help = 'target file ex')
parser.add_argument('--pot',                 type=str,             default="0",                              help = 'target file ex')


args = parser.parse_args()

file_ex = args.pos_file_ex
dire = args.dir
pot = args.pot
root = "../../data/nonlinearSE/generic_dataset_2/" + dire + "/"
pot_file_name = root + "potential" + file_ex
en_file_name = root + "energy" + file_ex
ekin_file_name = root + "ekin" + file_ex
epot_file_name = root + "epot" + file_ex
eint_file_name = root + "eint" + file_ex
dens_file_name = root + "dens" + file_ex


xsilFile = XSILFile("gp1d.xsil")

def firstElementOrNone(enumerable):
  for element in enumerable:
    return element
  return None

t_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].independentVariables if _["name"] == "t")
x_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].independentVariables if _["name"] == "x")
dens_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "dens")
phiR_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "phiR")
phiI_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "phiI")
t_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].independentVariables if _["name"] == "t")
norm_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "norm")
e1_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "e1")
e1kin_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "e1kin")
e1pot_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "e1pot")
e1int_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "e1int")
vir1_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "vir1")
mu1_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "mu1")
x_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].independentVariables if _["name"] == "x")
v1_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].dependentVariables if _["name"] == "v1")


v1_3 = v1_3.reshape(1, len(v1_3))
dens = dens_1[-1].T
dens = dens.reshape(1, len(dens))

f = open(pot_file_name, "a")
f2 = open(en_file_name, "a")
f3 = open(ekin_file_name, "a")
f4 = open(epot_file_name, "a")
f5 = open(eint_file_name, "a")
f6 = open(dens_file_name, "a")

#pickle.dump(v1_3, f, pickle.HIGHEST_PROTOCOL)
#pickle.dump(e1_2, f2, pickle.HIGHEST_PROTOCOL)
#pickle.dump(e1kin_2, f3, pickle.HIGHEST_PROTOCOL)
#pickle.dump(e1pot_2, f4, pickle.HIGHEST_PROTOCOL)
#pickle.dump(e1int_2, f5, pickle.HIGHEST_PROTOCOL)
#pickle.dump(dens, f6, pickle.HIGHEST_PROTOCOL)


np.savetxt(f, v1_3)
np.savetxt(f2, e1_2[-2:-1])
np.savetxt(f3, e1kin_2[-2:-1])
np.savetxt(f4, e1pot_2[-2:-1])
np.savetxt(f5, e1int_2[-2:-1])
np.savetxt(f6, dens)
