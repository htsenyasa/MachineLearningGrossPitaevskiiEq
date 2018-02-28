#!/usr/bin/env python
import numpy as np
import argparse
import matplotlib.pyplot as plt
#import cPickle as pickle
import sys

sys.path.append("/home/user/Desktop/")
from xpdeint.XSILFile import XSILFile


parser = argparse.ArgumentParser(description='xmds data')
parser.add_argument('--pos-file-ex',         type=str,             default="-test.dat",                    help = 'target file ex')
parser.add_argument('--dir',                 type=str,             default="",                             help = 'target file ex')
parser.add_argument('--pot',                 type=str,             default="0",                              help = 'target file ex')


args = parser.parse_args()

file_ex = args.pos_file_ex
dire = args.dir
pot = args.pot
root = "../../data/nonlinearSE/generic_dataset_2d/" + dire + "/"
pot_file_name = root + "potential" + file_ex
en_file_name = root + "energy" + file_ex
ekin_file_name = root + "ekin" + file_ex
epot_file_name = root + "epot" + file_ex
eint_file_name = root + "eint" + file_ex
dens_file_name = root + "dens" + file_ex


xsilFile = XSILFile("gp2d.xsil")

def firstElementOrNone(enumerable):
  for element in enumerable:
    return element
  return None

t_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].independentVariables if _["name"] == "t")
Nptls_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "Nptls")
t_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].independentVariables if _["name"] == "t")
x_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].independentVariables if _["name"] == "x")
y_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].independentVariables if _["name"] == "y")
Pot_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "Pot")
t_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].independentVariables if _["name"] == "t")
x_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].independentVariables if _["name"] == "x")
y_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].independentVariables if _["name"] == "y")
dens_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].dependentVariables if _["name"] == "dens")

# Write your plotting commands here.
# You may want to import pylab (from pylab import *) or matplotlib (from matplotlib import *)

f = open(pot_file_name, "ab")
pot = np.array(Pot_2)
np.save(f, Pot_2[-1])