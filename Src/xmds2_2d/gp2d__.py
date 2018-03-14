#!/usr/bin/env python

import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
import tables as tb
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

sys.path.append("/home/user/Desktop/")
from xpdeint.XSILFile import XSILFile


parser = argparse.ArgumentParser(description='xmds data')
parser.add_argument('--pos-file-ex',         type=str,             default=".h5",                    help = 'target file ex')
parser.add_argument('--dir',                 type=str,             default="",                             help = 'target file ex')


args = parser.parse_args()

file_ex = args.pos_file_ex
dire = args.dir
path = "../../data/nonlinearSE/generic_dataset_2d/" + dire + "/"
pot_file_name  = path + "potential" + file_ex
en_file_name   = path + "energy" + file_ex
ekin_file_name = path + "ekin" + file_ex
epot_file_name = path + "epot" + file_ex
eint_file_name = path + "eint" + file_ex
dens_file_name = path + "dens" + file_ex
gg_file_name   = path + "gg" + file_ex



xsilFile = XSILFile("gp2d.xsil")


def display_pot(Z):
    X = np.arange(-10, 10, 20/128)
    Y = np.arange(-10, 10, 20/128)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
    #plt.imshow(Z)
    plt.show()


def firstElementOrNone(enumerable):
  for element in enumerable:
    return element
  return None

t_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].independentVariables if _["name"] == "t")
Nptls_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "Nptls")
Ek_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "Ek")
Ep_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "Ep")
Ei_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "Ei")
En_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "En")
g0_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "g0")
t_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].independentVariables if _["name"] == "t")
x_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].independentVariables if _["name"] == "x")
y_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].independentVariables if _["name"] == "y")
Pot_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "Pot")
t_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].independentVariables if _["name"] == "t")
x_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].independentVariables if _["name"] == "x")
y_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].independentVariables if _["name"] == "y")
dens_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].dependentVariables if _["name"] == "dens")


#display_pot(Pot_2)
# Write your plotting commands here.
# You may want to import pylab (from pylab import *) or matplotlib (from matplotlib import *)

###features = [GG, Etot, Ekin, Epot, Eint]
features = np.array([g0_1[-1], En_1[-1], Ek_1[-1], Ep_1[-1], Ei_1[-1]])

print(features)
#f = tb.open_file(pot_file_name, "a")
#f2 = tb.open_file(dens_file_name, "a")
#f3 = tb.open_file(path + "features.h5", "a")
#
#pot = np.reshape(Pot_2, (1, 128, 128))
#dens = np.reshape(dens_3[-1], (1, 128, 128))
#features = np.reshape(features, (1, 5))
#
#f.root.data.append(pot)
#f2.root.data.append(dens)
#f3.root.data.append(features)
#
#f.close()
#f2.close()
#f3.close()
