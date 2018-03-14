#!/usr/bin/env python
from xpdeint.XSILFile import XSILFile
import matplotlib.pyplot as plt
import numpy as np

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

# Write your plotting commands here.
# You may want to import pylab (from pylab import *) or matplotlib (from matplotlib import *)


plt.plot(dens_1[-1])
plt.show()

#f = open("position.txt", "a")
#f2 = open("potential.txt", "a")

#v1_3 = v1_3.reshape(1, len(v1_3))
#dens = dens_1[-1].T
#dens = dens.reshape(1, len(dens))

#np.savetxt(f, dens)
#np.savetxt(f, x_1)
#plt.figure("density func")
#plt.plot(x_1, dens_1[-1,:], label = "Density")
##plt.plot(x_1, v1_3, label = "Potential")
#plt.legend()
#plt.xlabel("$z$", fontsize = 20)
#plt.ylabel("$|\phi|^2$", fontsize = 20)
##plt.figure("density func, imag part")
##plt.plot(x_1, phiR_1[-1,:])
##plt.figure("density func, real part")
##plt.plot(x_1, phiI_1[-1,:])
#plt.savefig("potvsdens-g-20.svg", format = "svg", dpi=1200)
##plt.show()
