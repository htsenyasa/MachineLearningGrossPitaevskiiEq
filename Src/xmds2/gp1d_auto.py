#!/usr/bin/env python
from xpdeint.XSILFile import XSILFile
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


v1_3 = v1_3.reshape(1, len(v1_3))
f = open("/home/user/Desktop/Study/Src/APPL/data/nonlinearSE/potential128.dat", "a")
f2 = open("/home/user/Desktop/Study/Src/APPL/data/nonlinearSE/energy128.dat", "a")
f3 = open("/home/user/Desktop/Study/Src/APPL/data/nonlinearSE/ekin128.dat", "a")
f4 = open("/home/user/Desktop/Study/Src/APPL/data/nonlinearSE/epot128.dat", "a")
f5 = open("/home/user/Desktop/Study/Src/APPL/data/nonlinearSE/eint128.dat", "a")

np.savetxt(f, v1_3)
np.savetxt(f2, e1_2[-2:-1])
np.savetxt(f3, e1kin_2[-2:-1])
np.savetxt(f4, e1pot_2[-2:-1])
np.savetxt(f5, e1int_2[-2:-1])
