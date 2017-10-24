#!/usr/bin/env python2.7

import sys
sys.path.insert(0, "/home/user/Desktop/Study/xmds/xmds-2.2.3/xpdeint")
import matplotlib.pyplot as plt
import numpy as np

from xpdeint.XSILFile import XSILFile

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
x_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].independentVariables if _["name"] == "x")
v1_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].dependentVariables if _["name"] == "v1")


f = open("potentials.dat", "a")
f2 = open("energies.dat", "a")
np.savetxt(f, v1_3)
np.savetxt(f2, e1_2[-2:-1])


#np.savetxt(f, np.concatenate((v1_3, e1_2[-2:-1])).reshape(1, np.size(v1_3)+1))

#plt.plot(x_1, dens_1[-1,:], '.')
#plt.plot(x_1, v1_3)

#fig, ax = plt.subplots(1,2,figsize=(18,6))
#ax[0].plot(x_1,dens_1[-1,:],'b.-', x_1, v1_3[:]/100., 'k-')

#plt.show()
print e1_2[-2:-1]
#print len(v1_3)
#print v1_3
