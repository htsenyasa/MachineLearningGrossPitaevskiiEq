#!/usr/bin/env python
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
e1kin_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "e1kin")
e1pot_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "e1pot")
e1int_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "e1int")
vir1_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "vir1")
mu1_2 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[1].dependentVariables if _["name"] == "mu1")
x_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].independentVariables if _["name"] == "x")
v1_3 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[2].dependentVariables if _["name"] == "v1")

<<<<<<< HEAD:Src/xmds2/gp1d_auto.py


v1_3 = v1_3.reshape(1, len(v1_3))
f = open("/home/user/Desktop/Study/Src/APPL/data/nonlinearSE/potential-g-1.dat", "a")
f2 = open("/home/user/Desktop/Study/Src/APPL/data/nonlinearSE/energy-g-1.dat", "a")
np.savetxt(f, v1_3)
np.savetxt(f2, e1_2[-2:-1])



#np.savetxt(f, np.concatenate((v1_3, e1_2[-2:-1])).reshape(1, np.size(v1_3)+1))

#plt.plot(x_1, dens_1[-1,:], '.')
#plt.plot(x_1, v1_3)

#fig, ax = plt.subplots(1,2,figsize=(18,6))
#ax[0].plot(x_1,dens_1[-1,:],'b.-', x_1, v1_3[:]/100., 'k-')
=======
# Write your plotting commands here.
# You may want to import pylab (from pylab import *) or matplotlib (from matplotlib import *)
>>>>>>> 1260355ec53faf50ce8a401f59c308c25036dc09:Src/xmds2/gp1d.py

