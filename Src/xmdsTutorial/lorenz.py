#!/usr/bin/env python2.7
from xpdeint.XSILFile import XSILFile


xsilFile = XSILFile("lorenz.xsil")

def firstElementOrNone(enumerable):
  for element in enumerable:
    return element
  return None

t_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].independentVariables if _["name"] == "t")
xR_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "xR")
yR_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "yR")
zR_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "zR")

# Write your plotting commands here.
# You may want to import pylab (from pylab import *) or matplotlib (from matplotlib import *)
print (t_1[-1])
