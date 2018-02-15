#!/usr/bin/env python2.7
from xpdeint.XSILFile import XSILFile
import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import math as m

xsilFile = XSILFile("grid.xsil")

num_grid = 128
domain_len = 20
ratio = num_grid/domain_len
lowest_pot = 0
inf_pot = 100

def firstElementOrNone(enumerable):
  for element in enumerable:
    return element
  return None

x_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].independentVariables if _["name"] == "x")
dummy_1 = firstElementOrNone(_["array"] for _ in xsilFile.xsilObjects[0].dependentVariables if _["name"] == "dummy")

rnd.seed(10)
lw = rnd.uniform(-5, -0.5) # left_well
rw = -lw # right_well
#lw_i = m.ceil(lw * ratio) + num_grid//2 #index of lw 
#rw_i = m.floor(rw * ratio) + num_grid//2 #index of rw

def f_x(x):
  return [not(lw < x[i] < rw) * 100 for i in range(len(x))]
  