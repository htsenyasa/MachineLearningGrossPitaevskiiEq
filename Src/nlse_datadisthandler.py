import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
from math import ceil

#path = "generic_dataset_MAIN/gaussian/"
#
#X = np.load(path + "pot_inter.dat.npy")
#y = np.loadtxt(path + "energy-generic.dat")
#bin_size = 20
#bins = np.histogram(y, bins=bin_size)
#y2 = np.digitize(y, bins[1])
#counts = Counter(y2)
#
#
#dsc = max(counts.values()) # DESIRED SAMPLE COUNT
#print(dsc)
#XX = X.copy()
#yy = y.copy()
#
#for item_class in counts:
#    asc = counts.get(item_class) # ACTUAL SAMPLE SIZE
#    diff = dsc - asc
#    repeat = (ceil(diff / asc) * (diff >= 0))
#    print(asc)
#    print(diff)
#    print(repeat)
#    if repeat == 0: continue
#    index = np.where(y2 == item_class)
#    index = np.repeat(index, repeat)
#    repeat_index = np.random.choice(index, diff)
#    XX = np.concatenate((XX, X[repeat_index]), axis = 0)
#    yy = np.concatenate((yy, y[repeat_index]), axis = 0)
#    print(y[repeat_index].shape)
#    
#
#plt.hist(yy, bins=bin_size)
#plt.show()

def make_balanced(X, y, dsc = None, bin_size = 20):
    ## WARNING : This function does not check length and bin_size difference
    bins = np.histogram(y, bins=bin_size)
    y2 = np.digitize(y, bins[1])
    counts = Counter(y2)
    
    if dsc == None:    
        dsc = max(counts.values()) # DESIRED SAMPLE COUNT
    XX = X.copy()
    yy = y.copy()
    
    for item_class in counts:
        asc = counts.get(item_class) # ACTUAL SAMPLE SIZE
        diff = dsc - asc
        repeat = (ceil(diff / asc) * (diff >= 0))
        if repeat == 0: continue
        index = np.where(y2 == item_class)
        index = np.repeat(index, repeat)
        repeat_index = np.random.choice(index, diff)
        XX = np.concatenate((XX, X[repeat_index]), axis = 0)
        yy = np.concatenate((yy, y[repeat_index]), axis = 0)
    
    return XX, yy
