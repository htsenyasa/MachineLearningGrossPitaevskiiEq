import numpy as np
import matplotlib.pyplot as plt
import random as rnd
import math as m

num_grid = 128
domain_len = 20
ratio = num_grid/domain_len
inf_pot = 100

pot = np.zeros([128])
lw = rnd.uniform(-5, -0.5) # left_well
rw = -lw # right_well
lw_i = m.ceil(lw * ratio) + num_grid//2 #index of lw 
rw_i = m.floor(rw * ratio) + num_grid//2 #index of rw
print("L: {}- R:{}".format(lw_i, rw_i))
pot[0:lw_i] = inf_pot
pot[rw_i:] = inf_pot
#plt.plot(pot)
#plt.show()
#plt.clf()

def f_x(x):
    lw = rnd.uniform(-5, -0.5) # left_well
    rw = -lw # right_well
    lw_i = m.ceil(lw * ratio) + num_grid//2 #index of lw 
    rw_i = m.floor(rw * ratio) + num_grid//2 #index of rw

    if (lw < x < rw):
        return 0
    return 100

x_in = np.arange(-10, 10, 128)