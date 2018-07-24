import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.integrate import quad
import scipy.optimize as spo
from scipy.interpolate import interp1d
import random as rnd


def g_dist(gg, en, en_std, g_min = 0, g_max = 30):
    inv_gg_pos_en_std = interp1d(en + en_std, gg, kind='cubic')
    inv_gg_neg_en_std = interp1d(en - en_std, gg, kind='cubic')
    en_plus_std = en + en_std
    en_min_std = en - en_std

    def g_length(en, g_min = g_min, g_max = g_max):
        low = g_min
        high = g_max
        if max(en_min_std) >= en >= min(en_plus_std):
            return np.abs(inv_gg_pos_en_std(en) - inv_gg_neg_en_std(en))
        if en < min(en_plus_std) and en >= min(en_min_std):
            return np.abs(inv_gg_neg_en_std(en) - g_min)
        if en > max(en_min_std):
            return np.abs(g_max - inv_gg_pos_en_std(en))
    
    x_min = min(en_min_std)
    x_max = max(en_plus_std)
    en_new = np.linspace(x_min, x_max, 101)
    
    g_len = np.zeros(len(en_new))
    for i, ee in enumerate(en_new):
        g_len[i] = g_length(ee)
        
    #return interp1d(en_new, g_len, kind='cubic')
    return interp1d(en_new, g_len, kind='cubic')

def inv_dist(X, Y, a = 0, b = 1):
    #X = np.loadtxt("../../data/nonlinearSE/generic_dataset_var/harmonic_gg/gg-generic.dat")
    #Y = np.loadtxt("../../data/nonlinearSE/generic_dataset_var/harmonic_gg/energy-generic.dat")
    func = interp1d(X, Y, kind='cubic')
    f = lambda x: derivative(func, x, dx=1e-6)
    F = lambda x, a: func(x) - func(a)
    norm = func(b) - func(a)
    x = np.linspace(a, b, 1001)
    Fs = np.array([F(arg, a) / norm for arg in x])
    #Fs = np.array([F(arg) for arg in x])
    return interp1d(Fs, x, kind='cubic')
 
 
   
def inv_dist_2(X, Y, Y2, a = 0, b = 1):
    X[0] -= 0.2
    X[-1] += 0.2
    func = interp1d(X, Y, kind='cubic')
    func2 = g_dist(X, Y, Y2, g_min = a, g_max = b)
    f = lambda x: derivative(func, x, dx=1e-6) * (1.0 / func2(func(x))) 
    F = lambda a, x: quad(f, a, x)[0]
    norm = F(a, b)
    x = np.linspace(a, b, 101)
    Fs = np.array([F(a, arg) / norm for arg in x])
    #Fs = np.array([F(arg) for arg in x])
    return interp1d(Fs, x, kind='cubic')



    
    
#def func(x):
#    return np.exp(x**2)
#    
#def f(x):
#    return derivative(func, x, dx=1e-6)
#    
#def F(x, a):
#    return quad(f, a, x)[0]
#    
#def inv_dist():
#    a = 0
#    b = 1
#    norm = quad(f, a, b)[0]
#    x = np.linspace(a, b, 1001)
#    Fs = np.array([F(arg, a) / norm for arg in x])
#    #Fs = np.array([F(arg) for arg in x])
#    inv_F = interp1d(Fs, x, kind='cubic')
#    
#    x = np.array([rnd.uniform(a, b) for i in range(1000)])
#    y = func(x)
#    x = np.array([rnd.uniform(0, 1) for i in range(1000)])
#    nx = inv_F(x)
#    ny = func(nx)