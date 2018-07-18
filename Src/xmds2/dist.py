import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from scipy.integrate import quad
import scipy.optimize as spo
from scipy.interpolate import interp1d
import random as rnd

a = 3
b = 7

def func(x):
    return x**2
    
def f(x):
    return derivative(func, x, dx=1e-6)
    
def F(x):
    return quad(f, a, x)[0]
    

norm = quad(f, a, b)[0]
x = np.linspace(a, b, 1001)
Fs = np.array([F(arg) / norm for arg in x])
#Fs = np.array([F(arg) for arg in x])
inv_F = interp1d(Fs, x, kind='cubic')

x = np.array([rnd.uniform(a, b) for i in range(1000)])
y = func(x)
x = np.array([rnd.uniform(0, 1) for i in range(1000)])
nx = inv_F(x)
ny = func(nx)
    