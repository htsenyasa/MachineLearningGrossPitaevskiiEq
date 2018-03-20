import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import h5py
import random as rnd
from mpl_toolkits.mplot3d import Axes3D

## NUMPY'S RANDOM NUMBER GENERATOR IS NOT THREAD SAFE SO WE USED PYTHON'S##

rnd.seed(100)


def save_as_h5(x, f):
    hf = h5py.File("func.h5", "w")
    hf.create_dataset("func", data=f)
    hf.create_dataset("x", data=x)
    hf.close()


def create_envolope_pot(beta, width = 10, Np = 128,  x0 = 5):
    x = np.arange(-width, width, 2 * width / Np)
    envelope = (np.tanh((x-x0) * beta) - np.tanh(beta * (x+x0)))
    return envelope


def generate_random_pot(sigma = 3, width = 10, Np = 128, inf_val = 100, exec_func = None):

    total_width = 2 * np.abs(width)
    step_size = total_width / 128
    x = np.arange(-width, width, step_size)
    
    beta = 1 + rnd.random() * 3
    envelope = create_envolope_pot(beta)
    
    c = np.array([rnd.gauss(0, 1) for i in range(Np)])
    f = np.zeros(Np)
    f[0] = c[0]
    for i in range(len(x)-1):
        f[i+1] = f[i] + c[i]

    f += np.abs(np.min(f))
    f *= envelope
    f = scipy.ndimage.filters.gaussian_filter1d(f, sigma)
    f *= inf_val / np.max(np.abs(f)) 
    f += np.abs(np.min(f))

    if exec_func != None:
        exec_func(x, f)

    return x, f


def generate_random_pot_2(sigma = None, width = 10, Np = 128, inf_val = 100, exec_func = None):
    """ Create random potential by using sine and cosine series with random coeffs. """

    total_width = 2 * np.abs(width)
    step_size = total_width / 128
    x = np.arange(-width, width, step_size)

    beta = 1 + rnd.random() * 3
    envelope = create_envolope_pot(beta)

    f = np.zeros(Np)
    Nterms = rnd.randint(1, 100)
    n_range = rnd.uniform(0, 20)
    if sigma == None:
        sigma = rnd.uniform(0.1, 10)

    for i in range(Nterms):
        A = rnd.gauss(0, 1)
        B = rnd.gauss(0, 1)
        n1 = (rnd.gauss(0, 1) * sigma) * np.pi / total_width
        n2 = (rnd.gauss(0, 1) * sigma) * np.pi / total_width
        f += A * np.sin(n1 * x) + B * np.cos(n2 * x)
    f += np.abs(np.min(f))
    f *= envelope
    f = scipy.ndimage.filters.gaussian_filter1d(f, 1.5)
    f *= inf_val / np.max(np.abs(f)) 
    f += np.abs(np.min(f))

    if exec_func != None:
        exec_func(x, f)

    return x, f


def potential_process(pot, procs, args):
    
    inf_val = 100
    Nprocs = len(procs)
    pot += np.abs(np.min(pot))
    pot *= create_envolope_pot(1 + rnd.random() * 3)

    for i in range(Nprocs):
        pot = procs[i](args[i])

    pot *= inf_val / np.max(np.abs(pot)) 
    pot += np.abs(np.min(pot))

    return pot







#def generate_random_pot_3(sigma = None, width = 10, Np = 128, inf_val = 100, exec_func = None):
#
#    total_width = 2 * np.abs(width)
#    step_size = total_width / 128
#    x = np.arange(-width, width, step_size)
#
#    pdeg = np.random.randint(2, 10)
#    sigma = np.random.uniform(0.01, 10, pdeg)
#    sigma = 3
#    c = np.random.randn(pdeg) * sigma
#    p = np.poly1d(c)
#
#    beta = 1 + rnd.random() * 3
#    x = np.arange(-1, 1, 2/128)
#
#    envelope = (np.tanh((x - 5) * beta)-np.tanh(beta * (x + 5)))
#
#    f = np.polyval(p, x)
#    f += np.abs(np.min(f))
#    #f *= envelope
#    f *= inf_val / np.max(np.abs(f)) 
#    f += np.abs(np.min(f))
#
#    return x, f

#for i in range(100):
#    plt.plot(*generate_random_pot_3())
#    plt.show()


#def display_pot(Z):
#    X = np.arange(-10, 10, 20/128)
#    Y = np.arange(-10, 10, 20/128)
#    X, Y = np.meshgrid(X, Y)
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)
#    #plt.imshow(Z)
#    plt.show()
