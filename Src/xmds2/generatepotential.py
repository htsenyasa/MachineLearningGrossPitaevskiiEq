import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import h5py
import random as rnd


class PotentialGenerator(object):
    cur_pot_type = None
    type_of_pots = 0

    def __init__(self, seed = 100, width = 10, Np = 128, inf_val = 100, g_exec_func = None):
        self.width = width
        self.total_width = 2 * width
        self.step_size = self.total_width / float(Np)
        self.Np = Np
        self.inf_val = inf_val
        self.g_exec_func = g_exec_func
        self.x = np.arange(-self.width, self.width, self.step_size)
        self.seed = seed

    def create_envolope_pot(self, beta = None, width = 10, Np = 128,  x0 = 5):

        if beta == None:
            beta = 1 + rnd.random() * 3            
        envelope = (np.tanh((self.x - x0) * beta) - np.tanh(beta * (self.x + x0)))

        return envelope


    def generate_random_pot(self, sigma = 3, exec_func = None):

        c = np.array([rnd.gauss(0, 1) for i in range(self.Np)])
        f = np.zeros(self.Np)
        f[0] = c[0]
        for i in range(self.Np - 1):
            f[i+1] = f[i] + c[i]

        procs = [scipy.ndimage.filters.gaussian_filter1d]
        args = [(f, sigma)]

        f = self.potential_process(f, procs, args)

        if self.g_exec_func != None:
            self.g_exec_func(self.x, f)

        if exec_func != None:
            exec_func(self.x, f)

        return f

    def potential_process(self, pot, procs = [], args = []):
        """ args: A list of len(procs) that involves arguments of procs as tuples"""

        Nprocs = len(procs)
        pot += np.abs(np.min(pot))
        pot *= self.create_envolope_pot(1 + rnd.random() * 3)

        for i in range(Nprocs):
            pot = procs[i](*args[i])

        pot *= self.inf_val / np.max(np.abs(pot)) 
        pot += np.abs(np.min(pot))

        return pot
    

    def generate_random_pot_2(self, sigma = None, width = 10, Np = 128, inf_val = 100, exec_func = None):
        """ Create random potential by using sine and cosine series with random coeffs. """

        f = np.zeros(Np)
        Nterms = rnd.randint(1, 100)
        n_range = rnd.uniform(0, 20)
        if sigma == None:
            sigma = rnd.uniform(0.1, 10)

        for i in range(Nterms):
            A = rnd.gauss(0, 1)
            B = rnd.gauss(0, 1)
            n1 = (rnd.gauss(0, 1) * sigma) * np.pi / self.total_width
            n2 = (rnd.gauss(0, 1) * sigma) * np.pi / self.total_width
            f += A * np.sin(n1 * self.x) + B * np.cos(n2 * self.x)

        procs = [scipy.ndimage.filters.gaussian_filter1d]
        args = [(f, sigma)]

        f = self.potential_process(f, procs, args)

        if self.g_exec_func != None:
            self.g_exec_func(self.x, f)

        if exec_func != None:
            exec_func(self.x, f)    

        return f

def save_as_h5(x, f):
    hf = h5py.File("func.h5", "w")
    hf.create_dataset("func", data=f)
    hf.create_dataset("x", data=x)
    hf.close()