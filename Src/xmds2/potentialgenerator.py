import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import h5py
import random as rnd


class PotentialGenerator(object):
    cur_pot_type = None
    type_of_pots = 0

    def __init__(self, seed = 100, width = 10, Np = 128, inf_val = 30, g_exec_func = None):
        self.width = width
        self.total_width = 2 * width
        self.step_size = self.total_width / float(Np)
        self.Np = Np
        self.inf_val = inf_val
        self.g_exec_func = g_exec_func
        self.x = np.arange(-self.width, self.width, self.step_size)
        self.seed = seed

    def create_envolope_pot(self, beta = None, width = 10, Np = 128,  l_x0 = -5, r_x0 = 5):
        if beta == None:
            beta = 1 + rnd.random() * 3
        envelope = (np.tanh((self.x + l_x0) * beta) - np.tanh(beta * (self.x + r_x0)))

        return envelope

    def potential_process(self, pot, procs = [], args = []):
        """ args: A list of len(procs) that involves arguments of procs as tuples"""
        Nprocs = len(procs)
        pot += np.abs(np.min(pot))
        pot *= self.create_envolope_pot(2)

        for i in range(Nprocs):
            pot = procs[i](*args[i])

        pot *= self.inf_val / np.max(np.abs(pot)) 
        pot += np.abs(np.min(pot))

        return pot

    def generate_random_pot(self, sigma = 3, exec_func = None):
        c = np.array([rnd.gauss(0, 1) for i in range(self.Np)])
        pot = np.zeros(self.Np)
        pot[0] = c[0]
        for i in range(self.Np - 1):
            pot[i+1] = pot[i] + c[i]

        procs = [scipy.ndimage.filters.gaussian_filter1d]
        args = [(pot, sigma)]

        pot = self.potential_process(pot, procs, args)

        if self.g_exec_func != None:
            self.g_exec_func(self.x, pot)

        if exec_func != None:
            exec_func(self.x, pot)

        return pot

    def generate_random_pot_2(self, sigma = None, exec_func = None):
        """ Create random potential by using sine and cosine series with random coeffs. """

        pot = np.zeros(self.Np)
        Nterms = rnd.randint(1, 100)
        n_range = rnd.uniform(0, 20)
        if sigma == None:
            sigma = rnd.uniform(0.1, 10)

        for i in range(Nterms):
            A = rnd.gauss(0, 1)
            B = rnd.gauss(0, 1)
            n1 = (rnd.gauss(0, 1) * sigma) * np.pi / self.total_width
            n2 = (rnd.gauss(0, 1) * sigma) * np.pi / self.total_width
            pot += A * np.sin(n1 * self.x) + B * np.cos(n2 * self.x)

        procs = [scipy.ndimage.filters.gaussian_filter1d]
        args = [(pot, sigma)]

        pot = self.potential_process(pot, procs, args)

        if self.g_exec_func != None:
            self.g_exec_func(self.x, pot)

        if exec_func != None:
            exec_func(self.x, pot)    

        return pot

    def generate_random_pot_3(self, sigma = None, scale_fac = 8, exec_func = None):
        
        if sigma == None:
            sigma = rnd.uniform(2.5, 10)

        bin_grid = np.array([rnd.randint(0, 1) for i in range(self.Np // scale_fac)])
        bin_grid = np.repeat(bin_grid, scale_fac)

        bin_grid_2 = np.array([rnd.randint(0, 1) for i in range(self.Np // (scale_fac * 2))])
        bin_grid_2 = np.repeat(bin_grid_2, scale_fac)

        padding = np.zeros((len(bin_grid) - len(bin_grid_2)) // 2)

        bin_grid_2 = np.concatenate((padding, bin_grid_2), axis = 0)
        bin_grid_2 = np.concatenate((bin_grid_2, padding), axis = 0)

        pot = bin_grid - bin_grid_2

        procs = [scipy.ndimage.filters.gaussian_filter1d]
        args = [(pot, sigma)]
        pot = self.potential_process(pot, procs, args)

        if self.g_exec_func != None:
            self.g_exec_func(self.x, pot)

        if exec_func != None:
            exec_func(self.x, pot)    

        return pot

    def generate_well_pot(self, exec_func = None):
        lc = rnd.uniform(-5, 5) 
        lw = rnd.uniform(1, 8)
        l, r = lc - lw/2, lc + lw/2

        #pot = np.array([0 if l < x < r else 100 for x in self.x])
        pot = self.create_envolope_pot(beta = 15, l_x0 = l, r_x0 = r)
        pot *= self.inf_val / np.max(np.abs(pot)) 
        pot += np.abs(np.min(pot))

        if self.g_exec_func != None:
            self.g_exec_func(self.x, pot)

        if exec_func != None:
            exec_func(self.x, pot)    

        return pot

    def generate_harmonic_pot(self, exec_func = None):
        x0 = rnd.uniform(-5, 5)
        omega = rnd.uniform(0.01, 4)
        pot =  0.5 * omega**2 * (self.x - x0)**2

        index = np.where(pot > 100)
        pot[index] = 100
        pot *= self.inf_val / np.max(np.abs(pot)) 

        if self.g_exec_func != None:
            self.g_exec_func(self.x, pot)

        if exec_func != None:
            exec_func(self.x, pot)    

        return pot

    def generate_gaussian_pot(self, exec_func = None):
        low = 0
        inc = 0.5
        high = 4

        l_1 = rnd.uniform(1, 10)
        mu_1 = rnd.uniform(-5, 5)
        s_1 = rnd.uniform(low + inc, high - (high / 7 * np.abs(mu_1)))
        l_2 = rnd.uniform(1, 10)
        mu_2 = rnd.uniform(-5, 5)
        s_2 = rnd.uniform(low + inc, high - (high / 7 * np.abs(mu_2)))

        pot = (-l_1 * np.exp(-((self.x - mu_1)**2) / (s_1**2))) + (-l_2 * np.exp(-((self.x - mu_2)**2) / (s_2**2)));
        pot += np.abs(np.min(pot))
        pot *= self.inf_val / np.max(np.abs(pot)) 

        if self.g_exec_func != None:
            self.g_exec_func(self.x, pot)

        if exec_func != None:
            exec_func(self.x, pot)    

        return pot

def save_as_h5(x, pot):
    hf = h5py.File("func.h5", "w")
    hf.create_dataset("func", data=pot)
    hf.create_dataset("x", data=x)
    hf.close()

def display_pot(x, pot):
    plt.plot(x, pot)
    plt.show()


#        index = (self.Np // self.total_width) * (self.width // 2)
#        mask = np.zeros(self.Np - index * 2)
#
#        k = rnd.randint(2, 7)
#
#        for i in range(k**2):
#            mask[rnd.randint(0, len(mask))] = 1
#
#        #mask = self.potential_process(mask, procs, args)
#
#        mask = np.concatenate((np.zeros(index), mask), axis = 0)
#        mask = np.concatenate((mask, np.zeros(index)), axis = 0)
#        mask = scipy.ndimage.filters.gaussian_filter1d(mask, sigma)
#
#        pot *= mask