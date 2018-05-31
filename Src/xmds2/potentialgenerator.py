import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import h5py
import random as rnd


class PotentialGenerator(object):
    cur_pot_type = None
    type_of_pots = 0

    def __init__(self, seed = 10, width = 10, Np = 256, inf_val = 50, g_exec_func = None, transform = None):
        self.width = width
        self.total_width = 2 * width
        self.step_size = self.total_width / float(Np)
        self.Np = Np
        self.inf_val = inf_val
        self.g_exec_func = g_exec_func
        self.x = np.arange(-self.width, self.width, self.step_size)
        self.seed = seed; rnd.seed(seed)
        self.proc_inf_val = 50
        if transform == None:
            transform = lambda x: x
        self.transform = transform
        self.omega = []

    #def create_envolope_pot(self, beta = None, width = 10,  l_x0 = -8, r_x0 = 8):
    #    if beta == None:
    #        beta = 1 + rnd.random() * 3
    #    envelope = (np.tanh((self.x + l_x0) * beta) - np.tanh(beta * (self.x + r_x0)))
#
    #    return envelope

    def create_envolope_pot(self, beta = None, width = 10,  l_x0 = -6, r_x0 = 6):
        if beta == None:
            beta = 3
            #beta = 1 + rnd.random() * 3
        a1 = (1 - np.tanh(beta * (self.x + r_x0)))/2
        a2 = (1 + np.tanh(beta * (self.x + l_x0)))/2
        a3 = 1 - a1 - a2
        return a3, a1+a2


    def potential_process(self, pot, procs = [], args = []):
        """ args: A list of len(procs) that involves arguments of procs as tuples"""
        #tpot = pot
        Nprocs = len(procs)
        #pot_min = np.min(pot)
        #if pot_min < 0:
        pot += np.abs(np.min(pot))

        for i in range(Nprocs):
            pot = procs[i](*args[i])

        a3, a1a2 = self.create_envolope_pot(4)
        Vmax = np.max(pot) * 2
        pot = pot * a3 + a1a2 * Vmax
        pot -= np.min(pot)
        pot *= self.inf_val / np.max(np.abs(pot)) 
        #pot += np.abs(np.min(pot))

        return pot
        #return tpot, pot


    def generate_random_pot(self, sigma = None, exec_func = None):
        points = np.array([rnd.gauss(0, 1) for i in range(self.Np)])
        pot = np.zeros(self.Np)
        pot[0] = points[0]
        for i in range(self.Np - 1):
            pot[i+1] = pot[i] + points[i]

        if sigma == None:
            sigma = 8 + 3 * rnd.random() + self.Np/128

        procs = [scipy.ndimage.filters.gaussian_filter1d]
        args = [(pot, sigma)]

        pot = self.potential_process(pot, procs, args)

        if self.g_exec_func != None:
            self.g_exec_func(self.x, pot)

        if exec_func != None:
            exec_func(self.x, pot)

        return pot

    def generate_random_pot_2(self, exec_func = None):
        """ Create random potential by using sine and cosine series with random coeffs. """

        #Vi = np.random.uniform(0,1,self.Np)
        Vi = np.array([rnd.uniform(0, 1) for i in range(self.Np)])
        k = np.fft.rfftfreq(self.Np)
        kc = .0125
        kc = rnd.uniform(1, 10) * k[1]
        #kc = np.random.uniform(1,10) * k[1]
        V0 = 3
        M = 4
        Vk = V0 * np.fft.rfft(Vi)
        pot = np.fft.irfft(np.exp(-(k/kc)**M)*Vk)

        #if sigma == None:
        #    sigma = rnd.uniform(0.1, 10)
        #procs = [scipy.ndimage.filters.gaussian_filter1d]
        #args = [(pot, sigma)]

        pot = self.potential_process(pot)

        if self.g_exec_func != None:
            self.g_exec_func(self.x, pot)

        if exec_func != None:
            exec_func(self.x, pot)    

        return pot

    def generate_random_pot_3(self, sigma = None, scale_fac = 8, exec_func = None):
        
        if sigma == None:
            sigma = rnd.uniform(2.5, 10) * (self.Np / 64) 

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
        #pot, a1a2 = self.create_envolope_pot(beta = 15, l_x0 = l, r_x0 = r)
        beta = 15
        pot = (np.tanh((self.x + l) * beta) - np.tanh(beta * (self.x + r)))
        pot *= self.inf_val / np.max(np.abs(pot)) 
        pot += np.abs(np.min(pot))

        if self.g_exec_func != None:
            self.g_exec_func(self.x, pot)

        if exec_func != None:
            exec_func(self.x, pot)    

        return pot

    def generate_harmonic_pot(self, exec_func = None):
        x0 = rnd.uniform(-1, 1)
        omega = rnd.uniform(0.5, 1.3)
        self.omega.append(omega)
        pot =  0.5 * omega**2 * (self.transform(self.x - x0))**2

        pot += np.abs(np.min(pot))
        a3, a1a2 = self.create_envolope_pot(4, l_x0 = -6 + x0, r_x0 = 6 + x0)
        #Vmax = np.max(pot) * 2
        Vmax = self.inf_val
        pot = pot * a3 + a1a2 * Vmax
        #pot *= self.inf_val / np.max(np.abs(pot)) 
        
        #pot = self.potential_process(pot)
        #pot =  0.5 * (self.transform(self.x - x0))**2
        #index = np.where(pot > self.inf_val)
        #pot[index] = self.inf_val
        #pot *= self.inf_val / np.max(np.abs(pot)) 

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

    def reset_seed(self, seed):
        self.seed = seed;
        rnd.seed(seed)

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

#
#import potentialgenerator as pg
#import numpy as np
#import matplotlib.pyplot as plt
#a = pg.PotentialGenerator()
#pot_gen = pg.PotentialGenerator()
#pot_generators = [
#                  pot_gen.generate_random_pot, 
#                  pot_gen.generate_random_pot_2, 
#                  pot_gen.generate_random_pot_3]
#                  
#pot_types = [0, 1, 2]
#x = np.arange(-10, 10, 20/512)
#for i in range(10):
#    for pot_type in pot_types:
#        plt.plot(x, pot_generators[pot_type]())
#        plt.show()


#    def generate_random_pot_2(self, sigma = None, exec_func = None):
#        """ Create random potential by using sine and cosine series with random coeffs. """
#
#        pot = np.zeros(self.Np)
#        Nterms = rnd.randint(1, 100)
#        if sigma == None:
#            sigma = rnd.uniform(0.1, 10)
#
#        for i in range(Nterms):
#            A = rnd.gauss(0, 1)
#            B = rnd.gauss(0, 1)
#            n1 = (rnd.gauss(0, 1) * sigma) * np.pi / self.total_width
#            n2 = (rnd.gauss(0, 1) * sigma) * np.pi / self.total_width
#            pot += A * np.sin(n1 * self.x) + B * np.cos(n2 * self.x)
#
#        procs = [scipy.ndimage.filters.gaussian_filter1d]
#        args = [(pot, sigma)]
#
#        pot = self.potential_process(pot, procs, args)
#
#        if self.g_exec_func != None:
#            self.g_exec_func(self.x, pot)
#
#        if exec_func != None:
#            exec_func(self.x, pot)    
#
#        return pot

#    def potential_process(self, pot, procs = [], args = []):
#        """ args: A list of len(procs) that involves arguments of procs as tuples"""
#        tpot = pot
#        Nprocs = len(procs)
#        pot += np.abs(np.min(pot))
#
#        for i in range(Nprocs):
#            pot = procs[i](*args[i])
#
#        pot *= self.create_envolope_pot(4)
#
#
#        pot *= self.inf_val / np.max(np.abs(pot)) 
#        pot += np.abs(np.min(pot))
#
#        #return tpot, pot
#        return pot