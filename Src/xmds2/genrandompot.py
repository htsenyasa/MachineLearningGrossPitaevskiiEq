import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import h5py

def generate_random_pot(sigma):
    Np = 128
    beta=30
    x = np.arange(-10,10,20/Np)
    envelope=(np.tanh((x-4.8)*beta)-np.tanh(beta*(x+4.8)))
    c = np.random.randn(Np)
    f = np.zeros(Np)
    f[0] = c[0]
    for i in range(len(x)-1):
        f[i+1] = f[i] + c[i]

    f += np.abs(np.min(f))
    f = scipy.ndimage.filters.gaussian_filter1d(f, sigma)
    f *= envelope

    hf = h5py.File("func.h5", "w")

    hf.create_dataset('func', data=f)
    hf.create_dataset('x', data=x)

    hf.close()

    return x, f

def generate_random_pot_2():
    Np = 128
    beta=30
    x = np.arange(-10,10,20/Np)
    envelope=(np.tanh((x-4.8)*beta)-np.tanh(beta*(x+4.8)))
    lim = 100

    for j in range(100):
        f = np.zeros(Np)
        for i in range(100):
            A = np.random.randn()
            n = (np.random.randn() * 5) * np.pi
            f += A * np.sin(n / 20 * x)
        f += np.abs(np.min(f))
        f *= envelope
        f *= lim / np.max(np.abs(f)) 
        f = scipy.ndimage.filters.gaussian_filter1d(f, 1.5)
        f += np.abs(np.min(f))
        plt.plot(f)
        plt.show()


    