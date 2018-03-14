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

    f+=np.abs(np.min(f))
    f = scipy.ndimage.filters.gaussian_filter1d(f, sigma)
    f *= envelope

    hf = h5py.File("func.h5", "w")

    hf.create_dataset('func', data=f)
    hf.create_dataset('x', data=x)

    hf.close()

    return x, f