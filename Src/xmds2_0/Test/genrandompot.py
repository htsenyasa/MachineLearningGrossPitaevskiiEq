import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def generate_random_pot(sigma):
    Np = 128
    beta=30
    x = np.linspace(-10,10,Np)
    envelope=(np.tanh((x-9.8)*beta)-np.tanh(beta*(x+9.8)))
    np.random.seed(102)
    c = np.random.randn(Np)
    f = np.zeros(Np)
    f[0] = c[0]
    for i in range(len(x)-1):
        f[i+1] = f[i] + c[i]

    f+=np.abs(np.min(f))
    f = scipy.ndimage.filters.gaussian_filter1d(f, sigma)
    f *= envelope
    return f, x