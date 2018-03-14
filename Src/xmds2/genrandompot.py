import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

def generate_random_pot():
    Np = 100
    beta=30
    x = np.linspace(-1,1,Np)
    envelope=(np.tanh((x-0.8)*beta)-np.tanh(beta*(x+0.8)))
    c = np.random.randn(Np)
    f = np.zeros(Np)
    f[0] = c[0]
    for i in range(len(x)-1):
        f[i+1] = f[i] + c[i]

    f+=np.abs(np.min(f))
    f = scipy.ndimage.filters.gaussian_filter1d(f,10)
    plt.plot(x,f * envelope)