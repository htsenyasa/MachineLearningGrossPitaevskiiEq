import numpy as np
import matplotlib.pyplot as plt
import random as rnd

#A*|x-c|

start = -10
stop = 10
X = np.arange(start, stop, (stop-start)/128)
low = 0
inc = 0.5
high = 4

np.random.seed(1234)

for i in range(100):

    l_1 = np.random.uniform(1, 10)
    mu_1 = np.random.uniform(-5, 5)
    s_1 = np.random.uniform(low + inc, high - (high / 7 * np.abs(mu_1)))

    l_2 = np.random.uniform(1, 10)
    mu_2 = np.random.uniform(-5, 5)
    s_2 = np.random.uniform(low + inc, high - (high/7 * np.abs(mu_2)))

    zz = [- l_1 * np.exp(-(x-mu_1)**2/s_1**2) - l_2 * np.exp(-(x-mu_2)**2/s_2**2) for x in X]

    fig, ax1 = plt.subplots()
    ax1.plot(X, zz)
    props = dict(boxstyle='square', facecolor='white', alpha=0.5)
    textstr = "l_1:{:.2f}\nmu_1:{:.2f}\ns_1:{:.2f}".format(l_1, mu_1, s_1)
    ax1.text(0.03, 0.85, textstr, transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)
    plt.show()


   #l_1 = 10
   #mu_1 = -7
   #s_1 = 4
   #zz = [- l_1*np.exp(-(x-mu_1)**2/s_1) for x in X]
   ##zz = [np.sqrt(l_1/(2 * np.pi * x**3)) * np.exp(-l_1 * (x - mu_1)**2 / (2 * mu_1**2 * x)) for x in X[:]]
