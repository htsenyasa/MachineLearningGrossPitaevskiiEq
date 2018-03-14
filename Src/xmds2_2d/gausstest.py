import numpy as np
import matplotlib.pyplot as plt



x = np.arange(-10, 10, 20/128)
print(x)

def gauss(x, mu, sigma):
    return (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(-(x-mu)**2 / 2 * sigma**2)

a = gauss(x, 0, 1)
b = gauss(x, 0, 1)

z = a * a

plt.plot(a)
plt.show()