import numpy as np
import matplotlib.pyplot as plt

dens = np.loadtxt("density.txt")
pots = np.loadtxt("potential.txt")
x = np.loadtxt("position.txt")


plt.plot(x, dens[0], label = "g = 0")
plt.plot(x, dens[1], label = "g = 0.1")
plt.plot(x, dens[2], label = "g = 1")
plt.plot(x, dens[3], label = "g = 10")
plt.plot(x, dens[4], label = "g = 20")
plt.xlabel("$z$", fontsize = 20)
plt.ylabel("$|\psi|^2$", fontsize = 20)
plt.legend()
figure = plt.gcf()
figure.set_size_inches(8,6)
plt.savefig("potvsdens.svg", format = "svg", dpi=1200)
#plt.show()
