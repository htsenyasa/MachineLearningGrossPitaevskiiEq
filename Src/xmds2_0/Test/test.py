import numpy as np
import genrandompot as pot
import h5py

x = np.arange(-10, 10, 20/128)
y, _ = pot.generate_random_pot(3)

hf = h5py.File("func.h5", "w")

hf.create_dataset('func', data=y)
hf.create_dataset('x', data=x)

hf.close()