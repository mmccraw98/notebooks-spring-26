import matplotlib.pyplot as plt
import numpy as np
plate_pos = np.load('plate_distance_hist.npy')
pe = np.load('pe_hist.npy')

i = np.argwhere(pe > 1e-16)[0][0]
pe = pe[i:]
h = plate_pos[i:]
h = h[0] - h

x = np.logspace(-1, 0, 10)
plt.plot(x, 1e-4 * x ** 2)

plt.scatter(h, pe)
plt.yscale('log')
plt.xscale('log')
plt.savefig('compression.png')
plt.close()