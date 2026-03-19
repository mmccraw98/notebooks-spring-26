from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt

data = np.load('gamma-sweep.npz')

gamma_norm = LogNorm(min(data['gamma']), max(data['gamma']))
cmap = plt.cm.viridis

for i, gamma in enumerate(data['gamma']):
    pe = data[f'pe_{i}']
    ke = data[f'ke_{i}']
    plt.plot(ke, c=cmap(gamma_norm(gamma)))
plt.yscale('log')
plt.savefig('gamma.png')