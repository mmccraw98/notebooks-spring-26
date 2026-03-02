import os
import numpy as np
import matplotlib.pyplot as plt

for p in os.listdir():
    if 'thermal_' not in p:
        continue
    data = np.load(p)

    C_v, _ = np.polyfit(
        np.mean(data['temp'], axis=-1),
        np.mean(data['pe'] + data['ke'] + data['ke_r'], axis=-1),
        1
    )
    plt.scatter(-data['delta_phi'], C_v / data['N'])
plt.xscale('log')
plt.savefig('figures/cv.png')
print(data['N'])