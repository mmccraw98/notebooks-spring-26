import os
import numpy as np
import matplotlib.pyplot as plt

for p in os.listdir('cv-data'):
    for f in os.listdir(os.path.join('cv-data', p)):
        if 'thermal_' not in f:
            continue
        data = np.load(os.path.join('cv-data', p, f))
        
        temp = data['temp']
        te = data['pe'] + data['ke'] + data['ke_r']

        C_v, _ = np.polyfit(
            np.mean(data['temp'], axis=0),
            np.mean(data['pe'] + data['ke'] + data['ke_r'], axis=0),
            1
        )
        plt.scatter(-data['delta_phi'], C_v / data['N'])
    plt.xscale('log')
    plt.savefig(f'figures/c_{p}.png')
    plt.close()