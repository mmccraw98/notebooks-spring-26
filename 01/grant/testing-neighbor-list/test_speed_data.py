import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import jaxdem as jd
import os

data_root = '/home/mmccraw/dev/data/26-01-01/grant/profiling-cell-list/jamming'
df = pd.read_csv('profile_data.csv')
vertex_rad = np.ones(df.shape[0])
print(df.keys())

mu_norm = LogNorm(1e-2, 1)
cmap = plt.cm.viridis

for alpha in df.alpha.unique():
    fig, ax = plt.subplots(1, 4, figsize=(8, 3), constrained_layout=True, sharex=True)

    for mu in df.mu.unique():
        state = jd.utils.h5.load(os.path.join(data_root, f'mu-{mu}-alpha-{alpha}', 'state.h5'))
        unique_radii, radii_counts = np.unique(state.rad, return_counts=True)
        vertex_diam = 2 * unique_radii[np.argmax(radii_counts)]

        mask = (df.alpha == alpha) & (df.mu == mu)
        vertex_rad[mask] = unique_radii[np.argmax(radii_counts)]
        _df = df[mask]

        print(_df.true_packing_fraction)

        c = cmap(mu_norm(mu))
        x = _df.skins / vertex_diam
        # x = _df.true_packing_fraction
        ax[0].plot(x, _df.run_time, c=c)
        ax[1].plot(x, _df.builds, c=c)
        ax[2].plot(x, _df.occupancy, c=c)
        ax[2].plot(x, _df.max_occupancy, ls='--', marker='x', c=c)
        ax[3].plot(x, _df.overflow, c=c)
    for a in ax:
        a.set_xlabel('Skin size / \sigma')
        a.set_xscale('log')
    ax[0].set_ylabel('Run time')
    ax[1].set_ylabel('Builds')
    ax[2].set_ylabel('Occupancy')
    ax[3].set_ylabel('Overflow')

    sm = plt.cm.ScalarMappable(norm=mu_norm, cmap=cmap)
    sm.set_array([1e-2, 1e-1, 1e0])
    cbar = plt.colorbar(sm, ax=ax[-1], label=r'$\mu_{eff}$')

    plt.savefig(f'new-profiles/alpha-{alpha}.png')
    plt.close()
    
df['vertex_rad'] = vertex_rad
df.to_csv('profile_data.csv', index=False)