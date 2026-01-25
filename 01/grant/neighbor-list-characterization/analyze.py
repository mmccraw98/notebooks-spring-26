import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
from bump_utils import create_clumps
import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import shutil

which = '2d'
plot_energies = True

if which == '3d':
    data_root = '/home/mmccraw/dev/data/26-01-01/grant/neighbor-list-characterization-3d/dynamics-1'
    save_name = 'nl-performance-3d.png'
else:
    data_root = '/home/mmccraw/dev/data/26-01-01/grant/neighbor-list-characterization/dynamics-20'
    save_name = 'nl-performance-2d.png'

if os.path.exists(os.path.join('figures', which)):
    shutil.rmtree(os.path.join('figures', which))
os.makedirs(os.path.join('figures', which))

skin = []
mu = []
overflow = []
builds = []
performance = []
radius = []
phi = []
phi_n = []
mean_occupancy = []
max_occupancy = []
predicted_max_occupancy = []
nl_cutoff = []
delta_te = []
imbalanced_force = []

for name in os.listdir(data_root):
    path = os.path.join(data_root, name)
    try:
        state_traj = jd.utils.h5.load(os.path.join(path, 'state.h5'))
        system_traj = jd.utils.h5.load(os.path.join(path, 'system.h5'))
        if plot_energies:
            energies = np.load(os.path.join(path, 'energies.npz'))
            pe = energies['pe']
            ke_r = energies['ke_r']
            ke_t = energies['ke_t']
            te = pe + ke_r + ke_t
            plt.plot(pe, label='pe')
            plt.plot(ke_t, label='ke_t')
            plt.plot(ke_r, label='ke_r')
            plt.plot(te, label='te')
            plt.legend()
            plt.savefig(f'figures/{which}/{name}.png', dpi=600)
            plt.close()
    except Exception as e:
        print(e)
        continue



    _, _mu, _, _skin = name.split('-')
    mu.append(float(_mu))
    skin.append(float(_skin))
    nl_cutoff.append(float(system_traj.collider.cutoff[0]) * (1.0 + float(_skin)))
    overflow.append(float(jnp.any(system_traj.collider.overflow)))
    builds.append(int(system_traj.collider.n_build_times[-1]))
    with open(os.path.join(path, 'stats.json'), 'r') as f:
        stats = json.load(f)
    performance.append(state_traj.N * stats['n_steps'] / stats['run_time'])
    delta_te.append(stats['delta_te'])
    radius.append(state_traj.rad.min())
    phi.append(float(jnp.sum(jnp.pi * state_traj.rad[0] ** state_traj.dim * ((1) if state_traj.dim == 2 else (4 / 3))) / jnp.prod(system_traj.domain.box_size[0])))
    phi_n.append(float(state_traj.N / jnp.prod(system_traj.domain.box_size[0])))
    occupancy = jnp.sum(system_traj.collider.neighbor_list != -1, axis=-1)
    mean_occupancy.append(jnp.mean(occupancy[occupancy != 0]))
    max_occupancy.append(jnp.max(occupancy))

    safety_factor = 1.2
    nl_volume = jnp.pi * (safety_factor * float(system_traj.collider.cutoff[0]) * (1.0 + float(_skin))) ** state_traj.dim * ((1) if state_traj.dim == 2 else (4 / 3))
    number_density = float(state_traj.N / jnp.prod(system_traj.domain.box_size[0]))
    occupancy = nl_volume * number_density
    predicted_max_occupancy.append(int(number_density * nl_volume) + 10)

    max_imbalance = -np.inf
    for i in range(state_traj.shape[0]):
        f = state_traj.force[i]
        cid = state_traj.clump_ID[i]
        order = jnp.argsort(state_traj.unique_ID[i])
        _, offsets = jnp.unique(state_traj.clump_ID[i][order], return_index=True)
        max_imbalance = max(max_imbalance, float(jnp.linalg.norm(jnp.sum(f[order][offsets], axis=0))))
    imbalanced_force.append(max_imbalance)

    print(jnp.mean(jax.vmap(lambda st: jd.utils.thermal.compute_temperature(st, is_rigid=True, subtract_drift=True))(state_traj)))

df = pd.DataFrame({
    'skin': skin,
    'mu': mu,
    'overflow': overflow,
    'builds': builds,
    'performance': performance,
    'radius': radius,
    'phi': phi,
    'phi_n': phi_n,
    'max_occupancy': max_occupancy,
    'mean_occupancy': mean_occupancy,
    'predicted_max_occupancy': predicted_max_occupancy,
    'nl_cutoff': nl_cutoff,
    'delta_te': delta_te,
    'imbalanced_force': imbalanced_force,
})

color_mu = False

if color_mu:
    norm = LogNorm(df.mu.min(), df.mu.max())
else:
    norm = LogNorm(df.phi.min(), df.phi.max())
cmap = plt.cm.viridis

fig, ax = plt.subplots(1, 5, figsize=(10, 4), constrained_layout=True)

for mu in df.mu.unique():
    mask = (df.mu == mu)
    _df = df[mask].sort_values(by='skin')
    if color_mu:
        c = cmap(norm(mu))
    else:
        c = cmap(norm(_df.phi.values[0]))
    x = _df.skin ** state_traj.dim * _df.phi
    # x = _df.nl_cutoff ** state.dim * _df.phi_n
    ax[0].plot(x, _df.performance, c=c)
    ax[1].plot(x, _df.max_occupancy, c=c)
    ax[1].plot(x, _df.predicted_max_occupancy, c=c, ls='--')
    ax[2].plot(x, _df.overflow, c=c)
    ax[3].plot(x, _df.delta_te, c=c)
    ax[4].plot(x, _df.imbalanced_force, c=c)

sp = np.logspace(-2, 3, 10)
ax[1].plot(sp, 2 * sp, linestyle='--', color='k', alpha=0.5, zorder=0)

for a in ax:
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlabel(r'$\tilde{s} \phi$', fontsize=16)
ax[0].set_ylabel(r'Performance ($N_{_{steps}} N_{_{particles}} / t$)', fontsize=16)
ax[1].set_ylabel('Max Occupancy', fontsize=16)
ax[2].set_ylabel('Overflow', fontsize=16)
ax[3].set_ylabel(r'$\delta_{E}$', fontsize=16)
ax[4].set_ylabel(r'$\max | \langle F \rangle |$', fontsize=16)
ax[0].axvline(2e-2, linestyle='--', color='k', alpha=0.5, zorder=0)

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
if color_mu:
    sm.set_array(np.unique(df.mu))
    cbar = plt.colorbar(sm, ax=plt.gca(), label=r'$\mu_{eff}$')
else:
    sm.set_array(np.unique(df.phi.astype(float)))
    cbar = plt.colorbar(sm, ax=plt.gca(), label=r'$\phi$')

plt.savefig(f'figures/{save_name}', dpi=600)
plt.close()