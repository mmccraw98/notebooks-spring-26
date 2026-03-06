import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import jax.numpy as jnp

from jaxdem.analysis import LagBinsPseudoLog, evaluate_binned, LagBinsLinear

from jaxdem_scripts.correlations import get_pseudo_log_bins_from_steps

phi_norm = plt.Normalize(0.7, 1.0)
cmap = plt.cm.viridis

def get_relaxation_time(f, t):
    tau = np.interp(1 / np.e, f[::-1], t[::-1])
    if not np.min(f) < 1 / np.e:
        return np.nan
    return tau

dt = 1e-2

data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-5/compression-disks-1/'

fig, ax = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
for stress_file in os.listdir('stress-correlations-disk'):
    corrs = np.load(os.path.join('stress-correlations-disk', stress_file))
    phi_float = float(stress_file.split('phi-')[-1].split('.npz')[0])

    mask = corrs['t'] < corrs['t'].max() / 10
    ax[0].plot(corrs['t'][mask], corrs['contact_stress'][:, 0, 1][mask] / corrs['contact_stress'][0, 0, 1], c=cmap(phi_norm(phi_float)))
    ax[1].plot(corrs['t'][mask], corrs['contact_stress'][:, 0, 0][mask] / corrs['contact_stress'][0, 0, 0], c=cmap(phi_norm(phi_float)))
for a in ax:
    a.set_xscale('log')
    a.set_xlabel(r'$t$', fontsize=16)
ax[0].set_ylabel(r'$\langle \sigma_{xy,C} (t) \sigma_{xy,C} (0) \rangle / \langle \sigma_{xy,C}^2 \rangle$', fontsize=16)
ax[1].set_ylabel(r'$\langle \sigma_{xx,C} (t) \sigma_{xx,C} (0) \rangle / \langle \sigma_{xy,C}^2 \rangle$', fontsize=16)
sm = plt.cm.ScalarMappable(norm=phi_norm, cmap=cmap)
sm.set_array([0.7, 0.8, 0.9, 1.0])
cbar = plt.colorbar(sm, ax=ax[-1], label=r'$\phi$')
plt.savefig('disk-contact-stress.png', dpi=600)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
for stress_file in os.listdir('stress-correlations-disk'):
    corrs = np.load(os.path.join('stress-correlations-disk', stress_file))
    phi_float = float(stress_file.split('phi-')[-1].split('.npz')[0])

    mask = corrs['t'] < corrs['t'].max() / 10
    ax[0].plot(corrs['t'][mask], corrs['stress'][:, 0, 1][mask] / corrs['stress'][0, 0, 1], c=cmap(phi_norm(phi_float)))
    ax[1].plot(corrs['t'][mask], corrs['stress'][:, 0, 0][mask] / corrs['stress'][0, 0, 0], c=cmap(phi_norm(phi_float)))
for a in ax:
    a.set_xscale('log')
    a.set_xlabel(r'$t$', fontsize=16)
ax[0].set_ylabel(r'$\langle \sigma_{xy} (t) \sigma_{xy} (0) \rangle / \langle \sigma_{xy}^2 \rangle$', fontsize=16)
ax[1].set_ylabel(r'$\langle \sigma_{xx} (t) \sigma_{xx} (0) \rangle / \langle \sigma_{xy}^2 \rangle$', fontsize=16)
sm = plt.cm.ScalarMappable(norm=phi_norm, cmap=cmap)
sm.set_array([0.7, 0.8, 0.9, 1.0])
cbar = plt.colorbar(sm, ax=ax[-1], label=r'$\phi$')
plt.savefig('disk-total-stress.png', dpi=600)
plt.close()
exit()

taus, phis = [], []

for phi_dir in os.listdir(data_root):
    corrs = np.load(os.path.join(data_root, phi_dir, 'corrs.npz'))
    phi_float = float(phi_dir.split('phi-')[-1])
    plt.plot(corrs['t'], corrs['isf_small'], c=cmap(phi_norm(phi_float)))

    taus.append(get_relaxation_time(corrs['isf_small'], corrs['t']))
    phis.append(phi_float)

plt.xscale('log')
    
plt.tight_layout()
os.makedirs('corr-figures-disk', exist_ok=True)
plt.tight_layout()
plt.savefig(f'corr-figures-disk/isf.png', dpi=600)
plt.close()

stress_phis = []
contact_tau, total_tau = [], []
for stress_file in os.listdir('stress-correlations-disk'):
    corrs = np.load(os.path.join('stress-correlations-disk', stress_file))
    phi_float = float(stress_file.split('phi-')[-1].split('.npz')[0])

    contact_tau.append(get_relaxation_time(corrs['contact_stress'][:, 0, 1], corrs['t']))
    total_tau.append(get_relaxation_time(corrs['stress'][:, 0, 1], corrs['t']))
    stress_phis.append(phi_float)

phis = np.array(phis)
taus = np.array(taus)

plt.scatter(phis[phis < 0.81], taus[phis < 0.81], c='k', marker='o', label='ISF')
plt.scatter(stress_phis, contact_tau, c='r', marker='x', label='Contact Stress')
plt.scatter(stress_phis, total_tau, c='g', marker='+', label='Stress')
plt.legend()
plt.xlabel(r'$\phi$', fontsize=16)
plt.ylabel(r'$\tau$', fontsize=16)
plt.yscale('log')
plt.tight_layout()
plt.savefig('corr-figures-disk/isf-tau.png', dpi=600)
