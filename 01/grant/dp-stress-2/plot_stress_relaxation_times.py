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
    # if not np.min(f) < 1 / np.e:
    if not np.min(f) < 0.01:
        return np.nan
    return tau

tau_disk, tau_disk_total, phi_disk = [], [], []
dt = 1e-2
data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-5/compression-disks-1/'
for stress_file in os.listdir('stress-correlations-disk'):
    corrs = np.load(os.path.join('stress-correlations-disk', stress_file))
    phi_float = float(stress_file.split('phi-')[-1].split('.npz')[0])
    mask = corrs['t'] < corrs['t'].max() / 10
    tau_disk.append(get_relaxation_time(corrs['contact_stress'][:, 0, 1][mask] / corrs['contact_stress'][0, 0, 1], corrs['t'][mask]))
    tau_disk_total.append(get_relaxation_time(corrs['stress'][:, 0, 1][mask] / corrs['stress'][0, 0, 1], corrs['t'][mask]))
    phi_disk.append(phi_float)
phi_disk = np.array(phi_disk)
tau_disk = np.array(tau_disk)
tau_disk_total = np.array(tau_disk_total)

tau_dp, tau_dp_bond, tau_dp_total, phi_dp = [], [], [], []
dt = 1e-3
data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-5/compression-4/low'
for stress_file in os.listdir('stress-correlations-4'):
    corrs = np.load(os.path.join('stress-correlations-4', stress_file))
    phi_float = float(stress_file.split('phi-')[-1].split('.npz')[0])
    mask = corrs['t'] < corrs['t'].max() / 10
    tau_dp.append(get_relaxation_time(corrs['contact_stress'][:, 0, 1][mask] / corrs['contact_stress'][0, 0, 1], corrs['t'][mask]))
    tau_dp_bond.append(get_relaxation_time(corrs['bonded_stress'][:, 0, 1][mask] / corrs['bonded_stress'][0, 0, 1], corrs['t'][mask]))
    tau_dp_total.append(get_relaxation_time(corrs['stress'][:, 0, 1][mask] / corrs['stress'][0, 0, 1], corrs['t'][mask]))
    phi_dp.append(phi_float)
phi_dp = np.array(phi_dp)
tau_dp = np.array(tau_dp)
tau_dp_bond = np.array(tau_dp_bond)
tau_dp_total = np.array(tau_dp_total)

fig, ax = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)

ax[0].scatter(phi_disk[tau_disk < 200], tau_disk[tau_disk < 200], label=r'$\tau_C$', c='k', marker='x')
# ax[0].scatter(phi_disk[tau_disk_total < 200], tau_disk_total[tau_disk_total < 200], label=r'$\tau$')

ax[1].scatter(phi_dp, tau_dp, label=r'$\tau_C$', c='k', marker='x')
ax[1].scatter(phi_dp, tau_dp_bond, label=r'$\tau_S$', c='r', marker='s')
# ax[1].scatter(phi_dp, tau_dp_total, label=r'$\tau$')

for a in ax:
    a.set_xlabel(r'$\phi$', fontsize=16)
    a.set_ylabel(r'$\tau$', fontsize=16)
    a.legend()

plt.savefig('stress-relaxation-times.png', dpi=600)
plt.close()