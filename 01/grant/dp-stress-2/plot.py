import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import jax.numpy as jnp

from jaxdem.analysis import LagBinsPseudoLog, evaluate_binned, LagBinsLinear

from jaxdem_scripts.correlations import get_pseudo_log_bins_from_steps

def stress_kernel(arrays, t0, t1):
    stress0 = arrays["stress_tensor"][t0]
    stress1 = arrays["stress_tensor"][t1]
    return stress0 * stress1

phi_norm = plt.Normalize(0.6, 1.0)
cmap = plt.cm.viridis

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("name")
args = parser.parse_args()
name = args.name

dt = 1e-3

phis = []
bonded_0 = []
contact_0 = []

for path in tqdm(os.listdir(os.path.join('virials-4', name))):
    fig, ax = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    data = np.load(os.path.join('virials-4', name, path))

    step_ids = data['step_ids']
    area = data['area']
    bv = data['bonded_virial'] / area
    cv = data['contact_virial'] / area
    kv = data['kinetic_virial'] / area
    phi = data['phi']
    v = bv + cv + kv

    bins, t = get_pseudo_log_bins_from_steps(step_ids, dt)


    # bins = LagBinsLinear(step_ids.shape[0], dt_min=1, dt_max=int(step_ids[-1] - step_ids[0]), step=1, timestep=step_ids)
    # t = bins.values() * dt

    bonded_stress = evaluate_binned(stress_kernel, {"stress_tensor": bv}, bins).mean
    contact_stress = evaluate_binned(stress_kernel, {"stress_tensor": cv}, bins).mean
    stress = evaluate_binned(stress_kernel, {"stress_tensor": v}, bins).mean
    print(bonded_stress.shape)

    contact_0.append(contact_stress[0, 0, 1])
    bonded_0.append(bonded_stress[0, 0, 1])
    phis.append(phi)

    for i, (c, plot_name) in enumerate(zip([bonded_stress, contact_stress, stress], ['Bonded', 'Contact', 'Total'])):
        # ax[i].plot(t, c[:, 0, 0])
        ax[i].plot(t * dt, c[:, 0, 1], c=cmap(phi_norm(phi)))
        ax[i].set_title(plot_name)
    
    for a in ax:
        a.set_xscale('log')
        # a.set_yscale('log')
        a.set_xlabel(r'$t$', fontsize=16)
        a.set_ylabel(r'$\langle \sigma_{xy}(t) \sigma_{xy}(0) \rangle$', fontsize=16)

    sm = plt.cm.ScalarMappable(norm=phi_norm, cmap=cmap)
    sm.set_array([0.6, 0.7, 0.8, 0.9, 1.0])
    cbar = plt.colorbar(sm, ax=ax[-1], label=r'$\phi$')

    os.makedirs(f'figures-4/{name}', exist_ok=True)
    plt.savefig(f'figures-4/{name}/{path.split(".npz")[0]}.png', dpi=600)
    plt.close()

plt.scatter(phis, contact_0, c='b', label='Contact')
plt.scatter(phis, bonded_0, c='k', marker='x', label='Bonded')
plt.yscale('log')
plt.xlabel(r'$\phi$', fontsize=16)
plt.ylabel(r'$\langle \sigma_{xy}^2 \rangle$', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('stresses_0.png', dpi=600)
plt.close()