import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

phi_norm = plt.Normalize(0.6, 1.0)
cmap = plt.cm.viridis

fig, ax = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)

dt = 1e-3

phis = []
bonded_0 = []
contact_0 = []

for path in tqdm(os.listdir()):
    if 'sphere-virials-' not in path:
        continue
    data = np.load(path)

    area = data['area']
    cv = data['contact_virial'] / area
    kv = data['kinetic_virial'] / area
    phi = data['phi']
    v = cv + kv
    t = []
    contact_stress, stress = [], []
    for i in range(1, cv.shape[0] - 1):
        t.append(i)
        contact_stress.append(np.mean(cv[i:] * cv[:-i], axis=0))
        stress.append(np.mean(v[i:] * v[:-i], axis=0))

    t = np.array(t)
    contact_stress = np.array(contact_stress)
    stress = np.array(stress)

    contact_0.append(contact_stress[0, 0, 1])
    phis.append(phi)

    for i, (c, name) in enumerate(zip([contact_stress, stress], ['Contact', 'Total'])):
        # ax[i].plot(t, c[:, 0, 0])
        ax[i].plot(t * dt, abs(c[:, 0, 1]), c=cmap(phi_norm(phi)))
        ax[i].set_title(name)
for a in ax:
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_xlabel(r'$t$', fontsize=16)
    a.set_ylabel(r'$|\langle \sigma_{xy}(t) \sigma_{xy}(0) \rangle|$', fontsize=16)

sm = plt.cm.ScalarMappable(norm=phi_norm, cmap=cmap)
sm.set_array([0.6, 0.7, 0.8, 0.9, 1.0])
cbar = plt.colorbar(sm, ax=ax[-1], label=r'$\phi$')

plt.savefig('sphere-stresses.png', dpi=600)
plt.close()

plt.scatter(phis, contact_0, c='b', label='Contact')
plt.yscale('log')
plt.xlabel(r'$\phi$', fontsize=16)
plt.ylabel(r'$\langle \sigma_{xy}^2 \rangle$', fontsize=16)
plt.legend()
plt.tight_layout()
plt.savefig('sphere-stresses_0.png', dpi=600)
plt.close()