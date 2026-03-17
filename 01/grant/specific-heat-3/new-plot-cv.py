import os
from matplotlib.colors import LogNorm
import numpy as np
import matplotlib.pyplot as plt

root = '/home/mmccraw/dev/data/26-01-01/grant/specific-heat/'

compression_norm = plt.Normalize(0, 4)
cmap = plt.cm.viridis

mu_markers = {
    0.05: 'o',
    0.1: 's',
    0.5: 'x',
    1.0: '*',
}
alpha_norm = plt.Normalize(1, 3)
mu_norm = LogNorm(min(list(mu_markers.keys())), max(list(mu_markers.keys())))


mus = []
for which in os.listdir(root):
    _, mu, _, alpha = which.split('-')
    mu = float(mu)
    alpha = float(alpha)

    non_rattler_cvs = []
    rattler_cvs = []
    dphis = []
    cv_root = os.path.join(root, which, '0', '4', 'cv-2')
    if not os.path.exists(cv_root):
        continue
    rattler_data = np.load(os.path.join(cv_root, 'rattler_ids.npz'))
    rattler_ids = rattler_data['rattler_ids']
    non_rattler_ids = rattler_data['non_rattler_ids']

    for delta_phi in os.listdir(cv_root):
        if 'rattler_ids.npz' in delta_phi:
            continue
        data = np.load(os.path.join(cv_root, delta_phi))
                
        temp = data['temp']
        te = data['pe'] + data['ke'] + data['ke_r']


        C_v_non_rattler, _ = np.polyfit(
            np.mean(data['temp'], axis=0),
            np.mean(np.sum(te[..., non_rattler_ids], axis=(2)), axis=0),
            1
        )

        C_v_rattler, _ = np.polyfit(
            np.mean(data['temp'], axis=0),
            np.mean(np.sum(te[..., rattler_ids], axis=(2)), axis=0),
            1
        )

        if -data['delta_phi'] < 1e-10:
            continue
        non_rattler_cvs.append(C_v_non_rattler / len(non_rattler_ids))  # data['N']
        rattler_cvs.append(C_v_rattler / len(rattler_ids))  # data['N']
        dphis.append(-data['delta_phi'])
        mus.append(mu)

    dphis = np.array(dphis)
    non_rattler_cvs = np.array(non_rattler_cvs)[np.argsort(dphis)]
    rattler_cvs = np.array(rattler_cvs)[np.argsort(dphis)]
    dphis = np.sort(dphis)

    plt.plot(dphis, non_rattler_cvs, c=cmap(mu_norm(mu)))
    # plt.plot(dphis, rattler_cvs, ls='--', c=cmap(mu_norm(mu)))

# plt.plot([], [], c='k', label='Non-Rattlers')
# plt.plot([], [], ls='--', c='k', label='Rattlers')
# plt.legend()
plt.xscale('log')
sm = plt.cm.ScalarMappable(norm=mu_norm, cmap=cmap)
sm.set_array(np.unique(mus))
cbar = plt.colorbar(sm, ax=plt.gca(), label=r'$\mu_{eff}$')
plt.xscale('log')
plt.xlabel(r'$\Delta \phi$', fontsize=16)
plt.ylabel(r'$C_v / N$', fontsize=16)
plt.axhline(1.5, c='k', ls='--', alpha=0.5, zorder=0)
plt.axhline(3, c='k', ls='--', alpha=0.5, zorder=0)
plt.tight_layout()
plt.savefig('test.png')
plt.close()