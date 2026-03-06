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

for which in os.listdir(root):
    for run_id in os.listdir(os.path.join(root, which)):
        for compression_step in os.listdir(os.path.join(root, which, run_id)):
            cvs = []
            dphis = []
            cv_root = os.path.join(root, which, run_id, compression_step, 'cv')
            if not os.path.exists(cv_root):
                continue
            for delta_phi in os.listdir(cv_root):
                data = np.load(os.path.join(cv_root, delta_phi))
                        
                temp = data['temp']
                te = data['pe'] + data['ke'] + data['ke_r']

                C_v, _ = np.polyfit(
                    np.mean(data['temp'], axis=0),
                    np.mean(data['pe'] + data['ke'] + data['ke_r'], axis=0),
                    1
                )

                cvs.append(C_v / data['N'])
                dphis.append(-data['delta_phi'])

            dphis = np.array(dphis)
            cvs = np.array(cvs)[np.argsort(dphis)]
            dphis = np.sort(dphis)

            plt.plot(dphis, cvs, c=cmap(compression_norm(int(compression_step))))
        plt.xscale('log')
        plt.xlabel(r'$\Delta \phi$', fontsize=16)
        plt.ylabel(r'$C_v / N$', fontsize=16)
        os.makedirs(os.path.join('figures', which), exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join('figures', which, 'change-with-compression.png'), dpi=600)
        plt.close()


mus, alphas, cv_lists, dphi_lists = [], [], [], []
for which in os.listdir(root):
    _, mu, _, alpha = which.split('-')
    mu = float(mu)
    alpha = float(alpha)
    for run_id in os.listdir(os.path.join(root, which)):
        for compression_step in os.listdir(os.path.join(root, which, run_id)):
            if compression_step != '4':
                continue
            cvs = []
            dphis = []
            cv_root = os.path.join(root, which, run_id, compression_step, 'cv')
            if not os.path.exists(cv_root):
                continue
            for delta_phi in os.listdir(cv_root):
                data = np.load(os.path.join(cv_root, delta_phi))
                        
                temp = data['temp']
                te = data['pe'] + data['ke'] + data['ke_r']

                C_v, _ = np.polyfit(
                    np.mean(data['temp'], axis=0),
                    np.mean(data['pe'] + data['ke'] + data['ke_r'], axis=0),
                    1
                )

                cvs.append(C_v / data['N'])
                dphis.append(-data['delta_phi'])

                cv_lists.append(C_v / data['N'])
                dphi_lists.append(-data['delta_phi'])
                mus.append(mu)
                alphas.append(alpha)

            dphis = np.array(dphis)
            cvs = np.array(cvs)[np.argsort(dphis)]
            dphis = np.sort(dphis)

            plt.plot(dphis, cvs, c=cmap(alpha_norm(alpha)), marker=mu_markers[mu])

sm = plt.cm.ScalarMappable(norm=alpha_norm, cmap=cmap)
sm.set_array(list(range(1, 4)))
cbar = plt.colorbar(sm, ax=plt.gca(), label=r'$\alpha$')

for mu, marker in mu_markers.items():
    plt.plot([], [], c='k', marker=marker, label=fr'$\mu=${mu}')
plt.legend()

plt.axhline(1.5, c='k', ls='--', alpha=0.5, zorder=0)
plt.axhline(3, c='k', ls='--', alpha=0.5, zorder=0)

plt.xscale('log')
# plt.yscale('log')
plt.xlabel(r'$\Delta \phi$', fontsize=16)
plt.ylabel(r'$C_v / N$', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join('figures', 'all-dense-case.png'), dpi=600)
plt.close()

alphas = np.array(alphas)
mus = np.array(mus)
cv_lists = np.array(cv_lists)
dphi_lists = np.array(dphi_lists)

for alpha in np.unique(alphas):
    for mu in np.unique(mus):
        mask = (alphas == alpha) & (mus == mu)
        x = dphi_lists[mask]
        y = cv_lists[mask]
        plt.plot(x[np.argsort(x)], y[np.argsort(x)], c=cmap(mu_norm(mu)))
    sm = plt.cm.ScalarMappable(norm=mu_norm, cmap=cmap)
    sm.set_array(np.unique(mus))
    cbar = plt.colorbar(sm, ax=plt.gca(), label=r'$\mu_{eff}$')
    plt.xscale('log')
    plt.xlabel(r'$\Delta \phi$', fontsize=16)
    plt.ylabel(r'$C_v / N$', fontsize=16)
    plt.axhline(1.5, c='k', ls='--', alpha=0.5, zorder=0)
    plt.axhline(3, c='k', ls='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', f'alpha-{alpha}.png'), dpi=600)
    plt.close()

for mu in np.unique(mus):
    for alpha in np.unique(alphas):
        mask = (alphas == alpha) & (mus == mu)
        x = dphi_lists[mask]
        y = cv_lists[mask]
        plt.plot(x[np.argsort(x)], y[np.argsort(x)], c=cmap(alpha_norm(alpha)))
    sm = plt.cm.ScalarMappable(norm=alpha_norm, cmap=cmap)
    sm.set_array(np.unique(alphas))
    cbar = plt.colorbar(sm, ax=plt.gca(), label=r'$\alpha$')
    plt.xscale('log')
    plt.xlabel(r'$\Delta \phi$', fontsize=16)
    plt.ylabel(r'$C_v / N$', fontsize=16)
    plt.axhline(1.5, c='k', ls='--', alpha=0.5, zorder=0)
    plt.axhline(3, c='k', ls='--', alpha=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(os.path.join('figures', f'mu-{mu}.png'), dpi=600)
    plt.close()

for which in os.listdir(root):
    _, mu, _, alpha = which.split('-')
    mu = float(mu)
    alpha = float(alpha)
    for run_id in os.listdir(os.path.join(root, which)):
        for compression_step in os.listdir(os.path.join(root, which, run_id)):
            if compression_step != '0':
                continue
            cvs = []
            dphis = []
            cv_root = os.path.join(root, which, run_id, compression_step, 'cv')
            if not os.path.exists(cv_root):
                continue
            for delta_phi in os.listdir(cv_root):
                data = np.load(os.path.join(cv_root, delta_phi))
                        
                temp = data['temp']
                te = data['pe'] + data['ke'] + data['ke_r']

                C_v, _ = np.polyfit(
                    np.mean(data['temp'], axis=0),
                    np.mean(data['pe'] + data['ke'] + data['ke_r'], axis=0),
                    1
                )

                cvs.append(C_v / data['N'])
                dphis.append(-data['delta_phi'])

            dphis = np.array(dphis)
            cvs = np.array(cvs)[np.argsort(dphis)]
            dphis = np.sort(dphis)

            plt.plot(dphis, 3 - cvs, c=cmap(alpha_norm(alpha)), marker=mu_markers[mu])

sm = plt.cm.ScalarMappable(norm=alpha_norm, cmap=cmap)
sm.set_array(list(range(1, 4)))
cbar = plt.colorbar(sm, ax=plt.gca(), label=r'$\alpha$')

for mu, marker in mu_markers.items():
    plt.plot([], [], c='k', marker=marker, label=fr'$\mu=${mu}')
plt.legend()

plt.axhline(0.5, c='k', ls='--', alpha=0.5, zorder=0)

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\Delta \phi$', fontsize=16)
plt.ylabel(r'$3 - C_v / N$', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join('figures', 'all-dilute-case.png'), dpi=600)
plt.close()