import jax
from matplotlib.colors import LogNorm
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy as sp

def get_eimask(_vals, eps=1e-12):
    vals = np.abs(_vals.copy())
    order = np.argsort(vals)
    vals = vals[order]
    min_val = max(vals[np.argmax(vals[1:] / vals[:-1])], vals[-1] * eps)
    return vals > min_val

alpha_norm = plt.Normalize(1, 3)
mu_norm = LogNorm(0.05, 1.0)
cmap = plt.cm.viridis

mu_markers = {
    0.05: 'o',
    0.1: 's',
    0.5: 'x',
    1.0: '*',
}

for alpha_target in [1.0, 1.5, 2.0, 2.5]:
    for which in os.listdir('modes'):
        if f"alpha-{alpha_target}" not in which:
            continue

        data = np.load(f'modes/{which}')
        _, mu, _, alpha = which.split('.npz')[0].split('-')
        mu = float(mu)
        alpha = float(alpha)

        H = data['H']
        M = data['M']

        N_c = H.shape[0] // 3

        # vals, vecs = sp.linalg.eigh(H, M)  # mass orthonormalization

        # DO NOT USE THE MASS ORTHONORMALIZATION FOR THE FOLLOWING:
        vals, vecs = sp.linalg.eigh(H)
        modes = vecs.T.reshape(-1, N_c, 3)

        # calculate the translational and rotational content
        trans = jnp.sum(modes[..., :-1] ** 2, axis=(-1, -2))
        rot = 1 - trans

        mask = get_eimask(vals) & (rot / trans > 1e-8)
        omega = np.sqrt(vals[mask])
        ratio = (rot / trans)[mask]

        # plt.plot(omega, ratio, marker=mu_markers[mu], c=cmap(alpha_norm(alpha)))
        plt.plot(omega, ratio, c=cmap(mu_norm(mu)))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\omega$', fontsize=16)
    plt.ylabel(r'$R/T$', fontsize=16)
    sm = plt.cm.ScalarMappable(norm=mu_norm, cmap=cmap)
    sm.set_array([0.05, 0.1, 0.5, 1.0])
    cbar = plt.colorbar(sm, ax=plt.gca(), label=r'$\mu_{eff}$')
    plt.tight_layout()
    plt.savefig(f'figures/alpha-{alpha}-ratios.png')
    plt.close()

for alpha_target in [1.0, 1.5, 2.0, 2.5]:
    for which in os.listdir('modes'):
        if f"alpha-{alpha_target}" not in which:
            continue

        data = np.load(f'modes/{which}')
        _, mu, _, alpha = which.split('.npz')[0].split('-')
        mu = float(mu)
        alpha = float(alpha)

        H = data['H']
        M = data['M']

        N_c = H.shape[0] // 3

        # vals, vecs = sp.linalg.eigh(H, M)  # mass orthonormalization

        # DO NOT USE THE MASS ORTHONORMALIZATION FOR THE FOLLOWING:
        vals, vecs = sp.linalg.eigh(H)
        modes = vecs.T.reshape(-1, N_c, 3)

        # calculate the participation ratios
        P = 1.0 / (N_c * jnp.sum(jnp.sum(modes ** 2, axis=-1) ** 2, axis=-1))

        mask = get_eimask(vals) & (jnp.sqrt(vals) > 1e-4)
        omega = np.sqrt(vals[mask])
        P = P[mask]

        plt.plot(omega, P, c=cmap(mu_norm(mu)))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\omega$', fontsize=16)
    plt.ylabel(r'$P$', fontsize=16)
    sm = plt.cm.ScalarMappable(norm=mu_norm, cmap=cmap)
    sm.set_array([0.05, 0.1, 0.5, 1.0])
    cbar = plt.colorbar(sm, ax=plt.gca(), label=r'$\mu_{eff}$')
    plt.tight_layout()
    plt.savefig(f'figures/alpha-{alpha}-participation.png')
    plt.close()

for alpha_target in [1.0, 1.5, 2.0, 2.5]:
    for which in os.listdir('modes'):
        if f"alpha-{alpha_target}" not in which:
            continue

        data = np.load(f'modes/{which}')
        _, mu, _, alpha = which.split('.npz')[0].split('-')
        mu = float(mu)
        alpha = float(alpha)

        H = data['H']
        M = data['M']

        N_c = H.shape[0] // 3

        # vals, vecs = sp.linalg.eigh(H, M)  # mass orthonormalization

        # DO NOT USE THE MASS ORTHONORMALIZATION FOR THE FOLLOWING:
        vals, vecs = sp.linalg.eigh(H)
        modes = vecs.T.reshape(-1, N_c, 3)

        # calculate the optical order parameter
        t = modes[..., -1]
        Q_opt = jnp.sum(t[:, :, None] * t[:, None, :], axis=(-1, -2)) / (N_c * jnp.sum(modes[..., 2] ** 2, axis=(-1)))

        mask = get_eimask(vals) & (jnp.sqrt(vals) > 1e-4)
        omega = np.sqrt(vals[mask])
        Q_opt = Q_opt[mask]

        plt.plot(omega, Q_opt, c=cmap(mu_norm(mu)))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\omega$', fontsize=16)
    plt.ylabel(r'$Q_{opt}$', fontsize=16)
    sm = plt.cm.ScalarMappable(norm=mu_norm, cmap=cmap)
    sm.set_array([0.05, 0.1, 0.5, 1.0])
    cbar = plt.colorbar(sm, ax=plt.gca(), label=r'$\mu_{eff}$')
    plt.tight_layout()
    plt.savefig(f'figures/alpha-{alpha}-Q_opt.png')
    plt.close()

for alpha_target in [1.0, 1.5, 2.0, 2.5]:
    for which in os.listdir('modes'):
        if f"alpha-{alpha_target}" not in which:
            continue

        data = np.load(f'modes/{which}')
        _, mu, _, alpha = which.split('.npz')[0].split('-')
        mu = float(mu)
        alpha = float(alpha)

        H = data['H']
        M = data['M']

        N_c = H.shape[0] // 3

        # vals, vecs = sp.linalg.eigh(H, M)  # mass orthonormalization

        # DO NOT USE THE MASS ORTHONORMALIZATION FOR THE FOLLOWING:
        vals, vecs = sp.linalg.eigh(H)
        modes = vecs.T.reshape(-1, N_c, 3)

        mask = get_eimask(vals) & (vals > 0)
        omega = np.sqrt(vals[mask])

        D, edges = np.histogram(omega, bins=int(np.sqrt(omega.size)), density=True)
        x = (edges[1:] + edges[:-1]) / 2

        plt.plot(x, D, c=cmap(mu_norm(mu)))
    plt.xlabel(r'$\omega$', fontsize=16)
    plt.ylabel(r'$D(\omega)$', fontsize=16)
    sm = plt.cm.ScalarMappable(norm=mu_norm, cmap=cmap)
    sm.set_array([0.05, 0.1, 0.5, 1.0])
    cbar = plt.colorbar(sm, ax=plt.gca(), label=r'$\mu_{eff}$')
    plt.tight_layout()
    plt.savefig(f'figures/alpha-{alpha}-D_omega.png')
    plt.close()