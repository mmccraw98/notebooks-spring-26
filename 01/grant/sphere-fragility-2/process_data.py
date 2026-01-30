import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
import argparse
import sys

from file_management import make_data_dir, save_arrs, load_arrs

from tqdm import tqdm

from jaxdem.analysis import LagBinsPseudoLog, evaluate_binned
from jaxdem.analysis.kernels import isf_self_isotropic_kernel

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def monte_carlo_sample_spheres(state, system, S_total, chunk, seed):
    S_total = int(S_total)
    chunk = int(chunk)
    key0 = jax.random.PRNGKey(seed)
    bounds = system.domain.box_size
    n_steps = S_total // chunk
    assert (S_total % chunk) == 0
    rad2 = state.rad[:, None] ** 2
    def body(i, hits):
        key_i = jax.random.fold_in(key0, i)
        samples = jax.random.uniform(key_i, shape=(chunk, state.dim)) * bounds
        disp = system.domain.displacement(
            state.pos[:, None, :],
            samples[None, :, :],
            system
        )
        r2 = jnp.sum(disp * disp, axis=-1)
        inside_any = jnp.any(r2 <= rad2, axis=0)
        return hits + jnp.sum(inside_any)
    hits = jax.lax.fori_loop(0, n_steps, body, 0)
    return hits / S_total


def get_relaxation_time(f, t):
    tau = np.interp(1 / np.e, f[::-1], t[::-1])
    if not np.min(f) < 1 / np.e:
        return np.nan
    return tau

which = '2d-2'
if which == '2d':
    from config import config2d as cfg
elif which == '2d-2':
    from config import config2d_2 as cfg
elif which == '3d':
    from config import config3d as cfg
else:
    raise ValueError(f'Which {which} is unknown')

root = f'/home/mmccraw/dev/data/26-01-01/grant/sphere-fragilitiy/version-2/{which}'

phis_hist = []
true_phis_hist = []
temperatures_hist = []
taus_hist = []
isfs_hist = []
ts_hist = []

for phi_dir in tqdm(os.listdir(root)):
    path = os.path.join(root, phi_dir)
    phi = phi_dir.split('phi-')[-1]
    try:
        state = jd.utils.h5.load(os.path.join(path, 'init', 'state.h5'))
        system = jd.utils.h5.load(os.path.join(path, 'init', 'system.h5'))
        traj = load_arrs(os.path.join(path, 'traj', 'data.h5'))
    except Exception as e:
        continue
    
    pos = traj['pos']
    T = pos.shape[0]

    bins = LagBinsPseudoLog(T, dt_min=1, dt_max=T-1)  # pseudo-log lags
    k = 2.0 * jnp.pi / (2 * jnp.min(state.rad[0]))
    temp = cfg.target_temperatures

    res = evaluate_binned(isf_self_isotropic_kernel, {"pos": pos}, bins, kernel_kwargs={"k": k})
    t = bins.values()

    t_dim = t[:, None] * cfg.dt * jnp.sqrt(temp[None, :])
    Fs = np.array(res.mean)
    for i, (st, sy) in enumerate(zip(
        jd.State.unstack(state),
        jd.System.unstack(system),
    )):
        tau = get_relaxation_time(Fs[:, i], t_dim[:, i])
        isfs_hist.append(Fs[:, i])
        ts_hist.append(t_dim[:, i])
        
        phis_hist.append(float(phi))
        temperatures_hist.append(cfg.target_temperatures[i])
        taus_hist.append(tau)

        true_phis_hist.append(
            monte_carlo_sample_spheres(
                st,
                sy,
                1e8,
                1e4,
                np.random.randint(0, 1e9)
            )
        )

        plt.axvline(tau)
        plt.plot(t_dim[:, i], Fs[:, i])
    plt.xscale('log')
    if not os.path.exists(f'figures/isf/{which}'):
        os.makedirs(f'figures/isf/{which}')
    plt.savefig(f'figures/isf/{which}isf-{phi}.png')
    plt.close()

phis_hist = np.array(phis_hist)
true_phis_hist = np.array(true_phis_hist)
temperatures_hist = np.array(temperatures_hist)
taus_hist = np.array(taus_hist)
ts_hist = np.array(ts_hist)
isfs_hist = np.array(isfs_hist)

np.savez(
    f'aggregated_data_{which}.npz',
    phi=phis_hist,
    true_phi=true_phis_hist,
    temperature=temperatures_hist,
    tau=taus_hist,
    t=ts_hist,
    isf=isfs_hist,
)