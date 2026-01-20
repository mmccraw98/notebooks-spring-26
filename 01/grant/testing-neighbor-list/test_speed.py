import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from dataclasses import replace
from copy import deepcopy

import pandas as pd
import numpy as np

from bump_utils import animate

from tqdm import tqdm

import time

import os
data_root = '/home/mmccraw/dev/data/26-01-01/grant/profiling-cell-list/jamming'

def energies_one_snapshot(state, system):
    ids = state.ID                              # (N,)
    counts = jnp.bincount(ids, length=state.N)  # (N,)
    w = 1.0 / counts[ids]                       # (N,)

    ke_trans = 0.5 * jnp.sum(w * state.mass * jnp.sum(state.vel**2, axis=-1))
    ke_rot   = 0.5 * jnp.sum(w * jnp.sum(state.inertia * state.angVel**2, axis=-1))

    pe_per_sphere = system.collider.compute_potential_energy(state, system)  # (N,)
    pe_total = jnp.sum(pe_per_sphere)

    return pe_total, ke_trans, ke_rot

if __name__ == "__main__":
    run_time = []
    builds = []
    occupancy = []
    max_occupancy = []
    overflow = []
    skins_hist = []
    alpha = []
    mu = []
    nv_small = []
    nv_large = []
    vertex_diam_small = []
    vertex_diam_large = []
    vertex_rad = []
    true_packing_fraction = []
    packing_fraction = []
    throughput = []
    for particle_name in os.listdir(data_root)[::-1]:
        _, _mu, _, _alpha = particle_name.split('-')
        skins = [0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5]
        for skin in tqdm(skins):
            # load the data
            state = jd.utils.h5.load(os.path.join(data_root, particle_name, 'state.h5'))
            system = jd.utils.h5.load(os.path.join(data_root, particle_name, 'system.h5'))

            cids, nv = jnp.unique(state.ID, return_counts=True)

            unique_radii, radii_counts = np.unique(state.rad, return_counts=True)

            # slightly decompress
            delta_phi = -1e-3
            phi = jnp.sum(state.volume) / jnp.prod(system.domain.box_size)
            scale = (phi / (phi + delta_phi)) ** (1 / state.dim)
            state.pos_c *= scale
            system.domain.box_size *= scale

            # give a small amount of kinetic energy
            key = jax.random.key(np.random.randint(0, 1e9))
            vel = 1e-2 * jax.random.normal(key, shape=(cids.size, state.dim))
            state.vel = vel[state.ID]

            max_neighbors = 100
            max_diam = 2.0 * jnp.max(state.rad)
            cutoff = max_diam
            guessed_cell_size = cutoff * (1 + skin)

            cell_size = min(
                (system.domain.box_size / jnp.round(system.domain.box_size / guessed_cell_size))[0],
                (system.domain.box_size / 3)[0]  # clamp to at least 3 cells in each dimension
            )

            system = jd.System.create(
                state_shape=state.shape,
                dt=system.dt,
                linear_integrator_type="verlet",
                rotation_integrator_type="verletspiral",
                domain_type="periodic",
                force_model_type="spring",
                collider_type="neighborlist",
                collider_kw=dict(
                    state=state,
                    cutoff=cutoff,
                    skin=skin,
                    # max_neighbors=max_neighbors,
                    # cell_size=cell_size,
                ),
                mat_table=system.mat_table,
                domain_kw=dict(
                    box_size=system.domain.box_size,
                ),
            )

            n_steps = 1_000
            save_stride = 100
            n_snapshots = n_steps // save_stride
            start = time.time()
            state, system, (state_traj, system_traj) = system.trajectory_rollout(
                state, system, n=n_snapshots, stride=save_stride
            )
            # with jax.profiler.trace(f'profiling/{particle_name}-{skin}'):
            #     state, system, (state_traj, system_traj) = system.trajectory_rollout(
            #         state, system, n=n_snapshots, stride=save_stride
            #     )
            #     state_traj.pos.block_until_ready()
            run_time.append(time.time() - start)
            builds.append(system.collider.n_build_times)
            occupancy.append(jnp.mean(jnp.sum(system_traj.collider.neighbor_list != -1, axis=-1)))
            max_occupancy.append(jnp.max(jnp.sum(system_traj.collider.neighbor_list != -1, axis=-1)))
            overflow.append(jnp.any(jnp.sum(system_traj.collider.neighbor_list != -1, axis=-1) == max_neighbors))
            skins_hist.append(skin)
            alpha.append(_alpha)
            mu.append(_mu)
            nv_small.append(jnp.min(nv))
            nv_large.append(jnp.max(nv))
            vertex_diam_small.append(jnp.min(state.rad))
            vertex_diam_large.append(jnp.max(state.rad))
            true_packing_fraction.append(jnp.pi * jnp.sum(state.rad ** 2) / jnp.prod(system.domain.box_size))
            packing_fraction.append(jnp.sum(state.volume) / jnp.prod(system.domain.box_size))
            vertex_rad.append(unique_radii[np.argmax(radii_counts)])
            throughput.append(state.N * n_steps / (run_time[-1]))

    df = pd.DataFrame({
        'run_time': run_time,
        'builds': builds,
        'occupancy': occupancy,
        'max_occupancy': max_occupancy,
        'overflow': overflow,
        'skins': skins_hist,
        'alpha': alpha,
        'mu': mu,
        'nv_small': nv_small,
        'nv_large': nv_large,
        'vertex_diam_small': vertex_diam_small,
        'vertex_diam_large': vertex_diam_large,
        'vertex_rad': vertex_rad,
        'true_packing_fraction': true_packing_fraction,
        'packing_fraction': packing_fraction,
        'throughput': throughput,
    })
    df.to_csv('profile_data.csv', index=False)
