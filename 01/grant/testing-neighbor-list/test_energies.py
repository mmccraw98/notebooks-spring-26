import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from dataclasses import replace
from copy import deepcopy

from bump_utils import animate
import numpy as np

import time

import os
data_root = '/home/mmccraw/dev/data/26-01-01/grant/profiling-cell-list/jamming'

# particles * steps / total time
# ^ should be close to 10m

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
    for particle_name in os.listdir(data_root)[::-1]:
        
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

        skin = 0.03
        max_neighbors = 100
        max_diam = 2.0 * jnp.max(state.rad)
        cutoff = max_diam
        guessed_cell_size = cutoff * (1 + skin)
        rebuild_threshold = cutoff / 2.0

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
                max_neighbors=max_neighbors,
                cell_size=cell_size,
            ),
            mat_table=system.mat_table,
            domain_kw=dict(
                box_size=system.domain.box_size,
            ),
        )

        n_steps = 10_000
        save_stride = 100
        n_snapshots = n_steps // save_stride
        state, system, (state_traj, system_traj) = system.trajectory_rollout(
            state, system, n=n_snapshots, stride=save_stride
        )

        animate(state_traj, system_traj, f'animations/{particle_name}.gif')

        pe, ke_trans, ke_rot = jax.vmap(energies_one_snapshot, in_axes=(0, 0))(state_traj, system_traj)
        plt.plot(pe + ke_trans + ke_rot, label='total')
        plt.plot(pe, label='pe')
        plt.plot(ke_trans, label='translation')
        plt.plot(ke_rot, label='rotation')
        plt.legend()
        plt.savefig(f'energies/{particle_name}.png')
        plt.close()
