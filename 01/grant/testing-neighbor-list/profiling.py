import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from dataclasses import replace
from copy import deepcopy

import time

import os
data_root = '/home/mmccraw/dev/data/26-01-01/grant/profiling-cell-list/jamming'
n_repeats = 10

if __name__ == "__main__":
    for particle_name in os.listdir(data_root)[::-1]:
        
        # load the data
        state = jd.utils.h5.load(os.path.join(data_root, particle_name, 'state.h5'))
        system = jd.utils.h5.load(os.path.join(data_root, particle_name, 'system.h5'))

        state.force *= 0.0

        # slightly compress to force contacts
        delta_phi = 1e-2
        phi = jnp.sum(state.volume) / jnp.prod(system.domain.box_size)
        scale = (phi / (phi + delta_phi)) ** (1 / state.dim)
        state.pos_c *= scale
        system.domain.box_size *= scale

        # calculate the expected forces and torques from the true solution (naive collider)
        _, offsets = jnp.unique(state.ID, return_index=True)
        state, system = system.collider.compute_force(state, system)
        baseline_forces = state.force.copy()[offsets]
        baseline_torques = state.torque.copy()[offsets]

        skin = 0.3
        max_neighbors = 100
        max_diam = 2.0 * jnp.max(state.rad)
        cutoff = max_diam
        guessed_cell_size = cutoff * (1 + skin)
        rebuild_threshold = cutoff / 2.0

        cell_size = min(
            (system.domain.box_size / jnp.round(system.domain.box_size / guessed_cell_size))[0],
            (system.domain.box_size / 3)[0]  # clamp to at least 3 cells in each dimension
        )

        starting_cell_size = deepcopy(cell_size)

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


        start = time.time()
        for i in range(n_repeats):
            inner_system = replace(system, collider=system.collider.cell_list)
            inner_state, inner_system, neighbor_list, overflow = system.collider.cell_list.create_neighbor_list(state, inner_system, cutoff * (1 + skin), max_neighbors)
        build_time = (time.time() - start) / n_repeats

        start = time.time()
        for i in range(n_repeats):
            state.force *= 0.0
            state, system = system.collider.compute_force(state, system)
        force_time = (time.time() - start) / n_repeats

        # do any sorting
        perm = jnp.argsort(state.unique_ID)
        state = jax.tree.map(lambda x: x[perm], state)

        # calculate forces to compare to the naive case
        _, offsets = jnp.unique(state.ID, return_index=True)
        state, system = system.collider.compute_force(state, system)
        forces = state.force.copy()[offsets]
        torques = state.torque.copy()[offsets]

        print(forces)
        print(baseline_forces)


        # print(inner_system.collider.cell_size)
        # print(inner_system.collider.neighbor_mask.shape)
        # print(neighbor_list)
        # state, system = system.collider.compute_force(state, system)
        # neighbor_list = system.collider.neighbor_list
        # print(neighbor_list)
        # print(jnp.sum(neighbor_list != -1, axis=1))


        exit()




    _, offsets = jnp.unique(state.ID, return_index=True)

    print(jnp.sum(state.force[offsets], axis=0))
    neighbor_list = system.collider.neighbor_list


    # system = new_system
    # state = new_state
