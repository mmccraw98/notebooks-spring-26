import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from dataclasses import replace
from copy import deepcopy

import os
data_root = '/home/mmccraw/dev/data/26-01-01/grant/profiling-cell-list/jamming'

if __name__ == "__main__":
    for particle_name in os.listdir(data_root)[::-1]:
        

        state = jd.utils.h5.load(os.path.join(data_root, particle_name, 'state.h5'))
        system = jd.utils.h5.load(os.path.join(data_root, particle_name, 'system.h5'))

        delta_phi = 1e-2
        phi = jnp.sum(state.volume) / jnp.prod(system.domain.box_size)
        scale = (phi / (phi + delta_phi)) ** (1 / state.dim)
        state.pos_c *= scale
        system.domain.box_size *= scale

        skin = 0.3
        max_neighbors = 100
        max_diam = 2.0 * jnp.max(state.rad)
        guessed_cell_size = max_diam * (1 + skin)
        # cutoff = skin * max_diam
        cutoff = guessed_cell_size
        rebuild_threshold = cutoff / 2.0
        print(cutoff)

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

        # exit()
    
        inner_system = replace(system, collider=system.collider.cell_list)
        state, system, neighbor_list, overflow = system.collider.cell_list.create_neighbor_list(state, inner_system, cutoff, max_neighbors)
        # print(inner_system.collider.cell_size)
        # print(inner_system.collider.neighbor_mask.shape)
        # print(neighbor_list)
        # state, system = system.collider.compute_force(state, system)
        # neighbor_list = system.collider.neighbor_list
        print(neighbor_list)
        print(jnp.sum(neighbor_list != -1, axis=1))

        pid_i = 20

        from matplotlib.patches import Circle
        def plot_grid(dim):
            i = 0
            while True:
                if dim == 0:
                    func = plt.axhline
                else:
                    func = plt.axvline
                func(i * starting_cell_size, zorder=0, color='k', alpha=0.5)
                i += 1
                if i * starting_cell_size > system.domain.box_size[dim]:
                    break
        plt.xlim(0, system.domain.box_size[0])
        plt.ylim(0, system.domain.box_size[1])
        plot_grid(0)
        plot_grid(1)
        plt.gca().set_aspect('equal')
        plt.gca().add_artist(Circle(jnp.mod(state.pos[pid_i], system.domain.box_size), cutoff, facecolor='k', alpha=0.2))
        for p, r in zip(jnp.mod(state.pos, system.domain.box_size), state.rad):
            plt.gca().add_artist(Circle(p, r))
        plt.gca().add_artist(Circle(jnp.mod(state.pos[pid_i], system.domain.box_size), state.rad[pid_i], facecolor='k'))
        for i in neighbor_list[pid_i]:
            if i == -1:
                continue
            plt.gca().add_artist(Circle(jnp.mod(state.pos[i], system.domain.box_size), state.rad[i], facecolor='r'))
        # p = jnp.mod(state.pos, system.domain.box_size)
        # f = state.force / jnp.linalg.norm(state.force, axis=-1, keepdims=True)
        # plt.quiver(p[:, 0], p[:, 1], f[:, 0], f[:, 1])
        plt.savefig('test.png')


        exit()

