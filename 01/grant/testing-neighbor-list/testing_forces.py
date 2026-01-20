import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

if __name__ == "__main__":
    state = jd.utils.h5.load('example-state.h5')
    system = jd.utils.h5.load('example-system.h5')

    delta_phi = 1e-2
    phi = jnp.sum(state.volume) / jnp.prod(system.domain.box_size)
    scale = (phi / (phi + delta_phi)) ** (1 / state.dim)
    state.pos_c *= scale
    system.domain.box_size *= scale



    # CREATE CELL LIST
    # min_rad = jnp.min(state.rad)
    max_rad = jnp.max(state.rad)
    # max_rad = 0.7
    # alpha = max_rad / min_rad

    cell_size = 5.0 * max_rad
    # cell_size = 2.0 * max_rad
    # if alpha < 2.5:
    #     cell_size = 2 * max_rad
    # else:
    #     cell_size = max_rad / 2


    # cutoff_sq = (2.0) ** 2
    cutoff_sq = 2 * cell_size ** 2
    max_neighbors = 100

    # calculate using carlos's code and compare
    system = jd.System.create(
        state_shape=state.shape,
        dt=system.dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="verletspiral",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="sortedneighborlist",
        collider_kw=dict(
            state=state,
            cutoff=jnp.sqrt(cutoff_sq),
            cell_size=cell_size,
            max_neighbors=max_neighbors
        ),
        mat_table=system.mat_table,
        domain_kw=dict(
            box_size=system.domain.box_size,
        ),
    )

    # THIS ALL SEEMS VERY SCREWY AND PROBABLY COULD BE ACCOMPLISHED WITH SYSTEM.COLLIDER.CELL_LIST?
    from dataclasses import replace
    list_cutoff = system.collider.cutoff + system.collider.skin
    inner_system = replace(system, collider=system.collider.cell_list)
    (
        state,
        _,
        sorted_nl_indices,
        _,
    ) = system.collider.cell_list.create_neighbor_list(
        state, inner_system, list_cutoff, system.collider.max_neighbors
    )


    # new_state, new_system = new_system.collider.compute_force(new_state, new_system)
    # print(jnp.sum(new_system.collider.neighbor_list != -1, axis=1))
    # print(new_state.unique_ID)
    # print(jnp.sum(new_state.force, axis=0))

    pos_p_global = state.q.rotate(state.q, state.pos_p)

    def per_particle_force(i, pos_p_i, neighbors):
        def per_neighbor_force(j):
            valid = j != -1
            safe_j = jnp.maximum(j, 0)
            f, t = system.force_model.force(i, safe_j, state, system)
            return f * valid, t * valid

        forces, torques = jax.vmap(per_neighbor_force)(neighbors)

        f_sum = jnp.sum(forces, axis=0)
        # Add contact torque: T_total = Sum(T_ij) + (r_i x F_total)
        t_sum = jnp.sum(torques, axis=0) + jnp.cross(pos_p_i, f_sum)

        return f_sum, t_sum

    # Vmap over particle IDs [0, 1, ..., N]
    iota = jax.lax.iota(int, state.N)
    total_force, total_torque = jax.vmap(per_particle_force)(
        iota, pos_p_global, sorted_nl_indices
    )

    # Aggregate over particles in clumps
    total_torque = jax.ops.segment_sum(total_torque, state.ID, num_segments=state.N)
    total_force = jax.ops.segment_sum(total_force, state.ID, num_segments=state.N)

    # Update state
    state.force += total_force[state.ID]
    state.torque += total_torque[state.ID]


    # exit()

    # print(total_force.shape)
    # print(total_force)

    print(jnp.sum(total_force, axis=0))


    # exit()
    neighbor_list = sorted_nl_indices

    # system = new_system
    # state = new_state

    pid_i = 10

    from matplotlib.patches import Circle
    plt.xlim(0, system.domain.box_size[0])
    plt.ylim(0, system.domain.box_size[1])
    plt.gca().set_aspect('equal')
    plt.gca().add_artist(Circle(jnp.mod(state.pos[pid_i], system.domain.box_size), jnp.sqrt(cutoff_sq), facecolor='k', alpha=0.2))
    for p, r in zip(jnp.mod(state.pos, system.domain.box_size), state.rad):
        plt.gca().add_artist(Circle(p, r))
    plt.gca().add_artist(Circle(jnp.mod(state.pos[pid_i], system.domain.box_size), state.rad[pid_i], facecolor='k'))
    for i in neighbor_list[pid_i]:
        if i == -1:
            continue
        plt.gca().add_artist(Circle(jnp.mod(state.pos[i], system.domain.box_size), state.rad[i], facecolor='r'))

    p = jnp.mod(state.pos, system.domain.box_size)
    f = state.force / jnp.linalg.norm(state.force, axis=-1, keepdims=True)
    plt.quiver(p[:, 0], p[:, 1], f[:, 0], f[:, 1])

    plt.savefig('test.png')