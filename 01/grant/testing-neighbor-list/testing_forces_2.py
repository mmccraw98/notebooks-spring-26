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

    state, system = system.collider.compute_force(state, system)

    _, offsets = jnp.unique(state.ID, return_index=True)

    print(jnp.sum(state.force[offsets], axis=0))
    neighbor_list = system.collider.neighbor_list


    # system = new_system
    # state = new_state

    pid_i = 20

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