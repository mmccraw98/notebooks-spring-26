import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os

import numpy as np
from tqdm import tqdm


root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-5/compression-disks-1/'

for phi_dir in tqdm(os.listdir(root)):
    n_steps = 1_000_000
    save_stride = 1

    with jd.CheckpointLoader(directory=os.path.join(root, phi_dir, 'final')) as loader:
        state, system = loader.load()
    
    def save_fn(st, sy):
        cutoff = jnp.max(st.rad) * 3
        max_neighbors = 100
        st_nl, sy_nl, nl, overflow = sy.collider.create_neighbor_list(
            st, sy, cutoff, max_neighbors
        )
        iota = jax.lax.iota(dtype=jnp.int32, size=st_nl.N)
        pos = st_nl.pos

        def per_particle_force_and_dist(i, neighbors):
            def per_neighbor_force_and_dist(j_id):
                valid = j_id != -1
                safe_j = jnp.maximum(j_id, 0)
                f, _ = sy_nl.force_model.force(i, safe_j, pos, st_nl, sy_nl)
                rij = sy_nl.domain.displacement(pos[i], pos[safe_j], sy_nl)
                return f * valid, rij * valid
            forces, dists = jax.vmap(per_neighbor_force_and_dist)(neighbors)
            return forces, dists

        neigh_force, neigh_dist = jax.vmap(per_particle_force_and_dist)(
            iota, nl
        )
        contact_virial = 0.5 * jnp.sum(
            neigh_dist[:, :, :, None] * neigh_force[:, :, None, :], axis=(0, 1)
        )
        kinetic_virial = jnp.sum(
            st_nl.mass[:, None, None] * st_nl.vel[:, :, None] * st_nl.vel[:, None, :], axis=0
        )
        return (
            sy.step_count,
            contact_virial,
            kinetic_virial,
        )

    # save_steps = jnp.arange(save_stride, n_steps + save_stride, save_stride)

    save_steps = jnp.asarray(jd.utils.make_save_steps_pseudolog(
        num_steps=n_steps,
        reset_save_decade=100,
        min_save_decade=save_stride,
        decade=10,
        include_step0=True,
    ))

    strides = jnp.diff(save_steps, prepend=0)
    state, system, logged = system.trajectory_rollout(
        state, system, strides=strides, save_fn=save_fn,
    )

    step_ids, contact_virial, kinetic_virial = logged

    os.makedirs(f'disk-virials', exist_ok=True)
    np.savez(
        f'disk-virials/{phi_dir}.npz',
        step_ids=step_ids,
        contact_virial=contact_virial,
        kinetic_virial=kinetic_virial,
        area=jnp.prod(system.domain.box_size),
        phi=float(phi_dir.split('phi-')[-1]),
    )

    jax.clear_caches()