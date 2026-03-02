import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os
from jaxdem.forces.force_manager import ForceManager
from functools import partial
import numpy as np
from tqdm import tqdm

data_root = '/home/mmccraw/dev/data/26-01-01/grant/sphere-fragilitiy/version-3/1'

n_steps = 10_000
save_stride = 1

for path in tqdm(os.listdir(data_root)):
    input_path = os.path.join(data_root, path, 'final')
    state = jd.utils.h5.load(os.path.join(input_path, 'state.h5'))
    system = jd.utils.h5.load(os.path.join(input_path, 'system.h5'))

    save_steps = jnp.arange(save_stride, n_steps + save_stride, save_stride)

    def save_fn(st, sy):
        # --- bonded virial (uses original ordering) ---
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
            contact_virial,
            kinetic_virial,
        )

    state, system, logged = system.trajectory_rollout_at_steps(
        state, system, save_steps=save_steps, save_fn=save_fn,
    )

    contact_virial, kinetic_virial = logged

    np.savez(
        f'sphere-virials-{path}.npz',
        contact_virial=contact_virial,
        kinetic_virial=kinetic_virial,
        area=jnp.prod(system.domain.box_size),
        phi=float(path.split('phi-')[-1]),
    )

    jax.clear_caches()