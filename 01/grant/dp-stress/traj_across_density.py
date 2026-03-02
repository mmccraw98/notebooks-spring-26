import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os
from jaxdem.forces.force_manager import ForceManager
from functools import partial
import numpy as np
from tqdm import tqdm

@partial(jax.jit, static_argnames=("N_dps",))
def get_com_pos(st, N_dps):
    total_pos = jax.ops.segment_sum(st.pos_c, st.deformable_ID, num_segments=N_dps)
    dp_counts = jax.ops.segment_sum(
        jnp.ones((st.N,), dtype=st.pos_c.dtype),
        st.deformable_ID,
        num_segments=N_dps,
    )
    return total_pos / jnp.maximum(dp_counts[:, None], 1.0)

data_root = '/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-4/med'

n_steps = 10_000
save_stride = 1

for path in tqdm(os.listdir(data_root)):
    input_path = os.path.join(data_root, path, 'final')
    state = jd.utils.h5.load(os.path.join(input_path, 'state.h5'))
    system = jd.utils.h5.load(os.path.join(input_path, 'system.h5'))
    dp = jd.utils.h5.load(os.path.join(input_path, 'dp.h5'))
    # dp.ec = None  # TURN OFF CONTENT ENERGY
    dp_force, dp_energy = dp.create_force_energy_functions(dp)
    system.force_manager = ForceManager.create(
        state_shape=state.shape,
        gravity=None,
        force_functions=[(dp_force, dp_energy, False)],
    )

    N_dps = int(jnp.max(state.deformable_ID) + 1)

    save_steps = jnp.arange(save_stride, n_steps + save_stride, save_stride)

    def save_fn(st, sy):
        # --- bonded virial (uses original ordering) ---
        pos_dp = get_com_pos(st, N_dps)[st.deformable_ID]
        bonded_force, _ = dp_force(st.pos, st, sy)
        pos_rel = st.pos - pos_dp
        bonded_virial = jnp.sum(
            pos_rel[:, :, None] * bonded_force[:, None, :], axis=0
        )

        # --- non-bonded (contact) virial ---
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
            bonded_virial,
            contact_virial,
            kinetic_virial,
        )

    state, system, logged = system.trajectory_rollout_at_steps(
        state, system, save_steps=save_steps, save_fn=save_fn,
    )

    bonded_virial, contact_virial, kinetic_virial = logged

    np.savez(
        f'virials-{path}.npz',
        bonded_virial=bonded_virial,
        contact_virial=contact_virial,
        kinetic_virial=kinetic_virial,
        area=jnp.prod(system.domain.box_size),
        phi=float(path.split('phi-')[-1]),
    )

    jax.clear_caches()