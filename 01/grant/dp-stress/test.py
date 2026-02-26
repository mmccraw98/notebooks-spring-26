import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os
from jaxdem.forces.force_manager import ForceManager
from functools import partial
import numpy as np

@partial(jax.jit, static_argnames=("N_dps",))
def get_com_pos(st, N_dps):
    total_pos = jax.ops.segment_sum(st.pos_c, st.deformable_ID, num_segments=N_dps)
    dp_counts = jax.ops.segment_sum(
        jnp.ones((st.N,), dtype=st.pos_c.dtype),
        st.deformable_ID,
        num_segments=N_dps,
    )
    return total_pos / jnp.maximum(dp_counts[:, None], 1.0)

input_path = 'data'
state = jd.utils.h5.load(os.path.join(input_path, 'state.h5'))
system = jd.utils.h5.load(os.path.join(input_path, 'system.h5'))
dp = jd.utils.h5.load(os.path.join(input_path, 'dp.h5'))
dp_force, dp_energy = dp.create_force_energy_functions(dp)
system.force_manager = ForceManager.create(
    state_shape=state.shape,
    gravity=None,
    force_functions=[(dp_force, dp_energy, False)],
)


# non bonded virial
N_dps = int(jnp.max(state.deformable_ID) + 1)
pos_dp = get_com_pos(state, N_dps)[state.deformable_ID]
bonded_force, _ = dp_force(state.pos, state, system)
pos_rel = state.pos -  pos_dp
bonded_virial = jnp.sum(pos_rel[:, :, None] * bonded_force[:, None, :], axis=0)


# bonded virial
cutoff = jnp.max(state.rad) * 3
max_neighbors = 100
state, system, nl, overflow = system.collider.create_neighbor_list(state, system, cutoff, max_neighbors)
iota = jax.lax.iota(dtype=int, size=state.N)
pos_p_global = state.q.rotate(state.q, state.pos_p)
pos = state.pos_c + pos_p_global
def per_particle_force_and_dist(i, pos_pi, neighbors):
    def per_neighbor_force_and_dist(j_id):
        valid = j_id != -1
        safe_j = jnp.maximum(j_id, 0)
        f, _ = system.force_model.force(i, safe_j, pos, state, system)
        rij = system.domain.displacement(pos[i], pos[safe_j], system)
        return f * valid, rij * valid
    forces, dists = jax.vmap(per_neighbor_force_and_dist)(neighbors)
    return forces, dists
neigh_force, neigh_dist = jax.vmap(per_particle_force_and_dist)(iota, pos_p_global, nl)
contact_virial = 0.5 * jnp.sum(neigh_dist[:, :, :, None] * neigh_force[:, :, None, :], axis=(0, 1))


print(bonded_virial + contact_virial)