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


n_dynamics_steps = 100_000
save_stride = 100
n_snapshots = n_dynamics_steps // save_stride
state, system, (state_traj, system_traj) = system.trajectory_rollout(
    state, system, n=n_snapshots, stride=save_stride
)


N_dps = int(jnp.max(state.deformable_ID) + 1)
def compute_virial_single_frame(carry, frame):
    """Process one trajectory frame, return both virial components."""
    system = carry  # system is constant across frames (or carried if it changes)
    state_t, system_t = frame

    # --- bonded virial ---
    pos_dp = get_com_pos(state_t, N_dps)[state_t.deformable_ID]
    bonded_force, _ = dp_force(state_t.pos, state_t, system_t)
    pos_rel = state_t.pos - pos_dp
    bonded_virial = jnp.sum(
        pos_rel[:, :, None] * bonded_force[:, None, :], axis=0
    )

    # --- non-bonded (contact) virial ---
    cutoff = jnp.max(state_t.rad) * 3
    max_neighbors = 100
    state_t, system_t, nl, overflow = system_t.collider.create_neighbor_list(
        state_t, system_t, cutoff, max_neighbors
    )
    iota = jax.lax.iota(dtype=int, size=state_t.N)
    pos_p_global = state_t.q.rotate(state_t.q, state_t.pos_p)
    pos = state_t.pos_c + pos_p_global

    def per_particle_force_and_dist(i, pos_pi, neighbors):
        def per_neighbor_force_and_dist(j_id):
            valid = j_id != -1
            safe_j = jnp.maximum(j_id, 0)
            f, _ = system_t.force_model.force(i, safe_j, pos, state_t, system_t)
            rij = system_t.domain.displacement(pos[i], pos[safe_j], system_t)
            return f * valid, rij * valid
        forces, dists = jax.vmap(per_neighbor_force_and_dist)(neighbors)
        return forces, dists

    neigh_force, neigh_dist = jax.vmap(per_particle_force_and_dist)(
        iota, pos_p_global, nl
    )
    contact_virial = 0.5 * jnp.sum(
        neigh_dist[:, :, :, None] * neigh_force[:, :, None, :], axis=(0, 1)
    )

    return carry, (bonded_virial, contact_virial)


# Run the scan over all frames
_, (bonded_virials, contact_virials) = jax.lax.scan(
    compute_virial_single_frame,
    system,                          # carry (constant)
    (state_traj, system_traj),       # xs — one frame per leading-axis slice
)

np.savez(
    'virials.npz',
    bonded_virial=bonded_virials,
    contact_virial=contact_virials,
)
