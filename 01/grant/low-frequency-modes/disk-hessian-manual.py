import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os
import numpy as np
from tqdm import tqdm
import scipy as sp

state = jd.utils.h5.load('data/jammed-disk/state.h5')
system = jd.utils.h5.load('data/jammed-disk/system.h5')

state, system, rattler_ids, non_rattler_ids = jd.utils.contacts.get_sphere_rattler_ids(state, system)
state, system = jd.utils.contacts.remove_rattlers_from_state(state, system, rattler_ids)

e_int = 1.0

r_ij = state.pos[:, None, :] - state.pos[None, ...]
r_ij -= system.domain.box_size * jnp.round(r_ij / system.domain.box_size)
dist = jnp.linalg.norm(r_ij, axis=-1)
rad_ij = state.rad[:, None] + state.rad[None, :]
r_hat = r_ij / dist[..., None]

t_ij = - e_int * (rad_ij - dist) * (dist < rad_ij)
c_ij = e_int * jnp.ones_like(t_ij) * (dist < rad_ij)

hessian = np.zeros((state.N, state.N, state.dim, state.dim))

for i in range(state.N):
    for j in range(state.N):
        if i == j:
            continue
        for a in range(state.dim):
            for b in range(state.dim):
                delta_ab = 1.0 if a == b else 0.0
                rr = r_hat[i, j, a] * r_hat[i, j, b]

                # off-diagonal block
                hessian[i, j, a, b] = (
                    - t_ij[i, j] / dist[i, j] * (delta_ab - rr)
                    - c_ij[i, j] * rr
                )

    # diagonal block: K_{ia,ib} = -sum_{j != i} K_{ia,jb}
    for a in range(state.dim):
        for b in range(state.dim):
            hessian[i, i, a, b] = -np.sum(hessian[i, :, a, b])

H_manual = hessian.transpose(2, 0, 3, 1).reshape(state.N * state.dim, state.N * state.dim)
M = np.diag(np.concatenate([state.mass for _ in range(state.dim)]))
vals_manual, vecs_manual = sp.linalg.eigh(H_manual, M)

state, system, H = jd.utils.contacts.compute_hessian_spheres(state, system, reshape=True)

# state, system, pair_ids, r_ij_unit, pair_dists, pair_t, pair_c = jd.utils.contacts.get_pair_potential_derivatives(state, system)

# valid = (pair_ids[:, 1] != -1) & (pair_ids[:, 0] != pair_ids[:, 1])
# ids = pair_ids[valid]
# nhat = r_ij_unit[valid]
# r = pair_dists[valid]
# t = pair_t[valid]
# c = pair_c[valid]

# hessian = jnp.zeros((state.N, state.N, state.dim, state.dim))
# nn = nhat[:, :, None] * nhat[:, None, :]
# eye = jnp.eye(state.dim)
# tr = (t / r)[:, None, None]
# block = -c[:, None, None] * nn + tr * (nn - eye[None])
# hessian = hessian.at[ids[:, 0], ids[:, 1], :state.dim, :state.dim].add(block) # off-diagonal
# hessian = hessian.at[ids[:, 0], ids[:, 0], :state.dim, :state.dim].add(-block) # diagonal

H = hessian.transpose(2, 0, 3, 1).reshape(state.N * state.dim, state.N * state.dim)
M = np.diag(np.concatenate([state.mass for _ in range(state.dim)]))
vals, vecs = sp.linalg.eigh(H, M)

# new way
state, system, pair_ids, _ = jd.utils.contacts.get_pair_forces_and_ids(
    state,
    system,
)
valid = (pair_ids[:, 1] != -1) & (pair_ids[:, 0] != pair_ids[:, 1])
pair_ids = pair_ids[valid]
n_pairs = pair_ids.shape[0]
def total_energy(pos):
    energies = jax.vmap(lambda k: system.force_model.energy(
        pair_ids[k, 0], pair_ids[k, 1], pos, state, system
    ))(jnp.arange(n_pairs))
    return jnp.sum(energies)
H_new = jax.hessian(total_energy)(state.pos)
H_new = H_new.transpose(0, 1, 2, 3).reshape(state.N * state.dim, state.N * state.dim)
M = np.diag(np.concatenate([state.mass for _ in range(state.dim)]))
vals_new, vecs_new = sp.linalg.eigh(H_new, M)

assert vals_manual.size == vals.size
assert vals_new.size == vals.size
assert jnp.all(jnp.isclose(vals_manual, vals))

print('All tests passed')