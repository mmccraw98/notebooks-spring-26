import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

with jd.CheckpointLoader(directory='/home/mmccraw/dev/data/26-01-01/grant/specific-heat/mu-1.0-alpha-1.0/0/0/jamming') as loader:
    state, system = loader.load()

state, system, rattler_ids, non_rattler_ids = jd.utils.contacts.get_clump_rattler_ids(state, system)
state, system = jd.utils.contacts.remove_rattlers_from_state(state, system, rattler_ids)
# state, system = system.collider.compute_force(state, system)

_, offset = np.unique(state.clump_id, return_index=True)
clump_mass = np.asarray(state.mass)[offset]
clump_inertia = np.asarray(state.inertia)[offset]

state, system, pair_ids, r_ij_unit, pair_dists, pair_t, pair_c = jd.utils.contacts.get_pair_potential_derivatives(state, system)

valid = (pair_ids[:, 1] != -1) & (pair_ids[:, 0] != pair_ids[:, 1])
ids = pair_ids[valid]
nhat = np.asarray(r_ij_unit[valid])
r = np.asarray(pair_dists[valid])
t = np.asarray(pair_t[valid])
c = np.asarray(pair_c[valid])

clump_id = np.asarray(state.clump_id)
pos = np.asarray(state.pos)
pos_c = np.asarray(state.pos_c)

N_c = int(jnp.max(state.clump_id)) + 1

hessian = np.zeros((N_c, N_c, 3, 3))

for k in range(len(ids)):
    mu, nu = ids[k, 0], ids[k, 1]
    I, J = clump_id[mu], clump_id[nu]
    if I == J:
        continue

    rhat = nhat[k]
    t_k, c_k, r_k = t[k], c[k], r[k]
    tr_k = t_k / r_k

    lev_mu = pos[mu] - pos_c[mu]
    lev_nu = pos[nu] - pos_c[nu]

    E_mu = np.array([[1, 0], [0, 1], [-lev_mu[1], lev_mu[0]]])
    E_nu = np.array([[1, 0], [0, 1], [-lev_nu[1], lev_nu[0]]])

    p_mu = E_mu @ rhat
    p_nu = E_nu @ rhat

    for a in range(3):
        for b in range(3):
            ee_cross = np.dot(E_mu[a], E_nu[b])
            pp_cross = p_mu[a] * p_nu[b]
            hessian[I, J, a, b] += -tr_k * ee_cross - (c_k - tr_k) * pp_cross

            ee_same = np.dot(E_mu[a], E_mu[b])
            pp_same = p_mu[a] * p_mu[b]
            hessian[I, I, a, b] += tr_k * ee_same + (c_k - tr_k) * pp_same

    hessian[I, I, 2, 2] += -t_k * np.dot(rhat, lev_mu)

df = state.dim + state.ang_vel.shape[1]
H_manual = hessian.transpose(2, 0, 3, 1).reshape(N_c * df, N_c * df)
M = np.diag(np.concatenate([clump_mass for _ in range(state.dim)] + [clump_inertia.ravel()]))
vals_manual, vecs_manual = sp.linalg.eigh(H_manual, M)





# clump_i = clump_id[ids[:, 0]]
# clump_j = clump_id[ids[:, 1]]
# inter_body = clump_i != clump_j
# ids, nhat, r, t, c = ids[inter_body], nhat[inter_body], r[inter_body], t[inter_body], c[inter_body]
# clump_i, clump_j = clump_i[inter_body], clump_j[inter_body]

# lev_mu = pos[ids[:, 0]] - pos_c[ids[:, 0]]
# lev_nu = pos[ids[:, 1]] - pos_c[ids[:, 1]]

# n = len(ids)
# E_mu = np.zeros((n, 3, 2))
# E_mu[:, 0, 0] = 1.0;  E_mu[:, 1, 1] = 1.0
# E_mu[:, 2, 0] = -lev_mu[:, 1];  E_mu[:, 2, 1] = lev_mu[:, 0]

# E_nu = np.zeros((n, 3, 2))
# E_nu[:, 0, 0] = 1.0;  E_nu[:, 1, 1] = 1.0
# E_nu[:, 2, 0] = -lev_nu[:, 1];  E_nu[:, 2, 1] = lev_nu[:, 0]

# p_mu = np.einsum('kac,kc->ka', E_mu, nhat)
# p_nu = np.einsum('kac,kc->ka', E_nu, nhat)
# tr = t / r

# EE_cross = np.einsum('kac,kbc->kab', E_mu, E_nu)
# pp_cross = p_mu[:, :, None] * p_nu[:, None, :]
# off_block = -tr[:, None, None] * EE_cross - (c - tr)[:, None, None] * pp_cross

# EE_same = np.einsum('kac,kbc->kab', E_mu, E_mu)
# pp_same = p_mu[:, :, None] * p_mu[:, None, :]
# diag_block = tr[:, None, None] * EE_same + (c - tr)[:, None, None] * pp_same
# diag_block[:, 2, 2] += -t * np.einsum('kc,kc->k', nhat, lev_mu)

# hessian = np.zeros((N_c, N_c, 3, 3))
# np.add.at(hessian, (clump_i, clump_j), off_block)
# np.add.at(hessian, (clump_i, clump_i), diag_block)

state, system, H = jd.utils.contacts.compute_hessian_clumps_2d(state, system, reshape=True)


df = state.dim + state.ang_vel.shape[1]
H = hessian.transpose(2, 0, 3, 1).reshape(N_c * df, N_c * df)
M = np.diag(np.concatenate([clump_mass for _ in range(state.dim)] + [clump_inertia.ravel()]))
vals, vecs = sp.linalg.eigh(H, M)




# new way
# pos_c: N_c
# q: N_c
# pair_ids: N_pairs for N_v spheres
def compute_potential(pos_c, q):
    # probably have to pass q as an array, index into it, then cast it as a quaternion object first, but anyways:
    pos = pos_c[state.clump_id] + q[state.clump_id].rotate(q[state.clump_id], pos_p)
    # ...



assert vals_manual.size == vals.size
assert jnp.all(jnp.isclose(vals_manual, vals))

print('All tests passed')