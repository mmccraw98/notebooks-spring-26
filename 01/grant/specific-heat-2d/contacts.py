import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
from dataclasses import replace

def get_pair_forces_and_ids(state, system, cutoff=None, max_neighbors=None):
    if cutoff is None:
        cutoff = jnp.max(state.rad) * 3.0
    if max_neighbors is None:
        max_neighbors = 100
    state, system, nl, overflow = system.collider.create_neighbor_list(state, system, cutoff, max_neighbors)
    if overflow:
        raise ValueError('Neighbor list overflowed.  Increase max_neighbors.')
    sphere_ids = jax.lax.iota(dtype=int, size=state.N)
    pos_p_global = state.q.rotate(state.q, state.pos_p)
    pos = state.pos_c + pos_p_global
    def per_pair_force(i, pos_pi, neighbors):
        def per_neighbor_force(j_id):
            valid = j_id != -1
            safe_j = jnp.maximum(j_id, 0)
            f, _ = system.force_model.force(i, safe_j, pos, state, system)
            return f * valid
        pair_forces = jax.vmap(per_neighbor_force)(neighbors)
        return pair_forces
    neigh_force = jax.vmap(per_pair_force)(sphere_ids, pos_p_global, nl)
    j_ids = nl.copy()
    n_neighbors = j_ids.shape[1]
    i_ids = jnp.column_stack(([sphere_ids for _ in range(n_neighbors)]))
    j_ids = j_ids.ravel()
    i_ids = i_ids.ravel()
    neigh_force = neigh_force.reshape(-1, state.dim)
    return state, system, jnp.column_stack((i_ids, j_ids)), neigh_force

def get_clump_rattler_ids(state, system, cutoff=None, max_neighbors=None, zc=None):
    state, system, pair_ids, neigh_force = get_pair_forces_and_ids(state, system, cutoff, max_neighbors)
    force_norm = jnp.linalg.norm(neigh_force, axis=-1)
    contact_mask = force_norm > 0
    pair_ids = pair_ids[contact_mask]
    N_clumps = int(jnp.max(state.clump_ID)) + 1
    if zc is None:
        dof = state.angVel.shape[1] + state.dim
        zc = dof + 1
    all_clump_ids = jnp.arange(N_clumps)
    clumps_in_contacts = jnp.unique(state.clump_ID[pair_ids.ravel()])
    rattler_ids = jnp.setdiff1d(all_clump_ids, clumps_in_contacts)
    while True:
        if pair_ids.shape[0] == 0:
            print('No valid particles remain!')
            break
        clump_i = state.clump_ID[pair_ids[:, 0]]
        vertex_contacts = jnp.bincount(clump_i, minlength=N_clumps)
        active_clumps = jnp.unique(clump_i)
        rattler_condition = vertex_contacts[active_clumps] < zc
        new_rattlers = jnp.setdiff1d(active_clumps[rattler_condition], rattler_ids)
        if len(new_rattlers) == 0:
            break
        rattler_ids = jnp.union1d(rattler_ids, new_rattlers)
        clump_j = state.clump_ID[pair_ids[:, 1]]
        keep = ~(jnp.isin(clump_i, rattler_ids) | jnp.isin(clump_j, rattler_ids))
        pair_ids = pair_ids[keep]
    non_rattler_ids = jnp.setdiff1d(all_clump_ids, rattler_ids)
    return state, system, rattler_ids, non_rattler_ids

def get_sphere_rattler_ids(state, system, cutoff=None, max_neighbors=None, zc=None):
    state, system, pair_ids, neigh_force = get_pair_forces_and_ids(state, system, cutoff, max_neighbors)
    force_norm = jnp.linalg.norm(neigh_force, axis=-1)
    contact_mask = force_norm > 0
    pair_ids = pair_ids[contact_mask]
    N = state.N
    if zc is None:
        zc = state.dim + 1
    all_ids = jnp.arange(N)
    in_contacts = jnp.unique(pair_ids.ravel())
    rattler_ids = jnp.setdiff1d(all_ids, in_contacts)
    while True:
        if pair_ids.shape[0] == 0:
            print('No valid particles remain!')
            break
        contacts = jnp.bincount(pair_ids[:, 0], length=N)
        active = jnp.unique(pair_ids[:, 0])
        rattler_condition = contacts[active] < zc
        new_rattlers = jnp.setdiff1d(active[rattler_condition], rattler_ids)
        if len(new_rattlers) == 0:
            break
        rattler_ids = jnp.union1d(rattler_ids, new_rattlers)
        keep = ~(jnp.isin(pair_ids[:, 0], rattler_ids) | jnp.isin(pair_ids[:, 1], rattler_ids))
        pair_ids = pair_ids[keep]
    non_rattler_ids = jnp.setdiff1d(all_ids, rattler_ids)
    return state, system, rattler_ids, non_rattler_ids

def _count_vertex_contacts_per_clump(pair_ids, clump_ID, N_clumps):
    clump_i = clump_ID[pair_ids[:, 0]]
    return jnp.bincount(clump_i, length=N_clumps)

def count_vertex_contacts(state, system, cutoff=None, max_neighbors=None):
    state, system, pair_ids, _ = get_pair_forces_and_ids(state, system, cutoff, max_neighbors)
    N_clumps = int(jnp.max(state.clump_ID)) + 1
    return state, system, _count_vertex_contacts_per_clump(pair_ids, state.clump_ID, N_clumps)

def _count_clump_contacts_per_clump(pair_ids, clump_ID, N_clumps):
    clump_i = clump_ID[pair_ids[:, 0]]
    clump_j = clump_ID[pair_ids[:, 1]]
    adj = jnp.zeros((N_clumps, N_clumps), dtype=bool)
    adj = adj.at[clump_i, clump_j].set(True)
    adj = adj & ~jnp.eye(N_clumps, dtype=bool)
    return jnp.sum(adj, axis=1)

def count_clump_contacts(state, system, cutoff=None, max_neighbors=None):
    state, system, pair_ids, _ = get_pair_forces_and_ids(state, system, cutoff, max_neighbors)
    N_clumps = int(jnp.max(state.clump_ID)) + 1
    return state, system, _count_clump_contacts_per_clump(pair_ids, state.clump_ID, N_clumps)

def remove_rattlers_from_state(state, rattler_clump_ids):
    """Remove all spheres belonging to rattler clumps and rebuilds the state."""
    keep = ~jnp.isin(state.clump_ID, rattler_clump_ids)
    idx = jnp.where(keep)[0]
    new_state = jax.tree.map(lambda x: x[idx], state)
    N_new = idx.shape[0]
    _, new_clump_ID = jnp.unique(new_state.clump_ID, return_inverse=True, size=N_new)
    _, new_deformable_ID = jnp.unique(new_state.deformable_ID, return_inverse=True, size=N_new)
    new_state = replace(
        new_state,
        clump_ID=new_clump_ID,
        deformable_ID=new_deformable_ID,
        unique_ID=jnp.arange(N_new, dtype=int),
    )
    return new_state

# TODO: ADD COLINEARITY CHECK FOR 2D AND COPLANARITY(?) CHECK FOR 3D
