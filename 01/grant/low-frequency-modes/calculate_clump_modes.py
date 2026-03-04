import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import numpy as np
import os

root = '/home/mmccraw/dev/data/26-01-01/grant/specific-heat'
for which in os.listdir(root):
    path = os.path.join(root, which, '0', '4', 'jamming')
    if not os.path.exists(path):
        continue
    with jd.CheckpointLoader(directory=path) as loader:
        state, system = loader.load()

    state, system, rattler_ids, non_rattler_ids = jd.utils.contacts.get_clump_rattler_ids(state, system)
    state, system = jd.utils.contacts.remove_rattlers_from_state(state, system, rattler_ids)

    N_c = int(jnp.max(state.clump_id)) + 1

    _, offset = np.unique(state.clump_id, return_index=True)
    clump_mass = np.asarray(state.mass)[offset]
    clump_inertia = np.asarray(state.inertia)[offset]
    M = jnp.diag(jnp.concatenate([clump_mass for _ in range(state.dim)] + [clump_inertia.ravel()]))

    state, system, H = jd.utils.contacts.compute_hessian_clumps_2d(state, system, reshape=True)

    np.savez(f'modes/{which}.npz', H=H, M=M, N_c=N_c)

    jax.clear_caches()