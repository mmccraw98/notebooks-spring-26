import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

def get_eimask(_vals, eps=1e-12):
    vals = np.abs(_vals.copy())
    order = np.argsort(vals)
    vals = vals[order]
    min_val = max(vals[np.argmax(vals[1:] / vals[:-1])], vals[-1] * eps)
    return vals > min_val

for i in range(4):
    with jd.CheckpointLoader(directory=f'/home/mmccraw/dev/data/26-01-01/grant/specific-heat/mu-1.0-alpha-1.0/0/{i}/jamming') as loader:
        state, system = loader.load()

    state, system, rattler_ids, non_rattler_ids = jd.utils.contacts.get_clump_rattler_ids(state, system)
    state, system = jd.utils.contacts.remove_rattlers_from_state(state, system, rattler_ids)

    N_c = int(jnp.max(state.clump_id)) + 1

    _, offset = np.unique(state.clump_id, return_index=True)
    clump_mass = np.asarray(state.mass)[offset]
    clump_inertia = np.asarray(state.inertia)[offset]
    M = jnp.diag(jnp.concatenate([clump_mass for _ in range(state.dim)] + [clump_inertia.ravel()]))

    state, system, H = jd.utils.contacts.compute_hessian_clumps_2d(state, system, reshape=True)

    # vals, vecs = sp.linalg.eigh(H, M)
    vals, vecs = sp.linalg.eigh(H)  # SKIPPING MASS ORTHONORMALIZATION

    np.savez('hessian.npz', H=H, M=M)

    modes = vecs.T.reshape(-1, N_c, 3)

    # verify the unit norm
    # assert jnp.allclose(np.sum(modes ** 2, axis=(-1, -2)), 1.0)
    print(np.sum(modes ** 2, axis=(-1, -2)))

    # calculate the translational and rotational content
    trans = jnp.sum(modes[..., :-1] ** 2, axis=(-1, -2))
    rot = 1 - trans

    mask = get_eimask(vals)
    omega = jnp.sqrt(vals[mask])
    trans = trans[mask]
    rot = rot[mask]

    plt.plot(omega, rot / trans)
plt.yscale('log')
plt.savefig('RT.png')