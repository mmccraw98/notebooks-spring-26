import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import numpy as np
from bump_utils import create_for_dynamics, render

import os
data_root = '/home/mmccraw/dev/data/26-01-01/grant/sample-system-data-ellipses/'

if __name__ == "__main__":
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    for mu_eff in [0.01, 0.1, 1.0]:
        for N in 2 * (np.logspace(2, 4, num=8).astype(int) // 2):
            target_temperature = 1e-5
            particle_name = f'mu-{mu_eff}'
            particle_root = os.path.join(data_root, f'N-{N}', particle_name)
            if not os.path.exists(particle_root):
                os.makedirs(particle_root)
            state, system = create_for_dynamics(
                0.8,
                N,
                mu_eff,
                2.0,
                20,
                1e-2,
                1.0
            )
            print('done')
            cids, offsets = jnp.unique(state.ID, return_index=True)
            key = jax.random.key(np.random.randint(0, 1e9))
            vel = jax.random.normal(key, (cids.size, state.dim))
            vel -= jnp.mean(vel, axis=0)
            dof = cids.size * state.dim + cids.size * state.angVel.shape[1]
            ke = jnp.sum(0.5 * state.mass[offsets, None] * vel ** 2)
            temperature = 2 * ke / (dof)
            scale = jnp.sqrt(target_temperature / temperature)
            vel *= scale
            state.vel = vel[state.ID]
            # render(state, system, f'sample-renders/{particle_name}.png')
            jd.utils.h5.save(state, os.path.join(particle_root, 'state.h5'))
            jd.utils.h5.save(system, os.path.join(particle_root, 'system.h5'))
