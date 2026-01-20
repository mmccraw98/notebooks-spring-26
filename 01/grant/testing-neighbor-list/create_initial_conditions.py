import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import numpy as np
from bump_utils import create

import os
data_root = '/home/mmccraw/dev/data/26-01-01/grant/profiling-cell-list/'

if __name__ == "__main__":
    jamming_root = os.path.join(data_root, 'jamming')
    if not os.path.exists(jamming_root):
        os.makedirs(jamming_root)

    for mu_eff in [0.01, 0.1, 1.0]:
        for aspect_ratio in [1.0, 1.5, 2.0]:
            particle_name = f'mu-{mu_eff}-alpha-{aspect_ratio}'
            particle_root = os.path.join(jamming_root, particle_name)
            if not os.path.exists(particle_root):
                os.makedirs(particle_root)
            state, system = create(0.4, 10, mu_eff, aspect_ratio, 20)
            state, system, phi, pe = jd.utils.bisection_jam(state, system, n_minimization_steps=1_000_00, n_jamming_steps=1_000_000, packing_fraction_increment=1e-2)
            jd.utils.h5.save(state, os.path.join(particle_root, 'state.h5'))
            jd.utils.h5.save(system, os.path.join(particle_root, 'system.h5'))