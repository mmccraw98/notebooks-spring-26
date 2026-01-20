import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import numpy as np
from config import default_config
from bump_utils import create, render

import os
data_root = '/home/mmccraw/dev/data/26-01-01/grant/specific-heat/'

if __name__ == "__main__":
    jamming_root = os.path.join(data_root, 'jamming')
    if not os.path.exists(jamming_root):
        os.makedirs(jamming_root)

    for mu_eff in [0.01, 0.1, 1.0][::-1]:
        for aspect_ratio in [1.0, 1.5, 2.0][::-1]:


            particle_name = f'mu-{mu_eff}-alpha-{aspect_ratio}'
            particle_root = os.path.join(jamming_root, particle_name)
            states, systems = [], []
            for i in range(default_config.n_duplicates):
                state, system = create(default_config.phi, default_config.N, mu_eff, aspect_ratio, default_config.min_nv, default_config.dt, default_config.e_int)
                states.append(state)
                systems.append(system)
                break
            # jd.utils.h5.save(state, 'example-state.h5')
            # jd.utils.h5.save(system, 'example-system.h5')

            # state = jd.State.stack(states)
            # system = jd.System.stack(systems)
            # state, system, final_pf, final_pe = jax.vmap(
            #     lambda st, sys: jd.utils.jamming.bisection_jam(st, sys)
            # )(state, system)

            state, system, phi, pe = jd.utils.bisection_jam(state, system, n_minimization_steps=100_000, n_jamming_steps=1_000, packing_fraction_increment=1e-2)
            render(state, system, f"figures/particle-shapes/{particle_name}.png")
            exit()



            # exit()

            # if not os.path.exists(particle_root):
            #     os.makedirs(particle_root)
            # else:
            #     continue

            # state, system, phi, pe = jd.utils.bisection_jam(state, system, n_minimization_steps=1_000_00, n_jamming_steps=1_000_000, packing_fraction_increment=1e-2)

            # jd.utils.h5.save(state, os.path.join(particle_root, 'state.h5'))
            # jd.utils.h5.save(system, os.path.join(particle_root, 'system.h5'))