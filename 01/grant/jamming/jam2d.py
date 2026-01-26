import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
from bump_utils import create_clumps


if __name__ == "__main__":
    
    N = 1000
    nv = 20
    phi_init = 0.6

    data_root = '/home/mmccraw/dev/data/26-01-01/grant/jamming/2d'

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    for mu in [0.01, 0.1, 1.0][::-1]:
        run_root = os.path.join(data_root, f'mu-{mu}-nv-{nv}')
        if not os.path.exists(run_root):
            os.makedirs(run_root)
        state, system = create_clumps(phi_init, N, mu, 1.0, nv, 1.0)

        state, system, final_pf, final_pe = jd.utils.jamming.bisection_jam(state, system)

        jd.utils.h5.save(state, os.path.join(run_root, 'state.h5'))
        jd.utils.h5.save(system, os.path.join(run_root, 'system.h5'))