import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
from bump_utils import create_clumps
import time
import json

if __name__ == "__main__":
    data_root = '/home/mmccraw/dev/data/26-01-01/grant/neighbor-list-characterization/dynamics'
    for mu in [0.01, 0.1, 1.0]:
        for skin in [0.01, 0.05, 0.1, 0.5, 1.0]:
            run_root = os.path.join(data_root, f'mu-{mu}-skin-{skin}')
            if not os.path.exists(run_root):
                os.makedirs(run_root)
            state, system = create_clumps(0.8, 100, mu, 1.0, 20, 1.0, skin)
            state = jd.utils.thermal.set_temperature(state, 1e-4, is_rigid=True, subtract_drift=True)

            # run dynamics
            print('Running dynamics...')
            n_steps = 10_000
            state, system = system.step(state, system, n=n_steps)
            start = time.time()
            state, system = system.step(state, system, n=n_steps)
            run_time = time.time() - start

            print('Done')

            # save the final state
            jd.utils.h5.save(state, os.path.join(run_root, 'state.h5'))
            jd.utils.h5.save(system, os.path.join(run_root, 'system.h5'))
            with open(os.path.join(run_root, 'stats.json'), 'w') as f:
                json.dump({
                    'n_steps': n_steps,
                    'run_time': run_time
                }, f)
