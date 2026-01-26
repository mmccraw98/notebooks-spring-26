import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
from bump_utils import create_clumps
import time
import json
import shutil

if __name__ == "__main__":
    
    N = 100
    max_neighbors = 500

    data_root = '/home/mmccraw/dev/data/26-01-01/grant/neighbor-list-characterization/dynamics-20'

    if os.path.exists(data_root):
        shutil.rmtree(data_root)
    os.makedirs(data_root)


    for mu in [0.01, 0.1, 1.0]:
        for skin in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            run_root = os.path.join(data_root, f'mu-{mu}-skin-{skin}')
            if not os.path.exists(run_root):
                os.makedirs(run_root)
            else:
                shutil.rmtree(run_root)
                os.makedirs(run_root)

            state, system = create_clumps(0.8, N, mu, 1.0, 10, 1.0, skin, max_neighbors)
            state = jd.utils.thermal.set_temperature(state, 1e-4, is_rigid=True, subtract_drift=True, seed=1)

            print('Running dynamics...')
            n_steps = 10_000
            save_stride = 100
            n_snapshots = int(n_steps) // int(save_stride)
            st, sy, (state_traj, system_traj) = system.trajectory_rollout(
                state, system, n=n_snapshots, stride=int(save_stride)
            )
            start = time.time()
            st, sy, (state_traj, system_traj) = system.trajectory_rollout(
                state, system, n=n_snapshots, stride=int(save_stride)
            )
            run_time = time.time() - start
            pe = jax.vmap(jd.utils.thermal.compute_potential_energy)(state_traj, system_traj)
            ke_t = jax.vmap(jd.utils.thermal.compute_translational_kinetic_energy)(state_traj)
            ke_r = jax.vmap(jd.utils.thermal.compute_rotational_kinetic_energy)(state_traj)
            te = pe + ke_r + ke_t

            print('Done')

            # save the final state
            np.savez(os.path.join(run_root, 'energies.npz'), pe=pe, ke_t=ke_t, ke_r=ke_r)
            jd.utils.h5.save(state_traj, os.path.join(run_root, 'state.h5'))
            jd.utils.h5.save(system_traj, os.path.join(run_root, 'system.h5'))
            with open(os.path.join(run_root, 'stats.json'), 'w') as f:
                json.dump({
                    'n_steps': n_steps,
                    'run_time': run_time,
                    'delta_te': float(np.std(te) / np.mean(te))
                }, f)
