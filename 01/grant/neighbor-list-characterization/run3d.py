import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
from bump_utils import create_clumps_3d, animate
import time
import json
import shutil


if __name__ == "__main__":
    
    max_neighbors = 500

    # N = 10  # this gives trouble with rad=0.499
    N = 100

    data_root = '/home/mmccraw/dev/data/26-01-01/grant/neighbor-list-characterization-3d/dynamics-1'

    if os.path.exists(data_root):
        shutil.rmtree(data_root)
    os.makedirs(data_root)

    for asperity_rad in [0.499, 0.4, 0.2]:
        for skin in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
            run_root = os.path.join(data_root, f'rad-{asperity_rad}-skin-{skin}')
            if not os.path.exists(run_root):
                os.makedirs(run_root)
            state, system = create_clumps_3d(0.5, N, asperity_rad, 1.0, 20, 1.0, skin, max_neighbors)
            state = jd.utils.thermal.set_temperature(state, 1e-4, is_rigid=True, subtract_drift=True)
            print(jnp.floor(system.domain.box_size / system.collider.cell_list.cell_size).astype(int))

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
