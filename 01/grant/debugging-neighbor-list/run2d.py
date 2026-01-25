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

    data_root = '/home/mmccraw/dev/data/26-01-01/grant/debugging-neighbor-list/1'
    figures_root = '/home/mmccraw/dev/analysis/spring-26/01/grant/debugging-neighbor-list/figures/energies'

    if os.path.exists(data_root):
        shutil.rmtree(data_root)
    if os.path.exists(figures_root):
        shutil.rmtree(figures_root)
    os.makedirs(data_root)
    os.makedirs(figures_root)

    # for mu in [0.01, 0.1, 1.0][::-1]:
    for mu in [0.5]:
        for skin in [0.001, 0.01, 0.1, 0.5, 1.0]:
            name = f'mu-{mu}-skin-{skin}'
            run_root = os.path.join(data_root, name)
            if not os.path.exists(run_root):
                os.makedirs(run_root)
            else:
                shutil.rmtree(run_root)
                os.makedirs(run_root)

            state, system = create_clumps(0.8, N, mu, 1.0, 10, 1.0, skin)
            state = jd.utils.thermal.set_temperature(state, 1e-3, is_rigid=True, subtract_drift=True, seed=1)

            print('Running dynamics...')
            n_steps = 1_000
            save_stride = 1
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

            import matplotlib.pyplot as plt
            plt.plot(pe, label='pe')
            plt.plot(ke_t, label='ke_t')
            plt.plot(ke_r, label='ke_r')
            plt.plot(te, label='te')
            plt.title(float(np.std(te) / np.mean(te)))
            plt.legend()
            plt.savefig(os.path.join(figures_root, f'{name}.png'), dpi=600)
            plt.close()