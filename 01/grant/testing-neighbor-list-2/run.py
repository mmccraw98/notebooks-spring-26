import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import numpy as np
from bump_utils import animate
import os
from dataclasses import replace
# data_root = '/home/mmccraw/dev/data/26-01-01/grant/sample-system-data/'
data_root = '/home/mmccraw/dev/data/26-01-01/grant/sample-system-data-ellipses/'

if __name__ == "__main__":
    for N_root in os.listdir(data_root):
        N_path = os.path.join(data_root, N_root)
        for mu_root in os.listdir(N_path):
            path = os.path.join(N_path, mu_root)
            state = jd.utils.h5.load(os.path.join(path, 'state.h5'))
            system = jd.utils.h5.load(os.path.join(path, 'system.h5'))

            system = replace(
                system,
                # collider=jd.colliders.Collider.create(
                #     "celllist",
                #     state=state,
                # ),
                collider=jd.colliders.Collider.create(
                    "neighborlist",
                    state=state,
                    cutoff=2.0 * jnp.max(state.rad),
                    skin=0.03,
                ),
            )

            n_steps = 1_000
            save_stride = 100
            n_snapshots = n_steps // save_stride

            state, system, (state_traj, system_traj) = system.trajectory_rollout(
                state, system, n=n_snapshots, stride=save_stride
            )

            import time
            start = time.time()
            state, system, (state_traj, system_traj) = system.trajectory_rollout(
                state, system, n=n_snapshots, stride=save_stride
            )
            # state, system, (state_traj, system_traj) = system.trajectory_rollout(
            #     state, system, n=n_snapshots, stride=save_stride
            # )
            # state, system, (state_traj, system_traj) = system.trajectory_rollout(
            #     state, system, n=n_snapshots, stride=save_stride
            # )
            state.pos.block_until_ready()
            print(time.time() - start)

            # with jax.profiler.trace("test-profile"):
            #     state, system, (state_traj, system_traj) = system.trajectory_rollout(
            #         state, system, n=n_snapshots, stride=save_stride
            #     )
            #     state.pos.block_until_ready()

            print(system.collider.n_build_times)

            exit()
