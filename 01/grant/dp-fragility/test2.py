from bump_utils import create_clumps, create_dps, render, animate
import jaxdem as jd
import jax
import numpy as np
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def sum_by_id(arr, ids):
    N_ids = jnp.unique(ids).size
    return jax.ops.segment_sum(arr, ids, num_segments=N_ids)

state, system = create_dps(0.7, 100, 0.1, 1.0, 20, em=1e0, eb=1e-1, ec=1e3, mass=1e1)

target_temperature = 1e-2
dpids = jnp.unique(state.deformable_ID)
key = jax.random.key(np.random.randint(0, 1e9))
vel_dp = jax.random.normal(key, shape=(dpids.size, state.dim))
vel_dp -= jnp.mean(vel_dp, axis=0)
mass_dp = sum_by_id(state.mass, state.deformable_ID)
ke = 0.5 * jnp.sum(mass_dp[:, None] * vel_dp ** 2)
scale = jnp.sqrt(target_temperature / (ke * 2 / vel_dp.size))
vel_dp *= scale
state.vel = vel_dp[state.deformable_ID]

save_stride = 100
n_snapshots = 10_000 // save_stride
state, system, (state_traj, system_traj) = system.trajectory_rollout(
    state, system, n=n_snapshots, stride=save_stride
)

jd.utils.h5.save(state, 'state.h5')
jd.utils.h5.save(system, 'system.h5')
jd.utils.h5.save(state_traj, 'state_traj.h5')
jd.utils.h5.save(system_traj, 'system_traj.h5')

animate(state_traj, system_traj, 'test.gif', id_name='deformable_ID')