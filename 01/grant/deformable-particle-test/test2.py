from bump_utils import create_clumps, create_dps, render, animate
import jaxdem as jd
import jax
import numpy as np
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

state, system = create_dps(0.7, 10, 0.5, 1.0, 10, em=1e0, eb=1e-5, ec=1000.0)

target_temperature = 1e-2
dpids = jnp.unique(state.deformable_ID)
key = jax.random.key(np.random.randint(0, 1e9))
vel = jax.random.normal(key, shape=(dpids.size, state.dim))
vel -= jnp.mean(vel, axis=0)
ke = 0.5 * jnp.sum(vel ** 2)
scale = jnp.sqrt(target_temperature / (ke * 2 / vel.size))
vel *= scale
state.vel = vel[state.deformable_ID]

save_stride = 100
n_snapshots = 10_000 // save_stride
state, system, (state_traj, system_traj) = system.trajectory_rollout(
    state, system, n=n_snapshots, stride=save_stride
)

animate(state_traj, system_traj, 'test.gif', id_name='deformable_ID')

jd.utils.h5.save(state, 'state.h5')
jd.utils.h5.save(system, 'system.h5')
jd.utils.h5.save(state_traj, 'state_traj.h5')
jd.utils.h5.save(system_traj, 'system_traj.h5')
