from bump_utils import create_clumps, create_dps, render, animate
import jaxdem as jd
import jax
import numpy as np
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

state = jd.utils.h5.load('state.h5')
system = jd.utils.h5.load('system.h5')
state_traj = jd.utils.h5.load('state_traj.h5')
system_traj = jd.utils.h5.load('system_traj.h5')


save_stride = 100
n_snapshots = 10_000 // save_stride
state, system, (state_traj, system_traj) = system.trajectory_rollout(
    state, system, n=n_snapshots, stride=save_stride
)

animate(state_traj, system_traj, 'test.gif', id_name='deformable_ID')
