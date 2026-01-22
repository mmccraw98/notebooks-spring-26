from bump_utils import create_clumps, create_dps, render, animate, create_disks
import jaxdem as jd
import jax
import numpy as np
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

phi = 0.7
N = 100
mass = 1.0
target_temperature = 1e-3

state, system = create_disks(phi, N, mass)
state = jd.utils.thermal.set_temperature(state, target_temperature, is_rigid=False, subtract_drift=False)

save_stride = 100
n_snapshots = 10_000 // save_stride
state, system, (state_traj, system_traj) = system.trajectory_rollout(
    state, system, n=n_snapshots, stride=save_stride
)

ke = jnp.sum((0.5 * state_traj.mass * jnp.vecdot(state_traj.vel, state_traj.vel)), axis=-1)
pe = jnp.sum(
    jax.vmap(
        lambda st, sys:
        sys.collider.compute_potential_energy(st, sys))(state_traj, system_traj),
    axis=-1
)

plt.plot(pe, label='Potential Energy')
plt.plot(ke, label='Kinetic Energy')
plt.plot(pe + ke, label='Total Energy')
plt.legend()
plt.savefig('energies_disk.png')
plt.close()

jd.utils.h5.save(state, 'disk-data/state.h5')
jd.utils.h5.save(system, 'disk-data/system.h5')
jd.utils.h5.save(state_traj, 'disk-data/state_traj.h5')
jd.utils.h5.save(system_traj, 'disk-data/system_traj.h5')