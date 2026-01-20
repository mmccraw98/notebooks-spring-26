import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
import h5py

from resources import *
from routines import run_nvt_compression
from config import default_config, config2
from file_management import make_data_dir, save_arrs

from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration


if __name__ == "__main__":
    data_root = '/home/mmccraw/dev/data/26-01-01/grant/sphere-fragilitiy/dynamics-2'

    particle_radii = []
    for _ in range(config2.target_temperatures.size):
        particle_radii.append(jd.utils.dispersity.get_polydisperse_radii(config2.N))
    pos, box_size = random_sphere_configuration(particle_radii, config2.phi, config2.dim)

    states, systems = [], []
    for p, rad, bs, dt in zip(pos, particle_radii, box_size, config2.dt):
        state, system = create(p, rad, bs, config2.e_int, dt)
        states.append(state)
        systems.append(system)
    state = jd.State.stack(states)
    system = jd.System.stack(systems)

    key = jax.random.PRNGKey(np.random.randint(0, 1e9))
    state.vel = jax.random.normal(key, state.vel.shape)
    state = scale_temps(state, config2.target_temperatures)

    # run thermalization without compressing the states
    state, system = run_nvt_compression(
        state,
        system,
        0.0,  # do not compress on the first run
        config2.target_temperatures,
        n_steps=config2.n_dynamics_steps // 10,
        nve_block_length=1_000
    )

    # save the initial data
    phi = compute_phi(state, system)
    run_root = os.path.join(data_root, f'phi-{phi[0]:.6f}')
    run_root_paths = make_data_dir(run_root)
    jd.utils.h5.save(state, os.path.join(run_root_paths['init'], 'state.h5'))
    jd.utils.h5.save(system, os.path.join(run_root_paths['init'], 'system.h5'))
    
    # run dynamics
    print('Running dynamics...')
    save_stride = 500
    n_snapshots = default_config.n_dynamics_steps // save_stride
    state, system, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=n_snapshots, stride=save_stride
    )
    print('Done')

    # save the trajectory
    save_arrs([state_traj.pos, state_traj.vel], ['pos', 'vel'], os.path.join(run_root_paths['traj'], 'data.h5'))

    # save the final state
    jd.utils.h5.save(state, os.path.join(run_root_paths['final'], 'state.h5'))
    jd.utils.h5.save(system, os.path.join(run_root_paths['final'], 'system.h5'))
