import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os

from resources import *
from routines import run_nvt_compression
from config import default_config
from file_management import make_data_dir, save_arrs
from bump_utils import create_clumps


if __name__ == "__main__":
    data_root = '/home/mmccraw/dev/data/26-01-01/grant/ga-fragilitiy/dynamics'

    state, system = create_clumps(default_config.phi, default_config.N, default_config.mu_eff, default_config.aspect_ratio, default_config.min_nv, default_config.mass)
    state = jd.utils.thermal.set_temperature(state, default_config.target_temperature, is_rigid=True, subtract_drift=True)

    # run thermalization without compressing the states
    state, system = run_nvt_compression(
        state,
        system,
        0.0,  # do not compress on the first run
        default_config.target_temperature,
        n_steps=default_config.n_dynamics_steps // 10,
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
    save_stride = 1000
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
