import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
import argparse
import sys

from resources import *
from routines import run_nvt_compression
from config import default_config, config2
from file_management import make_data_dir, save_arrs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    args = parser.parse_args()

    # load the final data from the previous run
    input_path = args.input_path.rstrip('/')
    data_root = os.path.dirname(input_path)
    state = jd.utils.h5.load(os.path.join(input_path, 'final', 'state.h5'))
    system = jd.utils.h5.load(os.path.join(input_path, 'final', 'system.h5'))

    # run thermalization while compressing the states
    state, system = run_nvt_compression(
        state,
        system,
        config2.delta_phi,
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
    n_snapshots = config2.n_dynamics_steps // save_stride
    state, system, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=n_snapshots, stride=save_stride
    )
    print('Done')

    # save the trajectory
    save_arrs([state_traj.pos, state_traj.vel], ['pos', 'vel'], os.path.join(run_root_paths['traj'], 'data.h5'))

    # save the final state
    jd.utils.h5.save(state, os.path.join(run_root_paths['final'], 'state.h5'))
    jd.utils.h5.save(system, os.path.join(run_root_paths['final'], 'system.h5'))

    # run another step if the packing fraction is less than the target
    if phi[0] < config2.phi_target:
        script = os.path.abspath(__file__)
        os.execv(sys.executable, [sys.executable, script, "--input_path", run_root])