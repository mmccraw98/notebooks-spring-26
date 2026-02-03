import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    args = parser.parse_args()

    # load the final data from the previous run
    input_path = args.input_path.rstrip('/')
    data_root = os.path.dirname(input_path)
    state = jd.utils.h5.load(os.path.join(input_path, 'final', 'state.h5'))
    system = jd.utils.h5.load(os.path.join(input_path, 'final', 'system.h5'))
    _, which = os.path.split(data_root)

    if which == '2d':
        from config import config2d as cfg
    elif which == '2d-2':
        from config import config2d_2 as cfg
    elif which == '2d-wca':
        from config import config2d_wca as cfg
    elif which == '3d':
        from config import config3d as cfg
    else:
        raise ValueError(f'Which {which} is unknown')

    # run thermalization while compressing the states
    print('Running NVT...')
    control = jax.vmap(
        lambda st, sys: jd.utils.control_nvt_density(
            st, sys,
            n=cfg.n_dynamics_steps // 100,
            rescale_every=100,
            temperature_delta=0.0,  # maintain temperature
            packing_fraction_delta=cfg.delta_phi,  # compress
            can_rotate=False,
            subtract_drift=True,
        ),
        in_axes=(0, 0),
    )
    state, system = control(state, system)
    print('Done')


    # run dynamics
    print('Running dynamics...')
    save_stride = cfg.save_stride
    n_snapshots = (cfg.n_dynamics_steps // 10) // save_stride
    state, system, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=n_snapshots, stride=save_stride
    )
    print('Done')

    jd.utils.h5.save(state, 'state.h5')
    jd.utils.h5.save(system, 'system.h5')
    jd.utils.h5.save(state_traj, 'state_traj.h5')
    jd.utils.h5.save(system_traj, 'system_traj.h5')