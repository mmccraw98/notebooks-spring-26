import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
from file_management import make_data_dir, save_arrs

from bump_utils import create_dps_2d
from jaxdem.utils.dynamicsRoutines import control_nvt_density


if __name__ == "__main__":
    which = '2d-soft'

    data_root = f'/Users/marshallmccraw/Projects/yale/data/s-26/grant/dp-fragility/{which}'
    if not os.path.exists(data_root):
        os.makedirs(data_root)

    if which == '2d-soft':
        from config import config2d_soft as cfg
    elif which == '2d-med':
        from config import config2d_med as cfg
    elif which == '2d-hard':
        from config import config2d_hard as cfg
    else:
        raise ValueError(f'Which {which} is unknown')

    seed = np.random.randint(0, 1e9)

    state, system, dp = create_dps_2d(
        phi=cfg.phi,
        N=cfg.N,
        mu_eff=cfg.mu_eff,
        aspect_ratio=1.0,
        min_nv=cfg.nv,
        mass=cfg.mass,
        eb=cfg.eb,
        el=cfg.el,
        ec=cfg.ec
    )
    # state = jd.utils.thermal.set_temperature(state, cfg.target_temperature, can_rotate=False, subtract_drift=True, seed=seed)
    key = jax.random.PRNGKey(seed)
    dp_vels = jax.random.normal(key, (cfg.N, state.dim))
    dp_vels -= jnp.mean(dp_vels, axis=0, keepdims=True)
    ke = 0.5 * jnp.sum(dp_vels ** 2) * cfg.mass
    temp = ke * 2 / (state.dim * state.N)
    dp_vels *= jnp.sqrt(cfg.target_temperature / temp)
    state.vel = dp_vels[state.deformable_ID]

    # run thermalization without compressing the states
    print('Running NVT...')
    # state, system, (traj_state, traj_system) = jd.utils.control_nvt_density_rollout(
    state, system = jd.utils.control_nvt_density(
        state,
        system,
        n=cfg.n_dynamics_steps // 10,
        # stride=100,
        rescale_every=100,
        temperature_delta=0.0,  # maintain temperature
        packing_fraction_delta=0.0,  # do not compress on the first run
        can_rotate=False,
        subtract_drift=True,
    )
    print('Done')

    # save the initial data
    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
    run_root = os.path.join(data_root, f'phi-{phi:.6f}')
    run_root_paths = make_data_dir(run_root)
    jd.utils.h5.save(state, os.path.join(run_root_paths['init'], 'state.h5'))
    jd.utils.h5.save(system, os.path.join(run_root_paths['init'], 'system.h5'))
    jd.utils.h5.save(dp, os.path.join(run_root_paths['init'], 'dp.h5'))

    # run dynamics
    print('Running dynamics...')
    save_stride = 500
    n_snapshots = cfg.n_dynamics_steps // save_stride
    state, system, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=n_snapshots, stride=save_stride
    )
    print('Done')

    # save the trajectory
    save_arrs([state_traj.pos, state_traj.vel, state_traj.unique_ID], ['pos', 'vel', 'unique_ID'], os.path.join(run_root_paths['traj'], 'data.h5'))

    # save the final state
    jd.utils.h5.save(state, os.path.join(run_root_paths['final'], 'state.h5'))
    jd.utils.h5.save(system, os.path.join(run_root_paths['final'], 'system.h5'))
    jd.utils.h5.save(dp, os.path.join(run_root_paths['final'], 'dp.h5'))
