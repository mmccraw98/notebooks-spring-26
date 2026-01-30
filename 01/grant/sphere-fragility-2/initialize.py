import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
from file_management import make_data_dir, save_arrs

from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration
from jaxdem.utils.dynamicsRoutines import control_nvt_density

def create(pos, rad, box_size, e_int, dt, force_model_type):
    state = jd.State.create(
        pos=pos,
        rad=rad,
        mass=jnp.ones(pos.shape[0])
    )
    if force_model_type == 'spring':
        mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    elif force_model_type == 'wca':
        mats = [jd.Material.create("lj", epsilon=1.0, density=1.0)]
    else:
        raise ValueError(f'force_model_type {force_model_type} unknown')
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type=force_model_type,
        collider_type="naive",
        # collider_type="neighborlist",
        # collider_kw=dict(
        #     state=state,
        #     cutoff=jnp.max(rad)
        # ),
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )
    return state, system

if __name__ == "__main__":
    which = '2d-wca'

    data_root = f'/home/mmccraw/dev/data/26-01-01/grant/sphere-fragilitiy/version-2/{which}'
    if not os.path.exists(data_root):
        os.makedirs(data_root)

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

    # create and batch the states
    states, systems = [], []
    for i in range(cfg.target_temperatures.size):
        seed = np.random.randint(0, 1e9)
        particle_radii = jd.utils.dispersity.get_polydisperse_radii(cfg.N)
        pos, box_size = random_sphere_configuration(particle_radii, cfg.phi, cfg.dim)
        state, system = create(pos, particle_radii, box_size, cfg.e_int, cfg.dt[i], cfg.force_model_type)
        state = jd.utils.thermal.set_temperature(state, cfg.target_temperatures[i], can_rotate=False, subtract_drift=True, seed=seed)
        states.append(state)
        systems.append(system)
    state = jd.State.stack(states)
    system = jd.System.stack(systems)

    # run thermalization without compressing the states
    print('Running NVT...')
    control = jax.vmap(
        lambda st, sys: jd.utils.control_nvt_density(
            st, sys,
            n=cfg.n_dynamics_steps // 10,
            rescale_every=100,
            temperature_delta=0.0,  # maintain temperature
            packing_fraction_delta=0.0,  # do not compress on the first run
            can_rotate=False,
            subtract_drift=True,
        ),
        in_axes=(0, 0),
    )
    state, system = control(state, system)
    print('Done')

    # save the initial data
    phi = jax.vmap(jd.utils.packingUtils.compute_packing_fraction)(state, system)
    run_root = os.path.join(data_root, f'phi-{phi[0]:.6f}')
    run_root_paths = make_data_dir(run_root)
    jd.utils.h5.save(state, os.path.join(run_root_paths['init'], 'state.h5'))
    jd.utils.h5.save(system, os.path.join(run_root_paths['init'], 'system.h5'))

    # run dynamics
    print('Running dynamics...')
    save_stride = 500
    n_snapshots = cfg.n_dynamics_steps // save_stride
    state, system, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=n_snapshots, stride=save_stride
    )
    print('Done')

    # save the trajectory
    save_arrs([state_traj.pos, state_traj.vel], ['pos', 'vel'], os.path.join(run_root_paths['traj'], 'data.h5'))

    # save the final state
    jd.utils.h5.save(state, os.path.join(run_root_paths['final'], 'state.h5'))
    jd.utils.h5.save(system, os.path.join(run_root_paths['final'], 'system.h5'))
