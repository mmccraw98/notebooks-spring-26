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
    elif force_model_type == 'wca_shifted':
        mats = [jd.Material.create("lj", epsilon=e_int, density=1.0)]
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
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )
    return state, system

if __name__ == "__main__":
    which = '2d-wca'

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

    n_steps_base = 10_000
    save_stride_base = 100
    dt_fractions = [1, 0.5, 0.1]
    dts = []
    energy_fluctuations = []
    temperatures = []

    for dt_fraction in dt_fractions:
        n_steps = int(n_steps_base / dt_fraction)
        save_stride = int(save_stride_base / dt_fraction)

        # create and batch the states
        states, systems = [], []
        for i in range(cfg.target_temperatures.size):
            seed = np.random.randint(0, 1e9)
            particle_radii = jd.utils.dispersity.get_polydisperse_radii(cfg.N)
            pos, box_size = random_sphere_configuration(particle_radii, cfg.phi, cfg.dim)
            state, system = create(pos, particle_radii, box_size, cfg.e_int, cfg.dt[i] * dt_fraction, cfg.force_model_type)
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
                n=n_steps // 10,
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

        # run dynamics
        print('Running dynamics...')
        n_snapshots = n_steps // save_stride
        state, system, (state_traj, system_traj) = system.trajectory_rollout(
            state, system, n=n_snapshots, stride=save_stride
        )
        print('Done')

        pe = jax.vmap(jax.vmap(jd.utils.thermal.compute_potential_energy))(state_traj, system_traj)
        ke = jax.vmap(jax.vmap(jd.utils.thermal.compute_translational_kinetic_energy))(state_traj)

        temperatures.extend(cfg.target_temperatures)
        dts.extend(cfg.dt)
        energy_fluctuations.extend(np.std(pe + ke, axis=0) / np.mean(pe + ke, axis=0))
        
    import matplotlib.pyplot as plt
    dts = np.array(dts)
    temperatures = np.array(temperatures)
    energy_fluctuations = np.array(energy_fluctuations)
    for temperature in temperatures:
        mask = temperatures == temperature
        plt.scatter(dts[mask], energy_fluctuations[mask], label=temperature)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel(r'$\Delta t$', fontsize=16)
    plt.ylabel(r'$\sigma(E) / \mu(E)$', fontsize=16)
    plt.savefig('energies.png', dpi=600)
    plt.close()