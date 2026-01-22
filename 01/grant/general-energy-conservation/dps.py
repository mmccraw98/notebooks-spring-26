
import jaxdem as jd
import jax
import numpy as np
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

EXPECTED_EXPONENT = 2.0

import math

from jaxdem.utils.geometricAsperityCreation import generate_ga_deformable_state

def calc_trans_ke(state, n_clumps):
    _, offsets = jnp.unique(state.clump_ID, return_index=True, size=n_clumps)
    return 0.5 * jnp.sum(
        ((1 - state.fixed[offsets]) * state.mass[offsets])[:, None] * state.vel[offsets] ** 2
    )

def get_system(state, e_int, dt, domain_type = 'free', domain_kw = None, collider_type = 'naive', collider_kw = None, force_manager_kw = None, rotation_integrator_type = "", linear_integrator_type = "verlet"):
    collider_kw = dict() if collider_kw is None else collider_kw
    force_manager_kw = dict() if force_manager_kw is None else force_manager_kw
    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="verlet",
        rotation_integrator_type=rotation_integrator_type,
        domain_type=domain_type,
        force_model_type="spring",
        collider_type=collider_type,
        collider_kw=collider_kw,
        mat_table=mat_table,
        domain_kw=domain_kw,
        force_manager_kw=force_manager_kw,
    )
    return system

def measure_energy_conservation_dps(phi, N, mass, target_temperature, dim, e_int, e_m, e_b, e_c, Nv, asperity_radius, collider_type):
    save_stride_max = 1_000
    n_steps_max = 100_000
    dts = jnp.array([1e-3, 3e-3, 1e-2])
    n_steps = (n_steps_max * dts.min() / dts).astype(dtype=type(n_steps_max))
    save_strides = (save_stride_max * dts.min() / dts).astype(dtype=type(n_steps_max))

    particle_radii = jd.utils.dispersity.get_polydisperse_radii(N, count_ratios=[1.0], size_ratios=[1.0])
    vertex_counts = jnp.ones_like(particle_radii) * Nv
    base_state, dp, box_size = generate_ga_deformable_state(particle_radii, vertex_counts, phi, dim, asperity_radius, add_core=False, em=e_m, eb=e_b, ec=e_c, mass=mass)

    fluctuation = []
    for dt, n_step, save_stride in zip(dts, n_steps, save_strides):
        n_step = int(n_step); save_stride = int(save_stride); dt = float(dt)
        state = jax.tree.map(lambda x: x, base_state)
        state = jd.utils.thermal.set_temperature(state, target_temperature, is_rigid=True, subtract_drift=True)

        if collider_type == "naive":
            collider_kw = dict()
        elif collider_type == "neighborlist":
            collider_kw = dict(
                state=state,
                cutoff=2.0 * jnp.max(state.rad),
                skin=0.05,
            )
        elif collider_type == "celllist":
            collider_kw = dict(
                state=state
            )
        elif collider_type == "staticcelllist":
            collider_kw = dict(
                state=state
            )

        system = get_system(
            state=state,
            e_int=e_int,
            dt=dt,
            domain_type='periodic',
            domain_kw=dict(box_size=box_size),
            collider_type=collider_type,
            collider_kw=collider_kw,
            force_manager_kw=dict(
                force_functions=(dp.create_force_function(dp),),
            )
        )

        n_snapshots = n_step // save_stride
        state, system, (state_traj, system_traj) = system.trajectory_rollout(
            state, system, n=n_snapshots, stride=save_stride
        )

        ke_t = jax.vmap(lambda st: calc_trans_ke(st, n_clumps=particle_radii.size))(state_traj)
        pe = jnp.sum(
            jax.vmap(
                lambda st, sys:
                sys.collider.compute_potential_energy(st, sys))(state_traj, system_traj),
            axis=-1
        )
        fluctuation.append(jnp.std(ke_t + pe) / jnp.mean(ke_t + pe))

        import matplotlib.pyplot as plt
        plt.plot(ke_t + pe)
        plt.plot(ke_t)
        plt.plot(pe)
        plt.savefig('energies.png')
        plt.close()

        from bump_utils import animate
        animate(state_traj, system_traj, 'test.gif', id_name='deformable_ID')
        exit()

    fluctuation = jnp.array(fluctuation)
    exponent, _ = np.polyfit(np.log(dts), np.log(fluctuation), 1)
    print(f"Total Energy Fluctuation ~ dt^{exponent:.2f}.  Expected: dt^{EXPECTED_EXPONENT}")


if __name__ == "__main__":
    dim = 2
    phi = 0.7
    N = 100
    Nv = 10
    asperity_radius = 0.15 if dim == 2 else 0.3
    mass = 1.0
    target_temperature = 1e-4
    e_int = 1.0
    e_m = 1.0
    e_b = 1e-5
    e_c = 1e3

    for collider_type in ["naive", "neighborlist"]:
        print(f'Running {collider_type}')
        measure_energy_conservation_dps(
            phi, N, mass, target_temperature, dim, e_int, e_m, e_b, e_c, Nv, asperity_radius,
            collider_type=collider_type,
        )
