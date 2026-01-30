import jaxdem as jd
import jax
import numpy as np
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from jaxdem.utils.geometricAsperityCreation import generate_ga_clump_state, generate_ga_deformable_state
from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration

EXPECTED_EXPONENT = 2.0
DTS = jnp.array([1e-3, 3e-3, 1e-2])
N_STEPS_MAX = 100_000
SAVE_STRIDE_MAX = 1_000


def calc_trans_ke(state, n_clumps):
    _, offsets = jnp.unique(state.clump_ID, return_index=True, size=n_clumps)
    return 0.5 * jnp.sum(
        ((1 - state.fixed[offsets]) * state.mass[offsets])[:, None] * state.vel[offsets] ** 2
    )


def calc_rot_ke(state, n_clumps):
    _, offsets = jnp.unique(state.clump_ID, return_index=True, size=n_clumps)
    w_body = state.angVel if state.dim == 2 else state.q.rotate_back(state.q, state.angVel)
    return 0.5 * jnp.sum(
        ((1 - state.fixed[offsets])[:, None] * state.inertia[offsets]) * w_body[offsets]**2
    )


def get_collider_kw(collider_type, state):
    if collider_type == "naive":
        return {}
    elif collider_type == "neighborlist":
        return dict(state=state, cutoff=2.0 * jnp.max(state.rad), skin=0.05)
    elif collider_type in ("celllist", "staticcelllist"):
        return dict(state=state)
    raise ValueError(f'Unknown collider_type: {collider_type}')


def get_domain_kw(domain_type, box_size):
    if domain_type == 'free':
        return {}
    elif domain_type in ('periodic', 'reflect', 'reflectsphere'):
        return dict(box_size=box_size)
    raise ValueError(f'Unknown domain_type: {domain_type}')


def get_system(state, e_int, dt, domain_type='free', domain_kw=None, collider_type='naive',
               collider_kw=None, force_manager_kw=None, rotation_integrator_type=""):
    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    mat_table = jd.MaterialTable.from_materials(mats, matcher=jd.MaterialMatchmaker.create("harmonic"))
    return jd.System.create(
        state_shape=state.shape, dt=dt, linear_integrator_type="verlet",
        rotation_integrator_type=rotation_integrator_type, domain_type=domain_type,
        force_model_type="spring", collider_type=collider_type,
        collider_kw=collider_kw or {}, mat_table=mat_table,
        domain_kw=domain_kw or {}, force_manager_kw=force_manager_kw or {},
    )


def compute_pe(state_traj, system_traj):
    return jnp.sum(jax.vmap(lambda st, sys: sys.collider.compute_potential_energy(st, sys))(
        state_traj, system_traj), axis=-1)


def measure_fluctuation_exponent(dts, fluctuations):
    exponent, _ = np.polyfit(np.log(dts), np.log(fluctuations), 1)
    print(f"Total Energy Fluctuation ~ dt^{exponent:.2f}.  Expected: dt^{EXPECTED_EXPONENT}")
    return exponent


def measure_energy_conservation(create_state_fn, compute_ke_fn, e_int, collider_type, domain_type,
                                rotation_integrator_type="", force_manager_kw=None):
    n_steps = (N_STEPS_MAX * DTS.min() / DTS).astype(int)
    save_strides = (SAVE_STRIDE_MAX * DTS.min() / DTS).astype(int)

    fluctuations = []
    for dt, n_step, save_stride in zip(DTS, n_steps, save_strides):
        state, box_size = create_state_fn()
        collider_kw = get_collider_kw(collider_type, state)
        domain_kw = get_domain_kw(domain_type, box_size)

        system = get_system(
            state, e_int, float(dt), domain_type, domain_kw, collider_type, collider_kw,
            force_manager_kw, rotation_integrator_type
        )

        n_snapshots = int(n_step) // int(save_stride)
        _, _, (state_traj, system_traj) = system.trajectory_rollout(
            state, system, n=n_snapshots, stride=int(save_stride)
        )

        ke = compute_ke_fn(state_traj)
        pe = compute_pe(state_traj, system_traj)
        total_energy = ke + pe
        fluctuations.append(jnp.std(total_energy) / jnp.mean(total_energy))

    return measure_fluctuation_exponent(DTS, jnp.array(fluctuations))


def test_spheres(phi=0.7, N=100, mass=1.0, target_temperature=1e-4, e_int=1.0,
               domain_type='periodic', colliders=("naive", "neighborlist")):
    print('\n\n', '-'*50, '\nTesting Spheres')
    dim = 2
    radii = jd.utils.dispersity.get_polydisperse_radii(N, count_ratios=[1.0], size_ratios=[1.0])
    pos, box_size = random_sphere_configuration(radii, phi, dim)

    def create_state():
        state = jd.State.create(pos=pos.copy(), rad=radii.copy(), mass=jnp.ones(N) * mass)
        state = jd.utils.thermal.set_temperature(state, target_temperature, can_rotate=False, subtract_drift=True)
        return state, box_size

    def compute_ke(state_traj):
        return 0.5 * jnp.sum(
            (1 - state_traj.fixed) * state_traj.mass * jnp.sum(state_traj.vel ** 2, axis=-1), axis=-1
        )

    for collider_type in colliders:
        print(f'\nRunning {collider_type}')
        measure_energy_conservation(create_state, compute_ke, e_int, collider_type, domain_type)


def test_clumps(phi=0.7, N=100, Nv=10, mass=1.0, target_temperature=1e-4, e_int=1.0,
                domain_type='periodic', colliders=("naive", "neighborlist")):
    print('\n\n', '-'*50, '\nTesting Clumps')
    dim = 2
    asperity_radius = 0.15 if dim == 2 else 0.3
    radii = jd.utils.dispersity.get_polydisperse_radii(N, count_ratios=[1.0], size_ratios=[1.0])
    vertex_counts = jnp.ones_like(radii) * Nv
    base_state, box_size = generate_ga_clump_state(radii, vertex_counts, phi, dim, asperity_radius, core_type=None, mass=mass)

    def create_state():
        state = jax.tree.map(lambda x: x, base_state)
        state = jd.utils.thermal.set_temperature(state, target_temperature, can_rotate=True, subtract_drift=True)
        return state, box_size

    def compute_ke(state_traj):
        ke_t = jax.vmap(lambda st: calc_trans_ke(st, n_clumps=N))(state_traj)
        ke_r = jax.vmap(lambda st: calc_rot_ke(st, n_clumps=N))(state_traj)
        return ke_t + ke_r

    for collider_type in colliders:
        print(f'\nRunning {collider_type}')
        measure_energy_conservation(create_state, compute_ke, e_int, collider_type, domain_type,
                                    rotation_integrator_type="verletspiral")


def test_dps(phi=0.7, N=100, Nv=10, mass=1.0, target_temperature=1e-4, e_int=1.0,
             e_m=1.0, e_b=1e-5, e_c=1e3, domain_type='periodic', colliders=("naive", "neighborlist")):
    print('\n\n', '-'*50, '\nTesting DPs')
    dim = 2
    asperity_radius = 0.15 if dim == 2 else 0.3
    radii = jd.utils.dispersity.get_polydisperse_radii(N, count_ratios=[1.0], size_ratios=[1.0])
    vertex_counts = jnp.ones_like(radii) * Nv
    base_state, dp, box_size = generate_ga_deformable_state(radii, vertex_counts, phi, dim, asperity_radius,
                                                            core_type=None, em=e_m, eb=e_b, ec=e_c, mass=mass)
    force_manager_kw = dict(force_functions=(dp.create_force_function(dp),))

    def create_state():
        state = jax.tree.map(lambda x: x, base_state)
        state = jd.utils.thermal.set_temperature(state, target_temperature, can_rotate=True, subtract_drift=True)
        return state, box_size

    def compute_ke(state_traj):
        return jax.vmap(lambda st: calc_trans_ke(st, n_clumps=N))(state_traj)

    for collider_type in colliders:
        print(f'\nRunning {collider_type}')
        measure_energy_conservation(create_state, compute_ke, e_int, collider_type, domain_type,
                                    force_manager_kw=force_manager_kw)


if __name__ == "__main__":
    test_spheres()
    test_clumps()
    # test_dps()  # NOT WORKING YET
