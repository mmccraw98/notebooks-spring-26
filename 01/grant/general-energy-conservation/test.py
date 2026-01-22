
import jaxdem as jd
import jax
import numpy as np
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration

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

EXPECTED_EXPONENT = 2.0

phi = 0.7
N = 100
mass = 1.0
target_temperature = 1e-3
dim = 2
e_int = 1.0
dt = 1e-2
collider_type = "naive"
collider_kw = dict()

save_stride_max = 1_000
n_steps_max = 100_000
dts = jnp.array([1e-3, 3e-3, 1e-2])
n_steps = (n_steps_max * dts.min() / dts).astype(dtype=type(n_steps_max))
save_strides = (save_stride_max * dts.min() / dts).astype(dtype=type(n_steps_max))

# dict(
#     box_size=box_size,
# )
# collider_type="neighborlist",
# collider_kw=dict(
#     state=state,
#     cutoff=2.0 * jnp.max(state.rad),
#     skin=0.03,
# ),
# dict(
#     force_functions=(dp_container.create_force_function(dp_container),),
# )

particle_radii = jd.utils.dispersity.get_polydisperse_radii(N)
pos, box_size = random_sphere_configuration(particle_radii, phi, dim)

fluctuation = []
for dt, n_step, save_stride in zip(dts, n_steps, save_strides):
    n_step = int(n_step); save_stride = int(save_stride); dt = float(dt)
    state = jd.State.create(
        pos=pos.copy(),
        rad=particle_radii.copy(),
        mass=jnp.ones_like(particle_radii) * mass
    )
    state = jd.utils.thermal.set_temperature(state, target_temperature, is_rigid=False, subtract_drift=True)
    system = get_system(
        state=state,
        e_int=e_int,
        dt=dt,
        domain_type='periodic',
        domain_kw=dict(box_size=box_size),
        collider_type=collider_type,
        collider_kw=collider_kw
    )
    n_snapshots = n_step // save_stride
    state, system, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=n_snapshots, stride=save_stride
    )
    ke = jd.utils.thermal.calculate_translational_kinetic_energy(state_traj)
    pe = jnp.sum(
        jax.vmap(
            lambda st, sys:
            sys.collider.compute_potential_energy(st, sys))(state_traj, system_traj),
        axis=-1
    )
    fluctuation.append(jnp.std(ke + pe) / jnp.mean(ke + pe))

fluctuation = jnp.array(fluctuation)
exponent, _ = np.polyfit(np.log(dts), np.log(fluctuation), 1)
print(f"Total Energy Fluctuation ~ dt^{exponent:.2f}.  Expected: dt^{EXPECTED_EXPONENT}")
