import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os

import numpy as np
import matplotlib.pyplot as plt

from contacts import get_clump_rattler_ids, remove_rattlers_from_state

root = os.path.join('data', '010_10')

state = jd.utils.h5.load(os.path.join(root, 'state.h5'))
system = jd.utils.h5.load(os.path.join(root, 'system.h5'))

state, system = system.collider.compute_force(state, system)  # force the neighbor list to  update

n_steps = 10_000
save_stride = 10
cutoff = jnp.max(state.rad) * 3
max_neighbors = 100
dt = 1e-2
e_int = 1.0
seed = np.random.randint(0, 1e9)
can_rotate = True
subtract_drift = True
n_phi_steps = 20
n_temperature_steps = 20

phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
temperatures = jnp.linspace(1e-5, 2e-5, n_temperature_steps)
delta_phis = - jnp.logspace(-4, jnp.log10(phi / 2), n_phi_steps)

state, system, rattler_ids, non_rattler_ids = get_clump_rattler_ids(state, system, cutoff, max_neighbors)
base_state = remove_rattlers_from_state(state, rattler_ids)

mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
matcher = jd.MaterialMatchmaker.create("harmonic")
mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
base_system = jd.System.create(
    state_shape=base_state.shape,
    dt=dt,
    linear_integrator_type="verlet",
    rotation_integrator_type="verletspiral",
    domain_type="periodic",
    force_model_type="spring",
    collider_type="neighborlist",
    collider_kw=dict(
        state=base_state,
        cutoff=2.0 * jnp.max(base_state.rad),
        skin=0.05,
        safety_factor=5.0,
    ),
    mat_table=mat_table,
    domain_kw=dict(
        box_size=system.domain.box_size,
    ),
)

save_steps = jnp.arange(save_stride, n_steps + save_stride, save_stride)

def save_fn(st, sy):
    return (
        jd.utils.thermal.compute_potential_energy(st, sy),
        jd.utils.thermal.compute_translational_kinetic_energy(st),
        jd.utils.thermal.compute_rotational_kinetic_energy(st),
        jd.utils.thermal.compute_temperature(st, can_rotate, subtract_drift),
    )

for delta_phi in delta_phis:
    interm_state, interm_system = jd.utils.packingUtils.scale_to_packing_fraction(base_state, base_system, phi + delta_phi)
    state = jd.State.stack([interm_state for _ in range(temperatures.size)])
    state = jax.vmap(
        lambda st, temp: jd.utils.thermal.set_temperature(
            st,
            temp,
            can_rotate,
            subtract_drift,
            seed
        )
    )(state, temperatures)
    system = jd.System.stack([interm_system for _ in range(temperatures.size)])

    state, system, logged = system.trajectory_rollout_at_steps(
        state, system, save_steps=save_steps, save_fn=save_fn,
    )

    pe, ke, ke_r, temp = logged

    np.savez(
        f'thermal_{delta_phi}.npz',
        pe=pe,
        ke=ke,
        ke_r=ke_r,
        temp=temp,
        target_temp=temperatures,
        delta_phi=delta_phi,
    )