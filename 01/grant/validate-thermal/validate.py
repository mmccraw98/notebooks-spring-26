"""
Validate all thermal functions for states of spheres and clumps.
Load basic test state,systems.
Verify:
- velocities are being assigned correctly (uniformly for clumps and corresponding to a temperature)
- temperatures are being set correctly
- energies are being calculated correctly
"""

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import os

def assert_uniform_per_clump(x, clump_ID, num_clumps, tol=0.0):
    counts = jnp.bincount(clump_ID, length=num_clumps)
    x_max = jax.ops.segment_max(x, clump_ID, num_segments=num_clumps)
    x_min = jax.ops.segment_max(-x, clump_ID, num_segments=num_clumps)
    spread = x_max + x_min
    spread = jnp.where(counts[:, None] > 0, spread, 0.0)
    assert jnp.all(spread <= tol)

if __name__ == "__main__":
    root = '/home/mmccraw/dev/data/26-01-01/grant/validate-thermal'
    data_root = os.path.join(root, 'initial-data')
    save_root = os.path.join(root, 'validate-data')

    target_temperature_1 = 1e-4
    target_temperature_2 = 1e-5

    for p_dim_type in ['2d-spheres', '3d-spheres', '2d-clumps', '3d-clumps']:
        print(f'Testing type: {p_dim_type}')
        can_rotate = 'spheres' not in p_dim_type
        state = jd.utils.h5.load(os.path.join(data_root, p_dim_type, 'state.h5'))
        system = jd.utils.h5.load(os.path.join(data_root, p_dim_type, 'system.h5'))
        pe = jd.utils.thermal.compute_potential_energy(state, system)
        pe_expected = jnp.sum(system.collider.compute_potential_energy(state, system))
        assert jnp.isclose(pe, pe_expected)
        assert jnp.isclose(pe, 0)

        cids, offsets = jnp.unique(state.clump_ID, return_index=True)
        N_particles = cids.size
        dim = state.dim

        for subtract_drift in [True, False]:
            total_dofs, t_dofs, r_dofs = jd.utils.thermal.count_dynamic_dofs(state, can_rotate, subtract_drift)
            t_dofs_expected = int(dim) * (N_particles - subtract_drift)
            r_dofs_expected = int(dim * (dim - 1) / 2) * (N_particles) * (can_rotate)
            assert t_dofs_expected == t_dofs
            assert r_dofs_expected == r_dofs
            assert t_dofs_expected + r_dofs_expected == total_dofs

            state = jd.utils.thermal.set_temperature(state, target_temperature_1, can_rotate, subtract_drift)
            temperature = jd.utils.thermal.compute_temperature(state, can_rotate, subtract_drift)
            assert jnp.isclose(temperature, target_temperature_1)

            assert_uniform_per_clump(state.vel, state.clump_ID, N_particles)
            assert_uniform_per_clump(state.angVel, state.clump_ID, N_particles)

            ke_t = jd.utils.thermal.compute_translational_kinetic_energy(state)
            ke_t_expected = 0.5 * jnp.sum(state.mass[offsets, None] * state.vel[offsets] ** 2)
            assert jnp.isclose(ke_t, ke_t_expected)

            ke_r = jd.utils.thermal.compute_rotational_kinetic_energy(state)
            if state.dim == 2:
                w_body = state.angVel
            else:
                w_body = state.q.rotate_back(state.q, state.angVel)
            ke_r_expected = 0.5 * jnp.sum(jnp.vecdot(w_body, state.inertia * w_body)[offsets]) * can_rotate
            assert jnp.isclose(ke_r, ke_r_expected)

            te = jd.utils.thermal.compute_energy(state, system)
            te_expected = pe_expected + ke_t_expected + ke_r_expected
            assert jnp.isclose(te, te_expected)

            temp_expected = 2 * (ke_r + ke_t) / (total_dofs)
            assert jnp.isclose(temp_expected, temperature)

            state = jd.utils.thermal.scale_to_temperature(state, target_temperature_2, can_rotate, subtract_drift)
            temperature = jd.utils.thermal.compute_temperature(state, can_rotate, subtract_drift)
            assert jnp.isclose(target_temperature_2, temperature)
        print(f'Test type: {p_dim_type} passed')
    