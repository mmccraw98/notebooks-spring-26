from __future__ import annotations

import jax
import jaxdem as jd
import jax.numpy as jnp
import numpy as np
from pathlib import Path
jax.config.update("jax_enable_x64", True)


# Import the generator from the installed jaxdem package
from jaxdem.utils.geometricAsperityCreation import generate_ga_clump_state


def main():
    # Parameters
    n_particles = 1000
    particle_radius = 0.5
    vertex_count = 40
    dim = 3
    phi = 0.4  # target packing fraction (adjust if needed)
    asperity_radius = 0.3  # radius of asperities on surface; tweak as desired
    seed = 42

    # Build arrays as plain NumPy types so external libs (meshzoo, trimesh) get Python/NumPy inputs
    particle_radii = np.ones((n_particles,), dtype=float) * float(particle_radius)
    vertex_counts = np.ones((n_particles,), dtype=int) * int(vertex_count)

    print(f"Building GA system: n={n_particles}, particle_radius={particle_radius}, vertex_count={vertex_count}")

    # Call the generator. For dim=3 pass an explicit 3-element aspect_ratio.
    # Many generator internals expect plain Python ints/tuples; we feed NumPy scalars/tuples.
    state, box_size = generate_ga_clump_state(
        particle_radii,
        vertex_counts,
        phi,
        dim,
        asperity_radius,
        seed=seed,
        add_core=True,
        use_uniform_mesh=False,
        mass=1.0,
        aspect_ratio=(1.0, 1.0, 1.0),
    )

    e_int = 1.0
    dt = 1e-2

    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

    cutoff = 2.0 * jnp.max(state.rad)
    number_density = state.N / jnp.prod(box_size)
    packing_fraction_true = 4 / 3 * jnp.pi * jnp.sum(state.rad**3) / jnp.prod(box_size)

    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="linearfire",
        rotation_integrator_type="rotationfire",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="neighborlist",
        collider_kw=dict(
                state=state,
                cutoff=cutoff,
                skin=0.05,
                max_neighbors=100,
                ),
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )

    state, system, final_pf, final_pe = jd.utils.jamming.bisection_jam(state, system)

    jd.utils.h5.save(state, 'state.h5')
    jd.utils.h5.save(system, 'system.h5')


if __name__ == "__main__":
    main()