import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import jaxdem as jd

import os

from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration

from utils import animate_flocks_2d

# GOOD:
# phi = 0.1, 0.4, 0.9
# eta = 0.1, 0.4, 0.8, 1.2

for phi in [0.1, 0.4, 0.9]:
    for eta in [0.1, 0.4, 0.8, 1.2]:

        fname_prefix = f'phi-{phi}-eta-{eta}'
        found = False
        for fname in os.listdir('examples'):
            if fname_prefix in fname and not found:
                found = True
        if found:
            continue

        N = 10000
        dim = 2
        dt = 1e-2
        seed = np.random.randint(0, 1e9)

        neighbor_radius = 1.5  # interaction radius for velocity alignment
        v0 = 1.0               # constant speed
        max_neighbors = 64

        skin = 0.1

        # Rollout parameters
        n_steps = 10_000
        save_stride = 20
        n_frames = n_steps // save_stride

        def build_microstate():
            # A mild polydispersity similar to temperature_density_control.py
            cr = [1.0]
            sr = [1.0]
            particle_radii = jd.utils.dispersity.get_polydisperse_radii(N, cr, sr)

            pos, box_size = random_sphere_configuration(particle_radii, phi, dim, seed)

            st = jd.State.create(
                pos=pos,
                rad=particle_radii,
                mass=jnp.ones(N),
            )

            mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
            matcher = jd.MaterialMatchmaker.create("harmonic")
            mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

            sys = jd.System.create(
                state_shape=st.shape,
                dt=dt,
                # Vicsek integrator:
                linear_integrator_type="vicsek_extrinsic",
                linear_integrator_kw=dict(
                    neighbor_radius=jnp.asarray(neighbor_radius, dtype=float),
                    eta=jnp.asarray(eta, dtype=float),
                    v0=jnp.asarray(v0, dtype=float),
                    max_neighbors=max_neighbors,
                ),
                rotation_integrator_type="",
                domain_type="periodic",
                domain_kw=dict(box_size=box_size),
                # Contact model still defines the collider force part of the Vicsek direction
                force_model_type="spring",
                mat_table=mat_table,
                # Use cached neighbor list collider
                # collider_type="naive",
                collider_type="StaticCellList",
                collider_kw=dict(
                    state=st,
                    cell_size=float(neighbor_radius),
                    max_occupancy=max_neighbors,
                ),
                # collider_type="NeighborList",
                # collider_kw=dict(
                #     state=st,
                #     cutoff=float(neighbor_radius),
                #     box_size=jnp.asarray(box_size, dtype=float),
                #     skin=skin,
                #     max_neighbors=max_neighbors,
                # ),
            )
            return st, sys

        state, system = build_microstate()
        print('Simulating...')
        state_f, system_f, (traj_state, traj_system) = jd.System.trajectory_rollout(
            state,
            system,
            n=n_frames,
            stride=save_stride,
            unroll=2,
        )
        print('Done.')

        pol = jnp.mean(jnp.linalg.norm(jnp.mean(traj_state.vel / v0, axis=-2), axis=-1))

        print('Animating...')
        # animate(traj_state, traj_system, 'anim.gif')
        animate_flocks_2d(traj_state, traj_system, save_path=f'examples/phi-{phi}-eta-{eta}-pol-{pol}.mp4')
        print('Done.')

