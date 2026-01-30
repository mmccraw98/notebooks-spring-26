# %%

import jax
import jax.numpy as jnp
import jaxdem as jd
from tqdm import tqdm
from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration

# %%
# Parameters
# ~~~~~~~~~~~~~~~~~~~~~
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

N = 10000
dim = 2
dt = 1e-2
seed = 0
e_int = 0.1

phis = [0.1, 0.4, 0.6, 0.9]
etas = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

cmap = plt.cm.viridis
norm = plt.Normalize(min(phis), max(phis))

for phi in phis:

    # Vicsek parameters
    neighbor_radius = 1.5  # interaction radius for velocity alignment
    v0 = 1.0               # constant speed
    max_neighbors = 64

    # NeighborList collider parameters
    skin = 0.1

    # Rollout parameters
    n_steps = 5_000
    save_stride = 50
    n_frames = n_steps // save_stride

    pols = []
    for eta in tqdm(etas):
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

            mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
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

        state_f, system_f, (traj_state, traj_system) = jd.System.trajectory_rollout(
            state,
            system,
            n=n_frames,
            stride=save_stride,
            unroll=2,
        )

        num_frames = traj_state.vel.shape[0]
        pols.append(jnp.mean(jnp.linalg.norm(jnp.mean(traj_state.vel / v0, axis=-2), axis=-1)[num_frames // 2:]))
    plt.plot(etas, pols, c=cmap(norm(phi)), marker='x')

sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array(phis)
cbar = plt.colorbar(sm, ax=plt.gca(), label=r'$\phi$')

plt.xlabel(r'$\eta$', fontsize=16)
plt.ylabel(r'$\Phi$', fontsize=16)
plt.tight_layout()
plt.savefig('extrinsic-pol.png', dpi=600)
plt.close()