from dataclasses import dataclass, fields
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import os
import numpy as np
import h5py
from tqdm import tqdm

from jaxdem.utils.geometricAsperityCreation import generate_mesh, compute_mesh_properties

def random_points_on_sphere(key, N, S=1):
    """Generate n random points uniformly distributed on the unit sphere."""
    points = jax.random.normal(key, shape=(S, N, 3))
    norms = jnp.linalg.norm(points, axis=-1, keepdims=True)
    return (points / norms).squeeze()

def riesz_energy(pos, alpha):
    """Riesz energy kernel.  alpha=1 reduces to the Thomson problem.  alpha=\infty reduces to the packing problem"""
    r_ij = pos[:, None, :] - pos[None, :, :]
    # squared distances (no gradient issue here)
    d_sq = jnp.sum(r_ij**2, axis=-1)
    # fill diagonal with 1.0 BEFORE sqrt, so grad(sqrt(1.0)) = 0.5, not inf
    n = pos.shape[0]
    d_sq = d_sq.at[jnp.diag_indices(n)].set(1.0)
    d_ij = jnp.sqrt(d_sq)
    e_ij = 1.0 / d_ij ** alpha
    # zero out the diagonal so self-interactions don't contribute
    e_ij = e_ij.at[jnp.diag_indices(n)].set(0.0)
    return jnp.sum(jnp.triu(e_ij, k=1))

def project_to_tangent(grad, pos):
    """Remove the radial component of the gradient (project onto tangent plane of sphere)."""
    # For unit sphere, the normal at each point is just the position itself
    radial = jnp.sum(grad * pos, axis=-1, keepdims=True) * pos
    return grad - radial

def minimize_on_sphere(pos, alpha, lr=0.01, steps=1000):
    energy_grad = jax.grad(riesz_energy)
    def step(pos, _):
        g = energy_grad(pos, alpha)
        g_tangent = project_to_tangent(g, pos)
        pos = pos - lr * g_tangent
        # retract back to the sphere (normalize each point)
        pos = pos / jnp.linalg.norm(pos, axis=-1, keepdims=True)
        return pos, riesz_energy(pos, alpha)
    pos, energies = jax.lax.scan(step, pos, None, length=steps)
    return pos, energies

@dataclass
class ThomsonParticle:
    nv: int
    asperity_radius: float
    particle_radius: float
    pos_c: jnp.array
    pos: jnp.array
    q: jnp.array
    inertia_dimensionless: jnp.array
    volume: float
    energy: float
    N_steps: int

SCALAR_FIELDS = {'nv', 'asperity_radius', 'particle_radius', 'volume', 'energy', 'N_steps'}

def cache_thomson_particle_properties(particles, cache_path):
    if not isinstance(particles, list):
        particles = [particles]
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with h5py.File(cache_path, 'a') as f:
        grp = f.require_group('particles')
        offset = len(grp)
        for i, particle in enumerate(particles):
            p_grp = grp.create_group(str(offset + i))
            for field in fields(particle):
                val = getattr(particle, field.name)
                if field.name in SCALAR_FIELDS:
                    p_grp.attrs[field.name] = val
                else:
                    p_grp.create_dataset(field.name, data=np.asarray(val))

def merge_thomson_particle_caches(source_paths, dest_path):
    cache_dir = os.path.dirname(dest_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with h5py.File(dest_path, 'a') as dest:
        grp = dest.require_group('particles')
        for src_path in source_paths:
            if not os.path.exists(src_path):
                continue
            with h5py.File(src_path, 'r') as src:
                if 'particles' not in src:
                    continue
                for key in src['particles']:
                    new_key = str(len(grp))
                    src.copy(src['particles'][key], grp, name=new_key)

def get_thomson_particle_properties(
    N,
    nv,
    asperity_radius,
    particle_radius,
    N_steps=None,
    energy=None,
    cache_path=None,
    load_from_cache=True,
    allow_generation=True,
    seed=None,
    mesh_subdivisions=5,
):
    matches = []

    if load_from_cache and cache_path is not None and os.path.exists(cache_path):
        with h5py.File(cache_path, 'r') as f:
            if 'particles' in f:
                matching_keys = []
                for key in f['particles']:
                    p = f['particles'][key]
                    if (p.attrs['nv'] == nv
                            and np.isclose(p.attrs['asperity_radius'], asperity_radius)
                            and np.isclose(p.attrs['particle_radius'], particle_radius)):
                        if N_steps is not None and p.attrs['N_steps'] != N_steps:
                            continue
                        if energy is not None and not np.isclose(p.attrs['energy'], energy):
                            continue
                        matching_keys.append(key)
                if len(matching_keys) > 0:
                    chosen = np.random.choice(
                        matching_keys, size=min(N, len(matching_keys)), replace=False
                    )
                    for key in chosen:
                        p = f['particles'][key]
                        matches.append(ThomsonParticle(
                            **{field.name: (
                                p.attrs[field.name] if field.name in SCALAR_FIELDS
                                else jnp.array(p[field.name][...])
                            ) for field in fields(ThomsonParticle)}
                        ))

    n_remaining = N - len(matches)
    if n_remaining > 0:
        if not allow_generation:
            raise ValueError(
                f"Only found {len(matches)} cached matches but requested {N}. "
                "Set allow_generation=True to generate the rest."
            )
        if N_steps is None:
            raise ValueError("N_steps is required when generating new particles.")
        if seed is None:
            seed = np.random.randint(0, 1e9)
        generated = generate_thomson_particle_properties(
            nv=nv,
            N=n_remaining,
            N_steps=N_steps,
            asperity_radius=asperity_radius,
            particle_radius=particle_radius,
            seed=seed,
            mesh_subdivisions=mesh_subdivisions,
        )
        if cache_path is not None:
            cache_thomson_particle_properties(generated, cache_path)
        matches.extend(generated)

    return matches

def generate_thomson_particle_properties(
    nv,
    N,
    N_steps,
    asperity_radius,
    particle_radius,
    seed,
    mesh_subdivisions,  # used for defining the shape properties
):
    random_energy_scale = nv * (nv - 1) / 2
    key = jax.random.PRNGKey(seed)
    asperity_pos_init = random_points_on_sphere(key, N=nv, S=N)
    if N_steps > 0:
        asperity_pos, config_energy = jax.vmap(
            lambda p: minimize_on_sphere(
                p,
                alpha=1,
                lr=0.01 / nv,
                steps=N_steps
            )
        )(asperity_pos_init)
    else:
        asperity_pos = asperity_pos_init
        config_energy = jnp.ones((N, 1))
    asperity_pos *= particle_radius
    particles = []
    for a_pos, c_en in tqdm(
        zip(asperity_pos, config_energy[:, -1]),
        desc='Calculating solid body properties...',
        total=len(asperity_pos)
    ):
        mesh = generate_mesh(
            asperity_positions=a_pos,
            asperity_radii=jnp.ones(a_pos.shape[0]) * asperity_radius,
            subdivisions=mesh_subdivisions,
        )
        pos_c, q, inertia_dimensionless, volume = compute_mesh_properties(mesh, mass=1.0)
        particles.append(
            ThomsonParticle(
                nv=nv,
                asperity_radius=asperity_radius,
                particle_radius=particle_radius,
                pos_c=pos_c,
                pos=a_pos,
                q=q,
                inertia_dimensionless=inertia_dimensionless,
                volume=volume,
                energy=c_en / random_energy_scale,
                N_steps=N_steps,
            )
        )
    return particles


for nv in [20, 50, 100]:
    for asperity_radius in [0.08, 0.12, 0.14, 0.18, 0.25, 0.35, 0.45]:
        for N_steps in [0, 10, 100, 1_000, 10_000, 100_000]:
            get_thomson_particle_properties(
                N=1000,
                nv=nv,
                asperity_radius=asperity_radius,
                particle_radius=0.5,
                N_steps=N_steps,
                energy=None,
                cache_path='thomson-particle-cache.h5',
                load_from_cache=True,
                allow_generation=True,
                seed=None,
                mesh_subdivisions=6,
            )

# TODO: validate the mesh subdivisions is high enough for stable calculation of nv=100 (what is the convergence?)
# TODO: allow defining the particles based on a target final energy as a ratio of the random energy
