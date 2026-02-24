from dataclasses import dataclass, fields
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


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



from typing import Optional, Tuple
from jaxdem import Material, MaterialTable, State
from jaxdem.utils.clumps import compute_clump_properties

def compute_clump_properties_from_spheres(
    positions: jax.Array,
    radii: jax.Array,
    n_samples: int = 50_000,
    batch_size: Optional[int] = None,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Return (pos_c, q, inertia, volume) for a single clump.

    Accepts either:
      - positions: (N, dim), radii: (N,)        → single clump
      - positions: (S, N, dim), radii: (S, N)   → batch of S clumps

    When batched, batch_size controls how many clumps are processed
    concurrently (None = all at once via vmap).
    """
    batched = positions.ndim == 3

    if not batched:
        positions = positions[None]  # (1, N, dim)
        radii = radii[None]          # (1, N)

    S, N, dim = positions.shape

    mat = Material.create("elastic", density=1.0, young=1.0, poisson=0.3)
    mat_table = MaterialTable.from_materials([mat])

    def single(pos_rad):
        pos, rad = pos_rad[:, :dim], pos_rad[:, dim]
        state = State.create(
            pos=pos,
            rad=rad,
            clump_ID=jnp.zeros(N, dtype=int),
        )
        state = compute_clump_properties(state, mat_table, n_samples=n_samples)

        volume = state.mass[0]
        pos_c = state.pos_c[0]
        q = jnp.concatenate([state.q.w[0], state.q.xyz[0]])
        inertia = state.inertia[0] / volume
        return jnp.concatenate([pos_c, q, inertia, volume[None]])

    # Pack pos and rad together so we have a single array to map over
    packed = jnp.concatenate([positions, radii[..., None]], axis=-1)  # (S, N, dim+1)

    if batch_size is None or batch_size >= S:
        results = jax.vmap(single)(packed)
    else:
        # Chunked sequential processing via lax.map over chunks
        n_chunks = -(-S // batch_size)  # ceil division
        pad_total = n_chunks * batch_size
        packed_padded = jnp.pad(
            packed,
            ((0, pad_total - S), (0, 0), (0, 0)),
            mode="edge",
        )
        chunks = packed_padded.reshape(n_chunks, batch_size, N, dim + 1)

        def process_chunk(chunk):
            return jax.vmap(single)(chunk)

        results = jax.lax.map(process_chunk, chunks)
        results = results.reshape(pad_total, -1)[:S]

    # Unpack
    pos_c = results[:, :dim]
    q = results[:, dim:dim + 4]
    inertia_dim = 1 if dim == 2 else 3
    inertia = results[:, dim + 4:dim + 4 + inertia_dim]
    volume = results[:, -1]

    if not batched:
        return pos_c[0], q[0], inertia[0], volume[0]

    return pos_c, q, inertia, volume




N_steps = 1_000
nv = 100
seed = 0
particle_radius = 0.5
asperity_rads = [0.01, 0.1, 0.4]

random_energy_scale = nv * (nv - 1) / 2
key = jax.random.PRNGKey(seed)
asperity_pos_init = random_points_on_sphere(key, N=nv)
asperity_pos, config_energy = minimize_on_sphere(
    asperity_pos_init,
    alpha=1,
    lr=0.01 / nv,
    steps=N_steps
)
asperity_pos *= particle_radius

for asperity_radius in asperity_rads:
    asperity_radii = jnp.ones(asperity_pos.shape[0]) * asperity_radius

    mesh_subdivisions = list(range(8))
    n_samples_list = [10_000, 100_000, 1_000_000]
    vols = []
    Is = []
    for ms in tqdm(mesh_subdivisions):
    # for n_samples in tqdm(n_samples_list):
        mesh = generate_mesh(
            asperity_positions=asperity_pos,
            asperity_radii=asperity_radii,
            subdivisions=ms,
        )
        pos_c, q, inertia_dimensionless, volume = compute_mesh_properties(mesh, mass=1.0)
        # pos_c, q, inertia_dimensionless, volume = compute_clump_properties_from_spheres(asperity_pos, asperity_radii, n_samples)
        vols.append(volume)
        Is.append(inertia_dimensionless)
    vols = np.array(vols)
    Is = np.array(Is)

    plt.plot(mesh_subdivisions, 1 - vols / vols[-1])
    # plt.plot(n_samples_list, abs(1 - vols / vols[-1]))
    print(vols)
plt.yscale('log')
plt.xlabel(r'$N_{sub}$', fontsize=16)
plt.ylabel(r'$|1 - V(n) / V(7)|$', fontsize=16)
plt.tight_layout()
plt.savefig('figures/mesh-validation-vols.png', dpi=600)