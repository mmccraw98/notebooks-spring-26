import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from jaxdem.utils.geometricAsperityCreation import generate_mesh, compute_mesh_properties

def random_points_on_sphere(key, N, S=1):
    """Generate n random points uniformly distributed on the unit sphere."""
    points = jax.random.normal(key, shape=(S, N, 3))
    norms = jnp.linalg.norm(points, axis=1, keepdims=True)
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


nv = 10
N = 10
N_steps = 1000
asperity_radius = 0.3
particle_radius = 0.5
mesh_subdivisions = 5
mass = 1.0

key = jax.random.PRNGKey(0)
asperity_pos_init = random_points_on_sphere(key, N=nv, S=N)

asperity_pos, config_energy = jax.vmap(
    lambda p: minimize_on_sphere(
        p,
        alpha=1,
        lr=0.01 / nv,
        steps=N_steps
    )
)(asperity_pos_init)

asperity_pos *= particle_radius  # scale to the right size

meshes = []
for a_pos in asperity_pos:
    mesh = generate_mesh(
        asperity_positions=a_pos,
        asperity_radii=jnp.ones(a_pos.shape[0]) * asperity_radius,
        subdivisions=mesh_subdivisions,
    )
    pos_c, q, inertia, vol = compute_mesh_properties(mesh, mass)  # should cache these: f(nv, arad, prad, mass) -> p, q, i, v

    print(vol, inertia)

