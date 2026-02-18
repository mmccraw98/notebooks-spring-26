import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

def random_points_on_sphere(key, n):
    """Generate n random points uniformly distributed on the unit sphere."""
    points = jax.random.normal(key, shape=(n, 3))
    norms = jnp.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def riesz_energy(pos, alpha):
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

# generate N random points on the unit sphere
N = 1000
alpha = 1
key = jax.random.PRNGKey(0)
pos = random_points_on_sphere(key, N)

final_pos, energy_trace = minimize_on_sphere(pos, alpha=1, lr=0.00001, steps=5000)

import matplotlib.pyplot as plt
plt.plot(energy_trace)
plt.yscale('log')
plt.xscale('log')
plt.savefig('energy.png')
plt.close()

# scale the sphere to a desired radius

# (make sure it can be vmapped)

# make the clumps