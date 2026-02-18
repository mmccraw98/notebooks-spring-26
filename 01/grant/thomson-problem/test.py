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

# TODO: add energy threshold
# TODO: prove the best lr scaling with N

# generate N random points on the unit sphere
N = 100
S = 10000
chunk_size = 100
alpha = 1
energy_rand = N * (N - 1) / 2  # this is the energy of a random distribution, it should be treated as the baseline
key = jax.random.PRNGKey(0)
# pos = random_points_on_sphere(key, N)
# # final_pos, energy_trace = minimize_on_sphere(pos, alpha=1, lr=0.01 / N, steps=10_000)
# final_pos, energy_trace = minimize_on_sphere(pos, alpha=1, lr=0.01 / N, steps=10000)
# energy_trace /= energy_rand



from tqdm import tqdm
from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt

def hist(val, bins=None, log=False):
    val = np.asarray(val)
    if bins is None:
        n = int(np.sqrt(len(val)))
        if log:
            bins = np.logspace(
                min(np.log10(val[val > 0])),
                max(np.log10(val[val > 0])),
                n
            )
        else:
            bins = np.linspace(
                min(val),
                max(val),
                n
            )
    p_val, edges = np.histogram(
        val,
        bins,
        density=True
    )
    return p_val, (edges[1:] + edges[:-1]) / 2

N = 100
S = 10000
chunk_size = 100
alpha = 1
energy_rand = N * (N - 1) / 2
key = jax.random.PRNGKey(0)

for N_steps in [0, 10, 100, 1_000, 10_000]:
    points = jax.random.normal(key, shape=(S, N, 3))
    norms = jnp.linalg.norm(points, axis=-1, keepdims=True)
    pos = points / norms

    final_pos = []
    for chunk_id in tqdm(range(S // chunk_size)):
        _pos = pos[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
        _final_pos, energy = jax.vmap(lambda p: minimize_on_sphere(p, alpha=alpha, lr=0.01 / N, steps=N_steps))(_pos)
        final_pos.append(_final_pos)
    final_pos = jnp.concatenate(final_pos)

    edge_lengths = []
    for s in range(S):
        hull = ConvexHull(final_pos[s])
        edge_lengths.append(np.linalg.norm(np.diff(hull.points[hull.simplices], axis=1), axis=-1).flatten())
    edge_lengths = np.concatenate(edge_lengths)
    p_e, e = hist(edge_lengths)
    plt.plot(e, p_e, label=N_steps)

plt.legend()
plt.yscale('log')
plt.savefig('edge_lengths.png')
plt.close()

# scale the sphere to a desired radius

# make the clumps