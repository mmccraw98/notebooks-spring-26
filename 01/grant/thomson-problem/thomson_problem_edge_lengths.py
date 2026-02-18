import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.spatial import ConvexHull
from tqdm import tqdm
from uuid import uuid4

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

# Measure edge-lengths for different N
for N in [10, 50, 100, 1000]:
    S = 10000
    memory_budget = 1e9
    bytes_per_pair = 8 * 10
    chunk_size = min(S, max(1, int(memory_budget / (bytes_per_pair * N ** 2))))
    alpha = 1
    energy_rand = N * (N - 1) / 2
    key = jax.random.PRNGKey(0)

    plot_values = []

    for N_steps in [0, 10, 100, 1_000, 10_000, 100_000]:
        key, sub_key = jax.random.split(key, 2)
        points = jax.random.normal(sub_key, shape=(S, N, 3))
        norms = jnp.linalg.norm(points, axis=-1, keepdims=True)
        pos = points / norms

        if N_steps == 0:
            final_pos = pos
            energy = jnp.ones(S)
        else:
            final_pos = []
            energy = []
            for chunk_id in tqdm(range(S // chunk_size)):
                _pos = pos[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
                _final_pos, _energy = jax.vmap(lambda p: minimize_on_sphere(p, alpha=alpha, lr=0.01 / N, steps=N_steps))(_pos)
                _energy /= energy_rand
                final_pos.append(_final_pos)
                energy.append(_energy[:, -1])
            final_pos = jnp.concatenate(final_pos)
            energy = jnp.concatenate(energy)

        edge_lengths = []
        for s in range(S):
            hull = ConvexHull(final_pos[s])
            edges = set()
            for simplex in hull.simplices:
                for i in range(3):
                    edge = tuple(sorted((simplex[i], simplex[(i+1) % 3])))
                    edges.add(edge)
            edge_lengths.append([np.linalg.norm(hull.points[e[0]] - hull.points[e[1]]) for e in edges])
        edge_lengths = np.concatenate(edge_lengths)
        p_e, e = hist(edge_lengths)
        
        plot_values.append([p_e, e, np.mean(energy)])

        np.savez(
            f'data/tp-{uuid4()}.npz',
            N=N,
            N_steps=N_steps,
            pos=final_pos,
            energies=energy,
            p_e=p_e,
            e=e,
            edge_lengths=edge_lengths,
        )

    cmap = plt.cm.viridis
    energies = np.array([en for p_e, e, en in plot_values])
    e_offset = 1e-3
    norm = LogNorm(e_offset, e_offset + np.max(energies) - np.min(energies))

    for p_e, e, en in plot_values:
        plt.plot(e, p_e, label=fr'$E/E_{{rand}}$={en:.3f}', c=cmap(norm(en - np.min(energies) + e_offset)))
    plt.legend()
    plt.yscale('log')
    plt.xlabel(r'$L$', fontsize=16)
    plt.ylabel(r'$P(L)$', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figures/N-{N}-edge-lengths.png', dpi=600)
    plt.close()