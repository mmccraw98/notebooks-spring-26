from utils import create_ga_clumps_2d
import numpy as np
import jaxdem as jd
import jax.numpy as jnp

from shapely.ops import unary_union
from shapely import Point, Polygon

data = np.load('data/ga-dynamics-data-3.npz')

ga_densities = {
    'phi': [],
    'true_phi': [],
    'mu': [],
    'nv': [],
    'alpha': [],
}

N = 1000
phi = 0.6
mass = 1.0
e_int = 1.0
dt = 1e-2

quad_segs = 1e4

for mu_nv_alpha in np.unique(np.column_stack((data['mu'], data['nv'], data['alpha'])), axis=0):
    mu, nv, alpha = mu_nv_alpha
    nv = int(nv)
    state, system = create_ga_clumps_2d(
        N,
        mu,
        nv,
        alpha,
        mass,
        e_int,
        phi,
        dt
    )
    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
    true_phi = 0
    for cid in [0, jnp.max(state.clump_ID)]:
        mask = state.clump_ID == cid
        shape = unary_union([Point(p).buffer(r, quad_segs=quad_segs) for p, r in zip(state.pos[mask], state.rad[mask])] + [Polygon(state.pos[mask])])
        true_phi += (N // 2) * shape.area
    true_phi /= jnp.prod(system.domain.box_size)

    ga_densities['phi'].append(phi)
    ga_densities['true_phi'].append(true_phi)
    ga_densities['mu'].append(mu)
    ga_densities['alpha'].append(alpha)
    ga_densities['nv'].append(nv)

np.savez(
    'data/ga-densities.npz',
    **ga_densities,
)