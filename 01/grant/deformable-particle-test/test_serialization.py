from bump_utils import create_clumps, render, animate
import jaxdem as jd
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import math
from typing import Tuple, List

def circle(r: float = 1.0, n: int = 40) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
    vertices = []
    edges = []
    for i in range(n):
        theta = 2.0 * math.pi * i / n
        vertices.append((r * math.cos(theta), r * math.sin(theta)))
    for i in range(n):
        edges.append((i, (i + 1) % n))
    return vertices, edges

# --- config ---
K = 2                 # number of deformable particles
R = 0.5               # deformable particle radius (geometry radius)
n_nodes = 20          # vertices per particle
node_rad = 0.2       # radius of the "spheres" (nodes) used for contacts

box_size = jnp.array([4.0, 4.0])
anchor   = jnp.array([0.0, 0.0])

# evenly-spaced centers along x, centered in y
centers = jnp.stack([
    anchor + jnp.array([(i + 0.5) * box_size[0] / K, 0.5 * box_size[1]])
    for i in range(K)
], axis=0)  # (K, 2)

# --- base mesh ---
verts_local, edges_local = circle(r=R, n=n_nodes)
verts_local = jnp.array(verts_local, dtype=float)   # (V, 2)
edges_local = jnp.array(edges_local, dtype=int)     # (M, 2)
V, M = verts_local.shape[0], edges_local.shape[0]

# --- positions: K copies of the vertices, shifted to centers ---
pos = (verts_local[None, :, :] + centers[:, None, :]).reshape((K * V, 2))

# --- elements + elements_ID: K copies of edges with index offsets ---
offsets = (jnp.arange(K) * V)[:, None, None]              # (K, 1, 1)
elements = (edges_local[None, :, :] + offsets).reshape((K * M, 2))
elements_ID = jnp.repeat(jnp.arange(K, dtype=int), M)     # (K*M,)

# --- per-body random velocities, broadcast to all vertices in that body ---
key = jax.random.PRNGKey(0)
v_body = 5e-2 * jax.random.uniform(key, (K, 2), minval=-1.0, maxval=1.0)  # (K, 2)
body_id_per_vertex = jnp.repeat(jnp.arange(K, dtype=int), V)             # (K*V,)
vel = v_body[body_id_per_vertex]                                        # (K*V, 2)

adj_local = jnp.stack(
    [jnp.arange(M, dtype=int), (jnp.arange(M, dtype=int) + 1) % M],
    axis=1
)  # (M, 2) in local segment-index space

# Offset adjacency per body (because elements are concatenated by body)
adj_offsets = (jnp.arange(K, dtype=int) * M)[:, None, None]     # (K,1,1)
element_adjacency = (adj_local[None, :, :] + adj_offsets).reshape((K * M, 2))
element_adjacency_ID = jnp.repeat(jnp.arange(K, dtype=int), M)  # (K*M,)

# --- state ---
state = jd.State.create(
    pos=pos,
    vel=vel,
    rad=node_rad * jnp.ones((K * V,), dtype=float),
    deformable_ID=body_id_per_vertex,
)

# --- deformable container (3 bodies) ---
dp = jd.DeformableParticleContainer.create(
    vertices=state.pos,
    elements=elements,
    elements_ID=elements_ID,
    ec=jnp.array([1000.0] * K),
    em=jnp.array([1.0] * K),
    eb=jnp.array([100.0] * K),
    element_adjacency=element_adjacency,
    element_adjacency_ID=element_adjacency_ID,
)

system = jd.System.create(
    state.shape,
    linear_integrator_type="verlet",
    rotation_integrator_type="",
    domain_type="periodic",
    domain_kw=dict(box_size=box_size, anchor=anchor),
    force_manager_kw=dict(
        force_functions=(dp.create_force_function(dp),),
    ),
)

jd.utils.h5.save(state, 'state_test.h5')
jd.utils.h5.save(system, 'system_test.h5')
jd.utils.h5.save(dp, 'dp_test.h5')

dp = jd.utils.h5.load('dp_test.h5')
state = jd.utils.h5.load('state_test.h5')
system = jd.utils.h5.load('system_test.h5')
system.force_manager.force_functions = (dp.create_force_function(dp),)
