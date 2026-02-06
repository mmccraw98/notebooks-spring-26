import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import jaxdem as jd
from jaxdem.utils.geometricAsperityCreation import generate_ga_clump_state
import trimesh
from tqdm import tqdm

CENTRAL_ID = 0
TRACER_ID = 1

def quat_from_x_to(d):
    """Unit quaternion (w, x, y, z) rotating [1, 0, 0] to unit vector d.
    
    Uses the half-way quaternion formula:
        q = normalize([1 + dot([1,0,0], d),  cross([1,0,0], d)])
          = normalize([1 + dx,  0,  -dz,  dy])
    """
    q = jnp.array([1.0 + d[0], 0.0, -d[2], d[1]])
    norm = jnp.linalg.norm(q)
    # Antiparallel fallback (d ≈ [-1,0,0]): 180° about y-axis
    return jnp.where(norm < 1e-8, jnp.array([0.0, 0.0, 1.0, 0.0]), q / norm)

def greedy_nearest_neighbor_order(points):
    """Return index order visiting each point once, always jumping to the closest unvisited."""
    N = points.shape[0]
    visited = np.zeros(N, dtype=bool)
    order = np.empty(N, dtype=int)
    order[0] = 0  # start at vertex 0
    visited[0] = True
    for k in range(1, N):
        last = points[order[k - 1]]
        dists = np.linalg.norm(points - last, axis=-1)
        dists[visited] = np.inf
        order[k] = np.argmin(dists)
        visited[order[k]] = True
    return order

def smooth_nearest_neighbor_order(points, alpha=0.5):
    """
    Greedy path that balances proximity and directional consistency.
    Cost = alpha * normalized_distance + (1 - alpha) * (1 - cos(turning_angle))
    alpha=1.0 is pure nearest-neighbor, alpha=0.0 is pure straight-line preference.
    """
    N = points.shape[0]
    visited = np.zeros(N, dtype=bool)
    order = np.empty(N, dtype=int)
    # Start at vertex 0
    order[0] = 0
    visited[0] = True
    # Pick nearest for the second point (no direction yet)
    dists = np.linalg.norm(points - points[0], axis=-1)
    dists[0] = np.inf
    order[1] = np.argmin(dists)
    visited[order[1]] = True
    for k in range(2, N):
        prev = points[order[k - 2]]
        curr = points[order[k - 1]]
        # Current travel direction
        travel_dir = curr - prev
        travel_norm = np.linalg.norm(travel_dir)
        if travel_norm > 1e-12:
            travel_dir /= travel_norm
        else:
            travel_dir = np.zeros_like(travel_dir)
        # Candidate vectors
        deltas = points - curr                                      # (N, 3)
        dists = np.linalg.norm(deltas, axis=-1)                    # (N,)
        # Normalized candidate directions
        safe_dists = np.where(dists > 1e-12, dists, 1.0)
        candidate_dirs = deltas / safe_dists[:, None]               # (N, 3)
        # Cosine of turning angle (1 = straight ahead, -1 = U-turn)
        cos_turn = candidate_dirs @ travel_dir                      # (N,)
        # Normalize distance to [0, 1] range
        max_dist = np.max(dists[~visited]) if np.any(~visited) else 1.0
        norm_dist = dists / max(max_dist, 1e-12)
        # Cost: low = good
        # Turn cost: 0 when straight ahead, 1 when U-turn
        turn_cost = (1.0 - cos_turn) / 2.0                         # [0, 1]
        cost = alpha * norm_dist + (1.0 - alpha) * turn_cost
        cost[visited] = np.inf
        order[k] = np.argmin(cost)
        visited[order[k]] = True

    return order

@jax.jit
def find_contact_point(state, system, max_separation, min_separation, tracer_mask, offsets, pe_target, separation_tolerance):
    # the jax code below is equivalent to this:
    # while ((separation_high - separation_low) > separation_tolerance):
    #     # check if the tracer is too far or too close based on the potential energy
    #     pe = jnp.sum(system.collider.compute_potential_energy(state, system))
    #     too_far = (pe < pe_target)
    #     too_close = (1 - too_far)
    #     # if too far, separation_high = separation
    #     separation_high = separation * (too_far) + separation_high * (too_close)
    #     # if too close, separation_low = separation
    #     separation_low = separation * (too_close) + separation_low * (too_far)

    #     # move the tracer according to the new mid-point
    #     new_separation = (separation_high + separation_low) / 2
    #     delta = separation - new_separation
    #     state.pos_c += delta * tracer_mask * direction  # shift along direction vector

    #     # update the separation
    #     separation = new_separation
    
    separation_high = max_separation
    separation_low = min_separation

    # calculate center of mass distance between each clump
    pos_c = state.pos_c[offsets]
    r_ij = pos_c[0] - pos_c[1]
    separation = jnp.linalg.norm(r_ij)
    direction = r_ij / separation

    # use this as the condition to continue running the while loop
    # while the brackets are too far away, continue running the while loop body
    def loop_condition(values):
        separation_high, separation_low, _, _ = values
        return (separation_high - separation_low) > separation_tolerance

    # run this function as the body of the while loop
    # it grows / shrinks the separation between the particles based on the potential energy
    def loop_body(values):
        separation_high, separation_low, separation, state = values
        # check if the tracer is too far or too close based on the potential energy
        pe = jnp.sum(system.collider.compute_potential_energy(state, system))
        too_far = (pe < pe_target)
        too_close = (1 - too_far)
        # if too far, separation_high = separation
        separation_high = separation * (too_far) + separation_high * (too_close)
        # if too close, separation_low = separation
        separation_low = separation * (too_close) + separation_low * (too_far)

        # move the tracer according to the new mid-point
        new_separation = (separation_high + separation_low) / 2
        delta = separation - new_separation
        state.pos_c += delta * tracer_mask[:, None] * direction  # shift along direction vector

        # update the separation
        separation = new_separation
        return separation_high, separation_low, separation, state

    # initial values for the while loop
    initial_values = (
        jnp.asarray(separation_high),
        jnp.asarray(separation_low),
        jnp.asarray(separation),
        state,
    )

    final_values = jax.lax.while_loop(
        loop_condition,
        loop_body,
        initial_values,
    )

    separation_high, separation_low, separation, state = final_values
    return separation, state


particle_radii = np.array([0.5, 0.5])
vertex_counts = np.array([2, 2])
asperity_radius = 0.2
core_type = "true"
mesh_type = "ico"

# create inputs
state, box_size = generate_ga_clump_state(
    particle_radii=particle_radii,
    vertex_counts=vertex_counts,
    phi=0.001,
    dim=3,
    asperity_radius=asperity_radius,
    core_type=core_type,
    mesh_type=mesh_type,
    use_random_orientations=False,
)

box_size = jnp.ones_like(box_size) * (jnp.sum(particle_radii) * 3)

mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
matcher = jd.MaterialMatchmaker.create("harmonic")
mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
system = jd.System.create(
    state_shape=state.shape,
    dt=1e-2,
    linear_integrator_type="",
    rotation_integrator_type="",
    domain_type="periodic",
    force_model_type="spring",
    collider_type="naive",
    mat_table=mat_table,
    domain_kw=dict(
        box_size=box_size,
        anchor=jnp.zeros_like(box_size),
    ),
)

# TODO: calculate friction coef
# TODO: vmap over orientations in chunks
# TODO: assign different initial tracer orientations
# TODO: record the tracer angles

# initial relative position of the tracer
pe_target = 1e-16
separation_tolerance = 1e-10

# get the first id in each clump
cids, offsets = jnp.unique(state.clump_ID, return_index=True)

# set guessed bounds
scale = 1.1
max_separation = np.sum(particle_radii) * scale
min_separation = (max_separation - 2 * asperity_radius) / scale
med_separation = (max_separation - min_separation) / 2

# move particle 0 to center, particle 1 to midpoint
tracer_mask = (state.clump_ID == TRACER_ID)
core_mask = (state.clump_ID == CENTRAL_ID)
state.pos_c = box_size / 2  # center both particles
state.pos_c += (med_separation) * tracer_mask[:, None] * jnp.array([1.0, 0.0, 0.0])  # offset the tracer

# zero out the orientations
state.q.w = 1.0
state.q.xyz *= 0.0

# Precompute all quaternions and positions
sphere_mesh = trimesh.creation.icosphere(
    subdivisions=2,
    radius=med_separation
)
directions = jnp.array(sphere_mesh.vertices)
quats = jax.vmap(quat_from_x_to)(directions)
tracer_positions = directions * med_separation

pos = []
rad = []
pid = []
bs = []

# apply the i-th rotation
# i = 0
# for i in tqdm(range(directions.shape[0])):
# order = greedy_nearest_neighbor_order(np.array(directions))
order = smooth_nearest_neighbor_order(np.array(directions))
for i in tqdm(order):
    state.pos_c = system.domain.box_size / 2  # center both
    state.pos_c += tracer_positions[i] * tracer_mask[:, None]  # adjust the tracer position
    state.q.w = jnp.where(tracer_mask[:, None], quats[i, 0:1], 1.0)  # rotate
    state.q.xyz = jnp.where(tracer_mask[:, None], quats[i, 1:4], 0.0)

    # find minimal separation point
    separation, state = find_contact_point(state, system, max_separation, min_separation, tracer_mask, offsets, pe_target, separation_tolerance)
    state, system = system.collider.compute_force(state, system)

    pos_c = state.pos_c[offsets]
    r_ij = pos_c[CENTRAL_ID] - pos_c[TRACER_ID]
    separation = jnp.linalg.norm(r_ij)
    direction = r_ij / separation
    force = jnp.sum(state.force[tracer_mask], axis=0)
    force_n_mag = jnp.sum(force * direction)
    force_t_mag = jnp.linalg.norm(force - force_n_mag * direction)
    mu = jnp.abs(force_t_mag / force_n_mag)
    print(separation, mu)
    pos.append(state.pos)
    rad.append(state.rad)
    pid.append(state.clump_ID)
    bs.append(system.domain.box_size)

# from anim_utils import render, animate
# # render(state, system, 'test.png')
# animate(
#     np.array(pos),
#     np.array(rad),
#     np.array(pid),
#     np.array(bs),
#     'anim.gif',
# )
