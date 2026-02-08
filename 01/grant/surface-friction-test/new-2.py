import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import jaxdem as jd
from jaxdem.utils.geometricAsperityCreation import generate_ga_clump_state
import trimesh
from tqdm import tqdm

# TODO: add docstrings and type hints to all functions

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

def create_state_and_system(particle_radii, vertex_counts, asperity_radius, core_type, mesh_type):
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

    return state, system, box_size

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


nv = 100
asperity_radius = 0.125

particle_radii = np.array([0.5, 0.5])
vertex_counts = np.array([nv, nv])
core_type = "true"
mesh_type = "ico"

# create inputs
state, system, box_size = create_state_and_system(
    particle_radii=particle_radii,
    vertex_counts=vertex_counts,
    asperity_radius=asperity_radius,
    core_type=core_type,
    mesh_type=mesh_type,
)

# TODO: vmap over orientations in chunks
# TODO: assign different initial tracer orientations
# TODO: record the tracer angles

# initial relative position of the tracer
pe_target = 1e-16
separation_tolerance = 1e-10
exterior_subdivisions = 5

batch_size = 10
chunk_size = 100

# set guessed bounds
scale = 1.1
max_separation = np.sum(particle_radii) * scale
min_separation = (max_separation - 2 * asperity_radius) / scale
med_separation = (max_separation - min_separation) / 2

# Precompute all quaternions and positions
sphere_mesh = trimesh.creation.icosphere(
    subdivisions=exterior_subdivisions,
    radius=med_separation
)

# get the first id in each clump
cids, offsets = jnp.unique(state.clump_ID, return_index=True)

# move particle 0 to center, particle 1 to midpoint
tracer_mask = (state.clump_ID == TRACER_ID)
core_mask = (state.clump_ID == CENTRAL_ID)
state.pos_c = box_size / 2  # center both particles
state.pos_c += (med_separation) * tracer_mask[:, None] * jnp.array([1.0, 0.0, 0.0])  # offset the tracer

# zero out the orientations
state.q.w = 1.0
state.q.xyz *= 0.0


directions = jnp.array(sphere_mesh.vertices)
quats = jax.vmap(quat_from_x_to)(directions)
tracer_positions = directions * med_separation

pos = []
rad = []
pid = []
bs = []
mu = []
r = []

# apply the i-th rotation
i = 0
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
# pos.append(state.pos)
# rad.append(state.rad)
# pid.append(state.clump_ID)
# bs.append(system.domain.box_size)
# mu.append(jnp.abs(force_t_mag / force_n_mag))
# r.append(separation)

