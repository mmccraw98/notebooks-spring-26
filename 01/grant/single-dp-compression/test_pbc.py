from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
import jaxdem as jd
import trimesh

jax.config.update("jax_enable_x64", True)

# IMPORTANT: this matches jaxdem.forces.deformable_particle.angle_between_normals
def angle_between_normals(n1: np.ndarray, n2: np.ndarray) -> np.ndarray:
    y = np.linalg.norm(n1 - n2, axis=-1)
    x = np.linalg.norm(n1 + n2, axis=-1)
    return 2.0 * np.arctan2(y, x)

def deformable_energy_per_particle(pos: jax.Array, state: jd.State, system: jd.System, container: jd.DeformableParticleContainer) -> jax.Array:
    # This mirrors the internal Pe(...) inside DeformableParticleContainer.create_force_function,
    # but returns per-particle energy (so ForceManager/thermal can report it).
    idx_map = (
        jnp.zeros((state.N,), dtype=int)
        .at[state.unique_ID]
        .set(jnp.arange(state.N))
    )

    dim = state.dim
    if dim != 3:
        raise ValueError("This script's deformable_energy_per_particle is implemented for 3D only.")

    def compute_element_properties_3D(simplex: jax.Array):
        r1 = simplex[0]
        r2 = simplex[1] - simplex[0]
        r3 = simplex[2] - simplex[0]
        face_normal = jnp.cross(r2, r3) / 2.0
        partial_vol = jnp.sum(face_normal * r1, axis=-1) / 3.0
        area2 = jnp.sum(face_normal * face_normal, axis=-1)
        area = jnp.where(area2 == 0, 1.0, jnp.sqrt(area2))
        return face_normal / area, area, partial_vol

    def angle_between_normals_jax(n1: jax.Array, n2: jax.Array) -> jax.Array:
        y = jnp.linalg.norm(n1 - n2, axis=-1)
        x = jnp.linalg.norm(n1 + n2, axis=-1)
        return 2.0 * jnp.atan2(y, x)

    K = container.num_bodies
    E_body = jnp.zeros((K,), dtype=float)

    element_normal = None

    # Area ("measure") + Volume ("content") + gamma
    if (container.em is not None) or (container.ec is not None) or (container.gamma is not None):
        current_element_indices = idx_map[container.elements]
        element_normal, element_measure, partial_content = jax.vmap(compute_element_properties_3D)(
            pos[current_element_indices]
        )

        if (container.em is not None) and (container.initial_element_measures is not None) and (container.elements_ID is not None):
            elment_0 = jnp.where(container.initial_element_measures == 0, 1.0, container.initial_element_measures)
            temp_elements = jax.ops.segment_sum(
                jnp.power(element_measure / elment_0 - 1.0, 2),
                container.elements_ID,
                num_segments=K,
            )
            E_body = E_body + 0.5 * container.em * temp_elements

        if (container.ec is not None) and (container.initial_body_contents is not None) and (container.elements_ID is not None):
            content = jax.ops.segment_sum(
                partial_content,
                container.elements_ID,
                num_segments=K,
            )
            content_0 = jnp.where(container.initial_body_contents == 0, 1.0, container.initial_body_contents)
            E_body = E_body + 0.5 * container.ec * jnp.power(content / content_0 - 1.0, 2)

        if (container.gamma is not None) and (container.elements_ID is not None):
            element = jax.ops.segment_sum(
                element_measure,
                container.elements_ID,
                num_segments=K,
            )
            E_body = E_body - container.gamma * element

    # Bending
    if (
        (container.eb is not None)
        and (container.element_adjacency is not None)
        and (container.initial_bending is not None)
        and (container.element_adjacency_ID is not None)
    ):
        if element_normal is None:
            current_element_indices = idx_map[container.elements]
            element_normal, _, _ = jax.vmap(compute_element_properties_3D)(
                pos[current_element_indices]
            )

        angles = jax.vmap(angle_between_normals_jax)(
            element_normal[container.element_adjacency[:, 0]],
            element_normal[container.element_adjacency[:, 1]],
        )
        bending_0 = jnp.where(container.initial_bending == 0, 1.0, container.initial_bending)
        temp_angles = jax.ops.segment_sum(
            jnp.power(angles / bending_0 - 1.0, 2),
            container.element_adjacency_ID,
            num_segments=K,
        )
        E_body = E_body + 0.5 * container.eb * temp_angles

    # Edge-length
    if (
        (container.el is not None)
        and (container.edges is not None)
        and (container.initial_edge_lengths is not None)
        and (container.edges_ID is not None)
    ):
        current_edge_indices = idx_map[container.edges]
        edge_vecs = pos[current_edge_indices[:, 0]] - pos[current_edge_indices[:, 1]]
        edge_lengths = jnp.linalg.norm(edge_vecs, axis=-1)
        edge_0 = jnp.where(container.initial_edge_lengths == 0, 1.0, container.initial_edge_lengths)
        temp_edges = jax.ops.segment_sum(
            jnp.power(edge_lengths / edge_0 - 1.0, 2),
            container.edges_ID,
            num_segments=K,
        )
        E_body = E_body + 0.5 * container.el * temp_edges

    # Distribute per-body energy across nodes belonging to that deformable body
    counts = jnp.bincount(state.deformable_ID, length=K)[state.deformable_ID]
    return E_body[state.deformable_ID] / counts

# --- rest mesh from trimesh (icosphere) ---
R = 2.0
box_size = np.ones(3) * 2.5 * R
subdiv = 2
mesh = trimesh.creation.icosphere(subdivisions=subdiv, radius=R)

V = np.asarray(mesh.vertices, dtype=float)          # (N, 3)
V += box_size / 2                                  # center the particle in [0, box_size]
F = np.asarray(mesh.faces, dtype=np.int32)          # (M, 3)

vel = np.random.normal(loc=0, scale=1e-2, size=V.shape)

# unique wireframe edges (for "length" energy el)
E = np.asarray(mesh.edges_unique, dtype=np.int32)   # (E, 2)

# face adjacency pairs (for "bending" energy eb)
A = np.asarray(mesh.face_adjacency, dtype=np.int32) # (A, 2)

# --- rest bending angles from the *rest* mesh geometry ---
v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
n = np.cross(v1 - v0, v2 - v0)
n /= np.linalg.norm(n, axis=1, keepdims=True)

theta0 = angle_between_normals(n[A[:, 0]], n[A[:, 1]])  # (A,)

# --- deformable particle container (single body => coeffs are length-1 arrays) ---
DP = jd.DeformableParticleContainer.create(
    vertices=jnp.asarray(V),
    elements=jnp.asarray(F),
    edges=jnp.asarray(E),
    element_adjacency=jnp.asarray(A),
    initial_bending=jnp.asarray(theta0),  # <- rest bending from mesh (NOT flat default)
    # em=jnp.array([1.0]),      # area ("measure") stiffness
    # ec=jnp.array([1.0]),      # content (volume) stiffness
    # eb=jnp.array([1.0]),      # bending stiffness
    el=jnp.array([1.0]),        # edge-length stiffness
    # gamma=jnp.array([0.0]),   # optional surface tension term
)

# --- state: one deformable body represented by its mesh vertices as collision spheres ---
node_radius = 0.1 * R
node_mass = 1.0
dt = 1e-2
e_int = 1.0
state = jd.State.create(
    pos=jnp.asarray(V),
    vel=jnp.asarray(vel),
    rad=node_radius * jnp.ones((V.shape[0],), dtype=float),
    mass=node_mass * jnp.ones((V.shape[0])),
    deformable_ID=jnp.zeros((V.shape[0],), dtype=int),  # all nodes belong to body 0
)

# --- system: register deformable force + deformable potential energy ---
mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
matcher = jd.MaterialMatchmaker.create("harmonic")
mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

dp_force = DP.create_force_function(DP)
dp_energy = lambda pos, st, sy: deformable_energy_per_particle(pos, st, sy, DP)

system = jd.System.create(
    state.shape,
    dt=dt,
    domain_type="reflectsphere",
    domain_kw=dict(
        box_size=box_size,
    ),
    force_manager_kw=dict(
        # gravity=jnp.array([0.0, 0.0, -0.01]),
        force_functions=((dp_force, dp_energy),),  # <-- now thermal potential includes deformable energy
    ),
    mat_table=mat_table,
)

# Example step:
# state, system = system.step(state, system, n=10)

n_steps = 10_000
save_stride = 100
n_snapshots = int(n_steps) // int(save_stride)
st, sy, (state_traj, system_traj) = system.trajectory_rollout(
    state, system, n=n_snapshots, stride=int(save_stride)
)

# Total potential (gravity + collider + deformable energy you registered):
pe = jax.vmap(jd.utils.thermal.compute_potential_energy)(state_traj, system_traj)

# Just deformable potential (optional convenience):
pe_dp = jax.vmap(lambda st, sy: jnp.sum(dp_energy(st.pos, st, sy)))(state_traj, system_traj)

ke = jax.vmap(partial(jd.utils.thermal.compute_translational_kinetic_energy))(state_traj)

import matplotlib.pyplot as plt
plt.plot(pe, label="PE")
plt.plot(pe_dp, label="PE_DP")
plt.plot(ke, label="KE")
plt.plot(pe + ke, label="TE")
plt.legend()
plt.savefig("energies.png")
plt.close()

from bump_utils import animate
animate(state_traj, system_traj, "test.gif", id_name="deformable_ID")