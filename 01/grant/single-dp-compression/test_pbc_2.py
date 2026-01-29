from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
import jaxdem as jd
from jaxdem.forces.deformable_particle import angle_between_normals
import trimesh

jax.config.update("jax_enable_x64", True)

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
    em=jnp.array([1.0]),      # area ("measure") stiffness
    ec=jnp.array([1.0]),      # content (volume) stiffness
    eb=jnp.array([1.0]),      # bending stiffness
    el=jnp.array([1.0]),        # edge-length stiffness
    # gamma=jnp.array([0.0]),   # optional surface tension term
)

# --- state: one deformable body represented by its mesh vertices as collision spheres ---
dts = [1e-3, 5e-3, 1e-2]
tes = []
for dt in dts:
    node_radius = 0.1 * R
    node_mass = 1.0
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

    dp_force, dp_energy = DP.create_force_energy_functions(DP)

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
    ke = jax.vmap(partial(jd.utils.thermal.compute_translational_kinetic_energy))(state_traj)

    import matplotlib.pyplot as plt
    plt.plot(pe)
    plt.plot(ke)
    plt.plot(pe + ke)
    plt.savefig('energies.png')
    plt.close()

    tes.append(np.std(pe + ke) / np.mean(pe + ke))

import matplotlib.pyplot as plt
dts = np.array(dts)
tes = np.array(tes)
x = np.log10(dts)
y = np.log10(tes)
a, b = np.polyfit(x, y, 1)
plt.plot(x, y)
plt.plot(x, a * x + b, label=r'$n=$ ' + str(a))
plt.legend()
plt.savefig('dt-test.png')
plt.close()
