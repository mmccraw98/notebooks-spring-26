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
box_size = np.ones(3) * 2.2 * R
subdiv = 2
mesh = trimesh.creation.icosphere(subdivisions=subdiv, radius=R)

V = np.asarray(mesh.vertices, dtype=float)          # (N, 3)
V += box_size / 2                                  # center the particle in [0, box_size]
F = np.asarray(mesh.faces, dtype=np.int32)          # (M, 3)

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
    em=jnp.array([1.0]),      # area ("measure") stiffness [x]
    ec=jnp.array([1.0]),      # content (volume) stiffness [x]
    # eb=jnp.array([1.0]),      # bending stiffness [NO]
    el=jnp.array([1.0]),        # edge-length stiffness [x]
    # gamma=jnp.array([0.0]),   # optional surface tension term
)

# --- state: one deformable body represented by its mesh vertices as collision spheres ---
node_rad = 0.2 * R
node_mass = 1.0
dt = 1e-2
e_int = 1.0

sphere_rad = 10 * R
upper_sphere_pos = box_size.copy() / 2
upper_sphere_pos[-1] = box_size[-1] + sphere_rad
lower_sphere_pos = box_size.copy() / 2
lower_sphere_pos[-1] = -sphere_rad

sphere_pos = np.stack([upper_sphere_pos, lower_sphere_pos], axis=0)  # (2,3)

pos = np.concatenate([V, sphere_pos], axis=0)  # (N+2,3)
rad = np.concatenate([node_rad * np.ones((V.shape[0],), float),
                      sphere_rad * np.ones((2,), float)], axis=0)

dp_id = np.concatenate([np.zeros((V.shape[0],), dtype=int),
                        np.array([1, 2], dtype=int)], axis=0)

mass = np.concatenate([node_mass * np.ones((V.shape[0],), float),
                       node_mass * np.ones((2,), float)], axis=0)

fixed = np.zeros_like(rad, dtype=int)
fixed[-2:] = 1

state = jd.State.create(
    pos=jnp.asarray(pos),
    rad=jnp.asarray(rad),
    mass=jnp.asarray(mass),
    deformable_ID=jnp.asarray(dp_id),
    fixed=jnp.asarray(fixed),
)

# --- system: register deformable force + deformable potential energy ---
mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
matcher = jd.MaterialMatchmaker.create("harmonic")
mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

dp_force, dp_energy = DP.create_force_energy_functions(DP)

system = jd.System.create(
    state.shape,
    dt=dt,
    domain_type="free",
    force_model_type="spring",
    linear_integrator_type="linearfire",
    force_manager_kw=dict(
        force_functions=((dp_force, dp_energy),),
    ),
    mat_table=mat_table,
)

pos_hist = []
rad_hist = []
id_hist = []
box_size_hist = []
force_hist = []

sphere_increment = 5e-3  # change sphere z positions each step
sphere_update_mask = np.zeros_like(pos)
sphere_update_mask[-2, -1] = -1
sphere_update_mask[-1, -1] = 1
sphere_update_mask = jnp.asarray(sphere_update_mask) * sphere_increment

from tqdm import tqdm
for i in tqdm(range(50)):
    state, system, steps, final_pe = jd.minimizers.minimize(state, system)
    pos_hist.append(state.pos[:-2])
    rad_hist.append(state.rad[:-2])
    id_hist.append(state.deformable_ID[:-2])
    box_size_hist.append(system.domain.box_size)
    force_hist.append(state.force)
    state.pos_c = state.pos_c + sphere_update_mask

force_hist = np.array(force_hist)
import matplotlib.pyplot as plt
plt.plot(np.linalg.norm(force_hist[-1], axis=-1))
plt.savefig('forces.png')
plt.close()

pos_hist = np.array(pos_hist)
rad_hist = np.array(rad_hist)
id_hist = np.array(id_hist)
box_size_hist = np.array(box_size_hist)
# box_size_hist = -1 * np.ones_like(box_size_hist)
path = 'test.gif'
frames = 100
fps = 15

import subprocess
from pathlib import Path
import h5py
with h5py.File("traj.h5", "w") as f:
    f.create_dataset("pos", data=np.asarray(pos_hist))
    f.create_dataset("rad", data=np.asarray(rad_hist))
    f.create_dataset("ID", data=np.asarray(id_hist))
    f.create_dataset("box_size", data=np.asarray(box_size_hist))

# --- Optional: generate a GIF animation (requires ParaView pvbatch) ---
script_dir = Path(__file__).resolve().parent
run_animation = "/home/mmccraw/dev/analysis/fall-25/12/testing-jaxdem-scripts/animation/run_animation.sh"
subprocess.run(
    [
        str(run_animation),
        "traj.h5",
        path,
        str(frames),   # num_frames (evenly sampled if traj has more)
        "1000",  # base_pixels
        str(fps),    # fps
    ],
    check=True,
)