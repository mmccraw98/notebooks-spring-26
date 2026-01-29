from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
import jaxdem as jd
from jaxdem.forces.deformable_particle import angle_between_normals
import trimesh

jax.config.update("jax_enable_x64", True)

def make_halfspace_harmonic_plate(
    *,
    k: float,                 # your e_plate
    plate: float,             # z_plate (or x/y plate depending on axis)
    axis: str | int = "z",    # "x","y","z" or 0,1,2
    side: str = "ge",         # "ge" => coord >= plate, "le" => coord <= plate
    respect_fixed: bool = True,
    species_id: int | None = None,  # optional: only apply to a species
):
    axis_map = {"x": 0, "y": 1, "z": 2}
    ax = axis_map.get(axis, axis)
    if side not in ("ge", "le"):
        raise ValueError("side must be 'ge' or 'le'")

    def _mask(coord, state):
        m = (coord >= plate) if side == "ge" else (coord <= plate)
        if respect_fixed:
            m = m & (~state.fixed)
        if species_id is not None:
            m = m & (state.species_id == species_id)
        return m

    def energy_fn(pos, state, system):
        coord = pos[..., ax]          # (..., N)
        delta = coord - plate
        m = _mask(coord, state).astype(pos.dtype)
        return 0.5 * k * jnp.square(delta) * m   # (..., N)

    def force_fn(pos, state, system):
        coord = pos[..., ax]
        delta = coord - plate
        m = _mask(coord, state).astype(pos.dtype)

        f = jnp.zeros_like(pos)       # (..., N, dim)
        f = f.at[..., ax].set(-k * delta * m)   # force points toward the plate
        tau = jnp.zeros_like(state.torque)      # no intrinsic torque about sphere center
        return f, tau

    return force_fn, energy_fn

# --- rest mesh from trimesh (icosphere) ---
R = 2.0
box_size = np.ones(3) * 2.01 * R
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


pos = np.array(V)
rad = node_rad * np.ones((V.shape[0],), float)
dp_id = np.zeros((V.shape[0],), dtype=int)
mass = node_mass * np.ones((V.shape[0],), float)

state = jd.State.create(
    pos=jnp.asarray(pos),
    rad=jnp.asarray(rad),
    mass=jnp.asarray(mass),
    deformable_ID=jnp.asarray(dp_id),
)

# --- system: register deformable force + deformable potential energy ---
mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
matcher = jd.MaterialMatchmaker.create("harmonic")
mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

dp_force, dp_energy = DP.create_force_energy_functions(DP)

plate_pos_upper = box_size[-1]
plate_pos_lower = 0.0

pos_hist = []
rad_hist = []
id_hist = []
box_size_hist = []
force_hist = []
pe_hist = []
plate_distance_hist = []

plate_increment = 1e-2  # change plate z positions each step

data_root = '/home/mmccraw/dev/data/26-01-01/grant/dp-compressions/test-data'
import os
if not os.path.exists(data_root):
    os.makedirs(data_root)

from tqdm import tqdm
for i in tqdm(range(50)):
    plate_pos_upper -= plate_increment
    plate_force_upper, plate_energy_upper = make_halfspace_harmonic_plate(
        k=1e4,
        plate=plate_pos_upper,
        axis="z",
        side="ge",          # only penalize z >= plate_pos_upper
        species_id=None,
    )

    plate_pos_lower += plate_increment
    plate_force_lower, plate_energy_lower = make_halfspace_harmonic_plate(
        k=1e4,
        plate=plate_pos_lower,
        axis="z",
        side="le",          # only penalize z <= plate_pos_lower
        species_id=None,
    )

    system = jd.System.create(
        state.shape,
        dt=dt,
        domain_type="free",
        force_model_type="spring",
        linear_integrator_type="linearfire",
        force_manager_kw=dict(
            force_functions=(
                (dp_force, dp_energy),
                (plate_force_upper, plate_energy_upper, False),
                (plate_force_lower, plate_energy_lower, False),
            ),
        ),
        mat_table=mat_table,
    )

    state, system, steps, final_pe = jd.minimizers.minimize(state, system)
    pos_hist.append(state.pos)
    rad_hist.append(state.rad)
    id_hist.append(state.deformable_ID)
    box_size_hist.append(system.domain.box_size)
    force_hist.append(state.force)
    pe_hist.append(final_pe)
    plate_distance_hist.append(plate_pos_upper - plate_pos_lower)

    jd.utils.h5.save(state, os.path.join(data_root, f'state_{i}.h5'))
    jd.utils.h5.save(system, os.path.join(data_root, f'system_{i}.h5'))
    

plate_distance_hist = np.array(plate_distance_hist)
pe_hist = np.array(pe_hist)
import matplotlib.pyplot as plt
plt.plot(plate_distance_hist, pe_hist)
plt.yscale('log')
plt.savefig('compression.png')
plt.close()

np.save('plate_distance_hist.npy', plate_distance_hist)
np.save('pe_hist.npy', pe_hist)

# force_hist = np.array(force_hist)
# import matplotlib.pyplot as plt
# plt.plot(np.linalg.norm(force_hist[-1], axis=-1))
# plt.savefig('forces.png')
# plt.close()

pos_hist = np.array(pos_hist)
rad_hist = np.array(rad_hist)
id_hist = np.array(id_hist)
box_size_hist = np.array(box_size_hist)
# box_size_hist = -1 * np.ones_like(box_size_hist)  # to ignore box size when rendering
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