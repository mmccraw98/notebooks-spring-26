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
    k: float,
    plate: float,
    axis: str | int = "z",
    side: str = "ge",
    respect_fixed: bool = True,
    species_id: int | None = None,
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
        coord = pos[..., ax]
        delta = coord - plate
        m = _mask(coord, state).astype(pos.dtype)
        return 0.5 * k * jnp.square(delta) * m

    def force_fn(pos, state, system):
        coord = pos[..., ax]
        delta = coord - plate
        m = _mask(coord, state).astype(pos.dtype)
        f = jnp.zeros_like(pos)
        f = f.at[..., ax].set(-k * delta * m)
        tau = jnp.zeros_like(state.torque)
        return f, tau

    return force_fn, energy_fn

R = 2.0
box_size = np.ones(3) * 2.01 * R
box_size[:2] *= 2.0
subdiv = 2
mesh = trimesh.creation.icosphere(subdivisions=subdiv, radius=R)

V = np.asarray(mesh.vertices, dtype=float)
V += box_size / 2
F = np.asarray(mesh.faces, dtype=np.int32)

E = np.asarray(mesh.edges_unique, dtype=np.int32)

A = np.asarray(mesh.face_adjacency, dtype=np.int32)

v0, v1, v2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
n = np.cross(v1 - v0, v2 - v0)
n /= np.linalg.norm(n, axis=1, keepdims=True)

theta0 = angle_between_normals(n[A[:, 0]], n[A[:, 1]])

DP = jd.DeformableParticleContainer.create(
    vertices=jnp.asarray(V),
    elements=jnp.asarray(F),
    edges=jnp.asarray(E),
    element_adjacency=jnp.asarray(A),
    initial_bending=jnp.asarray(theta0),
    em=jnp.array([1.0]),
    ec=jnp.array([1.0]),
    # eb=jnp.array([1.0]),
    el=jnp.array([1.0]),
    # gamma=jnp.array([0.0]),
)

node_rad = 0.2 * R
node_mass = 1.0
dt = 1e-3
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

plate_increment = 3e-2  # change plate z positions each step

data_root = '/home/mmccraw/dev/data/26-01-01/grant/dp-compressions/test-data'
import os
if not os.path.exists(data_root):
    os.makedirs(data_root)

from tqdm import tqdm
N_steps = 50
for direction in [1, -1]:
    for i in tqdm(range(N_steps)):
        plate_pos_upper -= plate_increment * direction
        plate_force_upper, plate_energy_upper = make_halfspace_harmonic_plate(
            k=1e4,
            plate=plate_pos_upper,
            axis="z",
            side="ge",  # only penalize z >= plate_pos_upper
            species_id=None,
        )

        plate_pos_lower += plate_increment * direction
        plate_force_lower, plate_energy_lower = make_halfspace_harmonic_plate(
            k=1e4,
            plate=plate_pos_lower,
            axis="z",
            side="le",  # only penalize z <= plate_pos_lower
            species_id=None,
        )

        system = jd.System.create(
            state.shape,
            dt=dt,
            collider_type="",
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
        # box_size_hist.append(system.domain.box_size)
        box_size_hist.append(box_size)
        force_hist.append(state.force)
        pe_hist.append(final_pe)
        plate_distance_hist.append(plate_pos_upper - plate_pos_lower)

        jd.utils.h5.save(state, os.path.join(data_root, f'state_{i}_{direction}.h5'))
        jd.utils.h5.save(system, os.path.join(data_root, f'system_{i}_{direction}.h5'))
    

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