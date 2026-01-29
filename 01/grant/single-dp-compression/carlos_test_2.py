import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

import jaxdem as jdem
import numpy as np
import trimesh
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from tqdm.auto import tqdm

# --- Simulation Parameters ---
rad = 1000.0
dp_rad = 2.4
dp_v_rad = 0.06
dp_v_mass = 0.02
subdivisions = 3
rate = 2e-3
target_compression_pct = 0.1
num_data_points = 200
dt = 0.0005
damping_coef = 1.0
stiffness_ec = 1000.0
stiffness_em = 1.0
stiffness_el = 1.0
stiffness_eb = 0.1

# --- Setup Mesh and Particles ---
mesh = trimesh.creation.icosphere(radius=dp_rad, subdivisions=subdivisions)
Nv = len(mesh.vertices)
state = jdem.State.create(
    pos=mesh.vertices,
    rad=dp_v_rad * jnp.ones(Nv),
    mass=dp_v_mass * jnp.ones(Nv),
)

DP_container = jdem.DeformableParticleContainer.create(
    vertices=state.pos,
    elements=jnp.array(mesh.faces, dtype=int),
    edges=jnp.array(mesh.edges, dtype=int),
    element_adjacency=jnp.array(mesh.face_adjacency, dtype=int),
    ec=[stiffness_ec],
    em=[stiffness_em],
    el=[stiffness_el],
    eb=[stiffness_eb],
)

# --- Bounds and System Initialization ---
y0_top = rad + dp_rad + dp_v_rad
y0_bot = -rad - dp_rad - dp_v_rad
total_disp = (2 * dp_rad) * target_compression_pct
total_time = total_disp / rate
total_steps = int(total_time / dt)
steps_per_cycle = max(1, int(total_steps / num_data_points))

state_sphere = jdem.State.create(
    pos=[[0, y0_bot, 0], [0, y0_top, 0]],
    vel=[[0, rate / 2, 0], [0, -rate / 2, 0]],
    rad=[rad, rad],
    fixed=[True, True],
)
state = state.merge(state, state_sphere)


def damping_force_fn(
    pos: jax.Array, state: jdem.State, system: jdem.System
) -> Tuple[jax.Array, jax.Array]:
    drag_force = -damping_coef * state.vel
    mask = jnp.where(jnp.arange(state.N) < Nv, 1.0, 0.0)[:, None]
    return drag_force * mask, jnp.zeros_like(state.torque)


system = jdem.System.create(
    state.shape,
    force_manager_kw=dict(
        force_functions=(
            DP_container.create_force_function(DP_container),
            damping_force_fn,
        ),
    ),
    dt=dt,
    mat_table=jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", density=0.27, young=1.0e5, poisson=0.3)],
    ),
)

# --- Data Storage ---
strain_data = []
force_data = []
energy_history: Dict[str, List[float]] = {
    "total": [],
    "E_element": [],
    "E_content": [],
    "E_gamma": [],
    "E_bending": [],
    "E_edge": [],
}

initial_gap = state.pos[-1][1] - state.pos[-2][1]


def record_step(s: jdem.State) -> None:
    # Compression and Force
    current_gap = s.pos[-1][1] - s.pos[-2][1]
    strain_data.append(initial_gap - current_gap)
    avg_f = (jnp.linalg.norm(s.force[-1]) + jnp.linalg.norm(s.force[-2])) / 2.0
    force_data.append(avg_f)

    # Energy
    e_total, e_dict = DP_container.compute_potential_energy(
        s.pos, s, system, DP_container
    )
    energy_history["total"].append(float(e_total))
    for k, v in e_dict.items():
        energy_history[k].append(float(v))


# --- Run Simulation ---
writer = jdem.VTKWriter()
print(f"Simulating {target_compression_pct*100}% compression...")
record_step(state)
writer.save(state, system)

for i in tqdm(range(num_data_points), desc="Simulation Progress", unit="sample"):
    state, system = system.step(state, system, n=steps_per_cycle)
    writer.save(state, system)
    record_step(state)

# --- Plot energy ---
plt.figure(figsize=(10, 6))
for key, values in energy_history.items():
    arr = np.array(values)
    if np.any(np.abs(arr) > 1e-12):
        label = key.replace("_", " ").title()
        plt.plot(strain_data, values, label=label, linewidth=2)

plt.xlabel(r"Displacement")
plt.ylabel("Potential Energy")
plt.yscale("log")
plt.xscale("log")
plt.ylim(1e-10, 1)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("energy_plot.png")
plt.show()
plt.close()

# --- Plot force ---
data_to_save = np.column_stack((strain_data, force_data))
np.savetxt(
    "compression_data.csv",
    data_to_save,
    delimiter=",",
    header="RelativeDisplacement,Force",
    comments="",
)
print("Data saved to compression_data.csv")

strain_arr = np.array(strain_data)
force_arr = np.array(force_data)
valid_mask = (strain_arr > 2e-2) * (force_arr > 0)
slope, intercept = np.polyfit(
    np.log(strain_arr[valid_mask]), np.log(force_arr[valid_mask]), 1
)
plt.figure(figsize=(8, 6))
plt.loglog(
    strain_data,
    force_data,
    marker="o",
    markersize=3,
    linestyle="None",
    label="Data",
)
x_min = np.min(strain_arr[valid_mask])
x_max = np.max(strain_arr[valid_mask])
x_fit = np.geomspace(x_min, x_max, 100)
y_fit = np.exp(intercept) * (x_fit**slope)
plt.loglog(x_fit, y_fit, "r--", linewidth=2, label=f"Fit (slope={slope:.3f})")
plt.text(
    0.05,
    0.9,
    f"Slope: {slope:.3f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)
plt.xlabel(r"Displacement")
plt.ylabel(r"Force")
plt.grid(True, which="both", ls="-", alpha=0.4)
plt.savefig("compression_plot.png")
plt.show()