import jax

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

import jax.numpy as jnp
import jaxdem as jdem
from jaxdem import utils
import math
from typing import Tuple, List
import trimesh
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


def save_data() -> None:
    current_force1 = jnp.linalg.norm(state.force[-1]).item()
    current_force2 = jnp.linalg.norm(state.force[-2]).item()
    current_force = (current_force1 + current_force2) / 2

    current_gap = state.pos[-1][1] - state.pos[-2][1]
    displacement = initial_gap - current_gap
    rel_disp = displacement  # / (2 * dp_rad)
    dim_force = current_force  # / (stiffness_ec * dp_rad)
    strain_data.append(rel_disp)
    dim_force_data.append(dim_force)

    energy, aux_data = DP_container.compute_potential_energy(
        state.pos, state, system, DP_container
    )

    total_energy.append(energy)
    element_energy.append(aux_data["E_element"])
    content_energy.append(aux_data["E_content"])


def damping_force_fn(pos, state, system):
    drag_force = -damping_coef * state.vel
    particle_indices = jnp.arange(state.N)
    mask = jnp.where(particle_indices < Nv, 1.0, 0.0)
    mask = mask[:, None]
    return drag_force * mask, jnp.zeros_like(state.torque)


# --- Control Parameters ---
rad = 1000.0
dp_rad = 2.4
dp_v_rad = 0.06
dp_v_mass = 0.04
subdivisions = 3

rate = 1e-3
target_compression_pct = 0.1
num_data_points = 200
dt = 0.0005
damping_coef = 0.01

stiffness_ec = 100000.0
stiffness_em = 0.1

mat_table = (
    jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", density=0.27, young=2.0e4, poisson=0.3)],
    ),
)

mesh = trimesh.creation.icosphere(radius=dp_rad, subdivisions=subdivisions)
vertices = mesh.vertices
faces = mesh.faces
Nv = len(vertices)

state = jdem.State.create(
    pos=vertices,
    rad=dp_v_rad * jnp.ones(Nv),
    mass=dp_v_mass * jnp.ones(Nv),
    # mat_table=mat_table,
)

DP_container = jdem.DeformableParticleContainer.create(
    vertices=state.pos,
    elements=jnp.array(faces, dtype=int),
    ec=[stiffness_ec],
    em=[stiffness_em],
)

y0_top = rad + dp_rad + dp_v_rad
y0_bot = -rad - dp_rad - dp_v_rad
total_disp = (2 * dp_rad) * target_compression_pct
total_time = total_disp / rate
total_steps = int(total_time / dt)
steps_per_cycle = max(1, int(total_steps / num_data_points))

state_sphere = jdem.State.create(
    pos=[
        [0, y0_bot, 0],
        [0, y0_top, 0],
    ],
    vel=[
        [0, rate / 2, 0],  # Bottom moves Up
        [0, -rate / 2, 0],  # Top moves Down
    ],
    rad=[rad, rad],
    fixed=[True, True],
    # mat_table=mat_table,
)
state = state.merge(state, state_sphere)
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

strain_data = []
dim_force_data = []
initial_gap = state.pos[-1][1] - state.pos[-2][1]

total_energy = []
element_energy = []
content_energy = []

writer = jdem.VTKWriter()
writer.save(state, system)
save_data()

print(
    f"Simulating {target_compression_pct*100}% compression over {num_data_points} data points..."
)

for i in range(num_data_points):
    print(f"Step {i+1}/{num_data_points}")
    state, system = system.step(state, system, n=steps_per_cycle)
    writer.save(state, system)
    save_data()


# --- Analysis and Saving ---

# 1. Save Data to CSV
data_to_save = np.column_stack((strain_data, dim_force_data))
np.savetxt(
    "compression_data.csv",
    data_to_save,
    delimiter=",",
    header="RelativeDisplacement,Force",
    comments="",
)
print("Data saved to compression_data.csv")

strain_arr = np.array(strain_data)
force_arr = np.array(dim_force_data)

valid_mask = (strain_arr > 0) & (force_arr > 0)
slope, intercept = np.polyfit(
    np.log(strain_arr[valid_mask]), np.log(force_arr[valid_mask]), 1
)

print(force_arr)

plt.figure(figsize=(8, 6))
plt.loglog(
    strain_data,
    dim_force_data,
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

# 4. Add Slope to Plot
plt.text(
    0.05,
    0.9,
    f"Slope: {slope:.3f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

plt.xlabel(r"Relative Displacement ($d / R_{dp}$)")
plt.ylabel(r"Force ($F$ or Dimensionless)")
plt.grid(True, which="both", ls="-", alpha=0.4)

# 5. Save Plot
plt.savefig("compression_plot.png")
print("Plot saved to compression_plot.png")

plt.show()
plt.close()

plt.figure(figsize=(8, 6))
plt.loglog(strain_data, total_energy, label="total energy")
plt.loglog(strain_data, element_energy, label="element energy")
plt.loglog(strain_data, content_energy, label="content energy")
plt.legend()
plt.xlabel(r"Relative Displacement ($d / R_{dp}$)")
plt.ylabel(r"Energy")
plt.savefig("energy.png")
plt.show()