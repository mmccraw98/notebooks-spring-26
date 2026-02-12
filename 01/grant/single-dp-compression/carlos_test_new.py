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

rad = 1000.0
dp_rad = 2.4
dp_v_rad = 0.06
dp_v_mass = 0.02
subdivisions = 3

target_compression_pct = 0.2
num_data_points = 400
learning_rate = 1e-7
n_relax_steps = 1_000_000

stiffness_ec = [4000.0]
stiffness_em = [1.0]
stiffness_el = None  # [stiffness_el]
stiffness_eb = None  # [0.1]

total_disp = 2 * dp_rad * target_compression_pct
rate = total_disp / num_data_points
dir = jnp.asarray([0, 1, 0], dtype=float)

displacement = []
force = []

mesh = trimesh.creation.icosphere(radius=dp_rad, subdivisions=subdivisions)
state = jdem.State.create(
    pos=mesh.vertices,
    rad=dp_v_rad * jnp.ones(len(mesh.vertices)),
    mass=dp_v_mass * jnp.ones(len(mesh.vertices)),
)

DP_container = jdem.DeformableParticleContainer.create(
    vertices=state.pos,
    elements=jnp.array(mesh.faces, dtype=int),
    edges=jnp.array(mesh.edges, dtype=int),
    element_adjacency=jnp.array(mesh.face_adjacency, dtype=int),
    ec=stiffness_ec,
    em=stiffness_em,
    el=stiffness_el,
    eb=stiffness_eb,
)

y0_top = rad + dp_rad
y0_bot = -rad - dp_rad
state_sphere = jdem.State.create(
    pos=[[0, y0_bot, 0], [0, y0_top, 0]],
    rad=[rad, rad],
    fixed=[True, True],
)
state = state.merge(state, state_sphere)
d0 = jnp.linalg.norm(state.pos[-2] - state.pos[-1])

system = jdem.System.create(
    state.shape,
    dt=1e-2,
    linear_integrator_type="linearfire",
    rotation_integrator_type="",
    # linear_integrator_kw=dict(learning_rate=learning_rate),
    force_manager_kw=dict(
        force_functions=(DP_container.create_force_function(DP_container),),
    ),
    mat_table=jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", density=0.27, young=1.0e5, poisson=0.3)],
    ),
)

# state, system = system.step(state, system, n=n_relax_steps)
writer = jdem.VTKWriter()
writer.save(state, system)
displacement.append(d0 - jnp.linalg.norm(state.pos[-2] - state.pos[-1]))
force.append(
    0.5 * (jnp.linalg.norm(state.force[-2]) + jnp.linalg.norm(state.force[-1]))
)

for i in tqdm(range(1, num_data_points + 1), desc="Simulation Progress", unit="sample"):
    # Bottom sphere
    state.pos_c = state.pos_c.at[-2].set(state.pos[-2] + dir * rate / 2)
    # Top sphere
    state.pos_c = state.pos_c.at[-1].set(state.pos[-1] - dir * rate / 2)

    # state, system = system.step(state, system, n=n_relax_steps)
    state, system, steps, pe = jdem.minimizers.minimize(
        state,
        system,
        max_steps=n_relax_steps,
        pe_tol=1e-16,
        pe_diff_tol=1e-16,
        initialize=True,
    )
    writer.save(state, system)

    displacement.append(d0 - jnp.linalg.norm(state.pos[-2] - state.pos[-1]))
    force.append(
        0.5 * (jnp.linalg.norm(state.force[-2]) + jnp.linalg.norm(state.force[-1]))
    )

print(displacement)
print(force)

# Convert to numpy arrays for fitting
disp_np = np.array(displacement)
force_np = np.array(force)

# Remove zeros or negative values for log-log fit
mask = (disp_np > 1e-1) & (force_np > 0)
disp_fit = disp_np[mask]
force_fit = force_np[mask]

# Linear fit in log-log space
log_disp = np.log(disp_fit)
log_force = np.log(force_fit)
slope, intercept = np.polyfit(log_disp, log_force, 1)

# Plot data and fit
plt.loglog(displacement, force, label="Data", marker="o", linestyle="")
plt.loglog(
    disp_fit, np.exp(intercept) * disp_fit**slope, "--", label=f"Fit: slope={slope:.2f}"
)
plt.xlabel("Displacement")
plt.ylabel("Force")
plt.grid()
plt.legend()
plt.show()

print(f"Log-log fit slope: {slope:.4f}")