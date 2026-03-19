import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os
import numpy as np
from tqdm import tqdm

from dataclasses import replace

from jaxdem_scripts.ga_utils import create_bidisperse_ga_dps_2d
from jaxdem_scripts.compression_loop_dp import run_1, JobConfig as JCFG
from jaxdem.integrators import Langevin
from jaxdem.colliders import Naive

e_b = 1e-1
# e_m = 1e-1  # length
e_l = 1e1  # length
e_c = 1e2  # area
tau_s = None

# TAU_S IS SET TO NONE IF E_L IS NOT DEFINED!  MAKES SENSE THAT YOU NEED E_L HERE, BUT ISN'T IT NOT NORMALIZED, UNLIKE E_M?

# THE PARTICLE PLASTICITY SEEMS TO DAMP OUT THE VERTEX VELOCITIES
# THEN, WHEN THERMOSTATING THE COMs, THE PARTICLES EVENTUALLY BECOME PSEUDO-RIGID BODIES (NO VERTEX MOTION)

# ACTUALLY, IT LOOKS LIKE ALL THERMOSTATS TESTED HERE (COM AND VERTEX ONLY) ALL EVENTUALLY TURN INTO RIGID BODIES
# PROBABLY NEED TO USE A LANGEVIN THERMOSTAT OR SOMETHING THAT ADDS NOISE

# NEED TO TEST LOADING AND RESUMING USING THE NEW DATA TYPES

size_ratios=(1.0, 1.0)

N_dps = 10
mu_eff = 0.1
min_nv = 30
phi = 0.8
aspect_ratio = 1.0
dp_mass = 1.0
dt = 1e-4
e_int = 10.0
can_rotate = False
subtract_drift = True
temp = 1e-3

cfg = JCFG(
    seed=np.random.randint(0, 1e9),
    delta_phi=1e-2,
    target_temperature=temp,
    n_steps=1_000_00,
    min_save_decade=1000,
)

data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-plastic-fragilitiy/version-1/'

state, system = create_bidisperse_ga_dps_2d(
    N_dps,
    mu_eff,
    min_nv,
    phi,
    aspect_ratio,
    dp_mass,
    dt,
    e_int,
    e_c=e_c,
    e_b=e_b,
    e_l=e_l,
    e_m=None,
    e_gamma=None,
    tau_s=tau_s,
    size_ratios=size_ratios,
)

state = jd.utils.thermal.set_temperature(
    state,
    temp,
    can_rotate,
    subtract_drift,
    np.random.randint(0, 1e9)
)

system = replace(
    system,
    linear_integrator=Langevin(
        gamma=jnp.asarray(1.0),
        temperature=jnp.asarray(temp),
        k_B=jnp.asarray(1.0)
    ),
    collider=Naive(),
)

length = 1000
phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
while phi < 0.95:
    state, system = jd.utils.packingUtils.scale_to_packing_fraction(state, system, phi + 1e-2)
    state, system = system.step(state, system, n=length // 2)
    state = jd.utils.thermal.scale_to_temperature(state, temp, can_rotate, subtract_drift)
    state, system = system.step(state, system, n=length // 2)
    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)

stride = cfg.min_save_decade
n = cfg.n_steps // stride
save_strides = jnp.full(n, stride)

def save_fn(st, sy):
    # perm = jnp.empty_like(st.unique_id)
    # perm = perm.at[st.unique_id].set(jnp.arange(st.unique_id.shape[0], dtype=st.unique_id.dtype))
    return dict(
        step_count=sy.step_count,
        #
        vel=st.vel,
        mass=st.mass,
        #
        pos=st.pos,
        rad=st.rad,
        pid=st.bond_id,
        # pos=st.pos[perm],
        # vel=st.vel[perm],
        pe=jd.utils.thermal.compute_potential_energy(st, sy),
        ke=jd.utils.thermal.compute_translational_kinetic_energy(st),
    )

rollout_kwargs = dict(strides=save_strides, save_fn=save_fn)

state, system, logged = jd.System.trajectory_rollout(
    state, system, **rollout_kwargs,
)

np.savez(
    'data.npz',
    **logged,
)

import matplotlib.pyplot as plt
plt.plot(logged['step_count'] * dt, logged['pe'], label='PE')
plt.plot(logged['step_count'] * dt, logged['ke'], label='KE')
plt.plot(logged['step_count'] * dt, logged['ke'] + logged['pe'], label='TE')
plt.legend()
# plt.axvline(tau_s, c='k', ls='--', alpha=0.5, zorder=0)
plt.savefig('energies.png')

from anim_utils import animate
animate(logged['pos'], logged['rad'], logged['pid'], system.domain.box_size, 'test.gif')
# print(logged['ke'])
# print(logged['pe'])


# while phi < 1.0:
#     state, system, mean_pe, _ = run_1(
#         state,
#         system,
#         data_root,
#         cfg,
#         save_strides=save_strides,
#         compress=True,
#         save_fn=None,
#         save_all=False
#     )
#     state, system = system.step(state, system, n=500_000)
#     state, system, mean_pe, _ = run_1(
#         state,
#         system,
#         data_root,
#         cfg,
#         save_strides=save_strides,
#         compress=False,
#         save_fn=None,
#         save_all=False
#     )
#     phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
