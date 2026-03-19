import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os
import numpy as np
from tqdm import tqdm

from jaxdem_scripts.ga_utils import create_bidisperse_ga_dps_2d
from jaxdem_scripts.compression_loop_dp import run_1, JobConfig as JCFG

e_b = 1e-2
# e_m = 1e-1  # length
e_l = 1e1  # length
e_c = 1e2  # area
tau_s = 1

size_ratios=(1.0, 1.0)

N_dps = 10
mu_eff = 0.2
min_nv = 20
phi = 0.8
aspect_ratio = 1.0
dp_mass = 1.0
dt = 1e-3
e_int = 1.0
can_rotate = False
subtract_drift = True
temp = 1e-4

cfg = JCFG(
    seed=np.random.randint(0, 1e9),
    delta_phi=1e-2,
    target_temperature=temp,
    n_steps=100_000,
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

length = 1000
phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
while phi < 0.85:
    state, system = jd.utils.packingUtils.scale_to_packing_fraction(state, system, phi + 1e-2)
    state, system = system.step(state, system, n=length // 2)
    state = jd.utils.thermal.scale_to_temperature(state, temp, can_rotate, subtract_drift)
    state, system = system.step(state, system, n=length // 2)
    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)

stride = cfg.min_save_decade
n = cfg.n_steps // stride
save_strides = jnp.full(n, stride)

state, system = jd.utils.dynamicsRoutines.control_nvt_density(
    state,
    system,
    n=cfg.n_steps,
    rescale_every=100,
    temperature_target=temp,
    packing_fraction_delta=0.0,
    can_rotate=can_rotate,
    subtract_drift=subtract_drift,
)


# state, system, (state_traj, system_traj) = jd.utils.dynamicsRoutines.control_nvt_density_rollout(
#     state,
#     system,
#     n=n,
#     stride=save_strides,
#     rescale_every=100,
#     temperature_target=temp,
#     packing_fraction_delta=0.0,
#     can_rotate=can_rotate,
#     subtract_drift=subtract_drift,
# )

# from anim_utils import animate
# animate(state_traj.pos, state_traj.rad, state_traj.bond_id, system.domain.box_size, 'test.gif')

from anim_utils import render
render(state, system, 'test.png', 'bond_id')
