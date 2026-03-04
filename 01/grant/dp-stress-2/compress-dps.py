import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os
import numpy as np
from tqdm import tqdm

from jaxdem_scripts.ga_utils import create_bidisperse_ga_dps_2d
from jaxdem_scripts.compression_loop_dp import run_1, JobConfig as JCFG

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("name")
args = parser.parse_args()
name = args.name

e_b_map = {
    'hi': 1e0,
    'mid': 1e-1,
    'low': 1e-2,
}

if name not in e_b_map.keys():
    raise ValueError('Name not understood')

e_b = e_b_map[name]  # bending
e_m = 1e0  # length
e_c = 1e4  # area

N_dps = 100
mu_eff = 0.1
min_nv = 20
phi = 0.7
aspect_ratio = 1.0
dp_mass = 1.0
dt = 1e-3
e_int = 1.0
can_rotate = False
subtract_drift = True
temp = 1e-3

cfg = JCFG(
    seed=np.random.randint(0, 1e9),
    delta_phi=1e-2,
    target_temperature=temp,
    n_steps=1_000_000,
    min_save_decade=100,
)

data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-5/compression-4/{name}'

state, system = create_bidisperse_ga_dps_2d(
    N_dps,
    mu_eff,
    min_nv,
    phi,
    aspect_ratio,
    dp_mass,
    dt,
    e_int,
    e_m,
    e_c,
    e_b,
    e_l=None,
    e_gamma=None
)

state = jd.utils.thermal.set_temperature(
    state,
    temp,
    can_rotate,
    subtract_drift,
    np.random.randint(0, 1e9)
)

phi = jd.utils.packingUtils.compute_packing_fraction(state, system)

stride = cfg.min_save_decade
n = cfg.n_steps // stride
save_strides = jnp.full(n, stride)

while phi < 1.0:
    state, system, mean_pe, _ = run_1(
        state,
        system,
        data_root,
        cfg,
        save_strides=save_strides,
        compress=True,
        save_fn=None,
        save_all=False
    )
    state, system = system.step(state, system, n=500_000)
    state, system, mean_pe, _ = run_1(
        state,
        system,
        data_root,
        cfg,
        save_strides=save_strides,
        compress=False,
        save_fn=None,
        save_all=False
    )
    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
