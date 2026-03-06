import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os
import numpy as np
from tqdm import tqdm

from jaxdem_scripts.sphere_utils import create_spheres
from jaxdem_scripts.compression_loop_spheres import run_1, JobConfig as JCFG

N = 100
phi = 0.7
dim = 2
mass = 1.0
dt = 1e-2
e_int = 1.0
can_rotate = False
subtract_drift = True
temp = 1e-4

cfg = JCFG(
    seed=np.random.randint(0, 1e9),
    delta_phi=1e-2,
    target_temperature=temp,
    n_steps=1_000_000,
    min_save_decade=100,
)

data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-5/compression-disks-1/'

state, system = create_spheres(N, dim, phi, mass, dt, e_int)

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
