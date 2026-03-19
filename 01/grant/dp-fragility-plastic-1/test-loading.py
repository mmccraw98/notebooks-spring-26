import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os
import numpy as np
from tqdm import tqdm

from jaxdem_scripts.ga_utils import create_bidisperse_ga_dps_2d
from jaxdem_scripts.compression_loop_dp import run_1, JobConfig as JCFG

# saving and loading works fine with h5 module

e_b = 1e-1
# e_m = 1e-1  # length
e_l = 1e1  # length
e_c = 1e2  # area
tau_s = 1.0

size_ratios=(1.0, 1.0)

N_dps = 10
mu_eff = 0.2
min_nv = 20
phi = 0.8
aspect_ratio = 1.0
dp_mass = 1.0
dt = 1e-4
e_int = 1.0
can_rotate = False
subtract_drift = True
temp = 1e-3

cfg = JCFG(
    seed=np.random.randint(0, 1e9),
    delta_phi=1e-2,
    target_temperature=temp,
    n_steps=1_000_000,
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

state.pos_c += jnp.asarray(np.random.normal(loc=0, scale=1e-6, size=(state.N, state.dim)))

state, system = system.collider.compute_force(state, system)

jd.utils.h5.save(state, 'state.h5')
jd.utils.h5.save(system, 'system.h5')

f1 = state.force
state.force *= 0.0

state = jd.utils.h5.load('state.h5')
system = jd.utils.h5.load('system.h5')

state, system = system.collider.compute_force(state, system)
f2 = state.force

assert jnp.all(jnp.isclose(f1, f2))