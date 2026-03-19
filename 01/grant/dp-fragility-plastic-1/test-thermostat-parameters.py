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
dt = 1e-3
e_int = 1.0
can_rotate = False
subtract_drift = True
temp = 1e-4

cfg = JCFG(
    seed=np.random.randint(0, 1e9),
    delta_phi=1e-2,
    target_temperature=temp,
    n_steps=1_000_00,
    min_save_decade=1000,
)

data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-plastic-fragilitiy/version-1/'

gamma_values = [1e-2, 1e-1, 1e0, 1e1, 1e2]

logged_values = {}

for i, gamma in enumerate(gamma_values):

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
        )
    )

    length = 10_000
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
        return dict(
            pe=jd.utils.thermal.compute_potential_energy(st, sy),
            ke=jd.utils.thermal.compute_translational_kinetic_energy(st),
        )

    rollout_kwargs = dict(strides=save_strides, save_fn=save_fn)
    state, system, logged = jd.System.trajectory_rollout(
        state, system, **rollout_kwargs,
    )

    logged_values.update({f'{k}_{i}': v for k, v in logged.items()})

np.savez(
    'gamma-sweep.npz',
    gamma=gamma_values,
    **logged_values,
)