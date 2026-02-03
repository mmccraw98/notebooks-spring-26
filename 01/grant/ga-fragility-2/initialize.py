import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
from bump_utils import create_ga_2d, step

from config import config as cfg

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mu', type=float, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    args = parser.parse_args()

    mu = args.mu
    alpha = args.alpha
    nv = cfg.nv

    which = f'mu-{mu}-alpha-{alpha}-nv-{nv}'
    data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-fragility-2/{which}'

    if not os.path.exists(data_root):
        os.makedirs(data_root)

    state, system = create_ga_2d(
        phi=cfg.phi,
        N=cfg.N,
        mu_eff=mu,
        aspect_ratio=alpha,
        min_nv=cfg.nv,
        mass=cfg.mass,
        e_int=cfg.e_int,
        dt=cfg.dt,
    )

    # initialize the temperature
    state = jd.utils.thermal.set_temperature(
        state,
        cfg.target_temperature,
        can_rotate=cfg.can_rotate,
        subtract_drift=cfg.subtract_drift,
        seed=np.random.randint(0, 1e9),
    )


    # run thermalization without compressing the states
    print('Running NVT...')
    state, system = jd.utils.control_nvt_density(
        state,
        system,
        n=cfg.n_dynamics_steps // 10,
        rescale_every=100,
        temperature_delta=cfg.target_temperature,  # maintain temperature
        packing_fraction_delta=0.0,  # do not compress on the first run
        can_rotate=cfg.can_rotate,
        subtract_drift=cfg.subtract_drift,
    )
    print('Done')

    # save the initial data
    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
    run_root = os.path.join(data_root, f'phi-{phi:.6f}')

    # run the dynamics, calculate quantities of interest, and save the data
    step(state, system, dp, cfg, run_root)
