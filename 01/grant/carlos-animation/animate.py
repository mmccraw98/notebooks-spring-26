import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os

from bump_utils import create_ga_2d
from jaxdem.utils.jamming import bisection_jam

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mu', type=float, required=True)
    parser.add_argument('--alpha', type=float, required=True)
    parser.add_argument('--nv', type=int, required=True)
    args = parser.parse_args()

    mu = args.mu
    alpha = args.alpha
    nv = args.nv

    phi = 0.75
    N = 1000
    temp = 1e-4
    n_steps = 10_000
    save_stride = 100

    which = f'mu-{mu}-alpha-{alpha}-nv-{nv}'
    data_root = f'/home/mmccraw/dev/data/26-01-01/grant/carlos-animation/{which}'

    if not os.path.exists(data_root):
        os.makedirs(data_root)
    
    state, system = create_ga_2d(
        phi=phi,
        N=N,
        mu_eff=mu,
        aspect_ratio=alpha,
        min_nv=nv,
        mass=1.0,
    )

    state = jd.utils.thermal.set_temperature(
        state,
        temp,
        can_rotate=True,
        subtract_drift=True
    )

    state, system = jd.utils.control_nvt_density(
        state,
        system,
        n=10_000,
        rescale_every=100,
        temperature_delta=temp,
        packing_fraction_delta=0.0,
        can_rotate=True,
        subtract_drift=True,
    )

    state, system = jd.utils.control_nvt_density(
        state,
        system,
        n=10_000,
        rescale_every=100,
        temperature_delta=temp,
        packing_fraction_delta=0.0,
        can_rotate=True,
        subtract_drift=True,
    )

    n_snapshots = n_steps // save_stride
    state, system, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=n_snapshots, stride=save_stride
    )


    jd.utils.h5.save(state, os.path.join(data_root, 'state.h5'))
    jd.utils.h5.save(system, os.path.join(data_root, 'system.h5'))

    from bump_utils import animate
    animate(state_traj, system_traj, f'{which}.gif')