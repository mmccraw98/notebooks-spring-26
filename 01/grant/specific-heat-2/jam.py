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

    n_repeats = 100
    phi = 0.5
    N = 100

    which = f'mu-{mu}-alpha-{alpha}-nv-{nv}'
    for run_id in range(n_repeats):
        data_root = f'/Users/marshallmccraw/Projects/yale/data/s-26/grant/specific-heat/jamming/{which}/{run_id}'
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        if os.path.exists(os.path.join(data_root, 'state.h5')):
            continue
        
        state, system = create_ga_2d(
            phi=phi,
            N=N,
            mu_eff=mu,
            aspect_ratio=alpha,
            min_nv=nv,
            mass=1.0,
        )

        state, system, _, _ = bisection_jam(
            state,
            system,
            pe_tol=1e-16,
            pe_diff_tol=1e-16,
            # f_tol=1e-12,
            # f_diff_tol=1e-12,
            # packing_fraction_increment=1e-4,
        )

        jd.utils.h5.save(state, os.path.join(data_root, 'state.h5'))
        jd.utils.h5.save(system, os.path.join(data_root, 'system.h5'))

        from bump_utils import render
        render(state, system, f'{which}.png')
        break