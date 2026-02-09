import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import jaxdem as jd
import os
import subprocess

from utils import create_ga_clumps_2d, reorder_state

from config import config_010_10, config_010_12, config_010_15, config_010_20, config_050_10, config_050_12, config_050_15, config_050_20, config_100_10, config_100_12, config_100_15, config_100_20

cfg_map = {
    k: v for k, v in zip(
        ['010_10', '010_12', '010_15', '010_20', '050_10', '050_12', '050_15', '050_20', '100_10', '100_12', '100_15', '100_20'],
        [config_010_10, config_010_12, config_010_15, config_010_20, config_050_10, config_050_12, config_050_15, config_050_20, config_100_10, config_100_12, config_100_15, config_100_20],
    )
}

import argparse

n_repeats = 100

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--which', type=str, required=True)
    args = parser.parse_args()

    which = args.which
    if which not in cfg_map.keys():
        raise ValueError(f'which {which} not understood')
    cfg = cfg_map[which]

    for run_id in range(n_repeats):
        data_root = f'/path/{which}/{run_id}'
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        if os.path.exists(os.path.join(data_root, 'state.h5')):
            continue
        
        state, system = create_ga_clumps_2d(cfg)
        state, system, final_pf, final_pe = jd.utils.jamming.bisection_jam(state, system)
        state = reorder_state(state)  # un-permute the indices

        jd.utils.h5.save(state, os.path.join(data_root, 'state.h5'))
        jd.utils.h5.save(system, os.path.join(data_root, 'system.h5'))
