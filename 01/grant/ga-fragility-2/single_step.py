import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
import argparse
import sys
from bump_utils import step
from config import config as cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    args = parser.parse_args()

    # load the final data from the previous run
    input_path = args.input_path.rstrip('/')
    data_root = os.path.dirname(input_path)
    state = jd.utils.h5.load(os.path.join(input_path, 'final', 'state.h5'))
    system = jd.utils.h5.load(os.path.join(input_path, 'final', 'system.h5'))

    # run thermalization while compressing the states
    print('Running NVT...')
    state, system = jd.utils.control_nvt_density(
        state,
        system,
        n=cfg.n_dynamics_steps // 10,
        rescale_every=100,
        temperature_target=cfg.target_temperature,  # maintain temperature
        packing_fraction_delta=cfg.delta_phi,  # compress
        can_rotate=cfg.can_rotate,
        subtract_drift=cfg.subtract_drift,
    )
    print('Done')

    # save the initial data
    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
    run_root = os.path.join(data_root, f'phi-{phi:.6f}')

    # run the dynamics, calculate quantities of interest, and save the data
    step(state, system, cfg, run_root)

    # run another step if the packing fraction is less than the target
    if phi < cfg.phi_target:
        script = os.path.abspath(__file__)
        os.execv(sys.executable, [sys.executable, script, "--input_path", run_root])