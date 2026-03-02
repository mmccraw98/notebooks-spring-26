import jax
jax.config.update("jax_enable_x64", True)

from jaxdem_scripts.ga_utils import create_bidisperse_ga_clumps_2d
from jaxdem.minimizers import LinearFIRE, RotationFIRE
from jaxdem.integrators import LinearIntegrator, RotationIntegrator
import jaxdem as jd
import numpy as np
import dataclasses
import os

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mu_eff", type=float, required=True)
    parser.add_argument("--aspect_ratio", type=float, required=True)
    parser.add_argument("--run_id", type=int, required=True)
    args = parser.parse_args()

    mu_eff = args.mu_eff
    aspect_ratio = args.aspect_ratio
    run_id = args.run_id

    N_clumps = 100
    min_nv = 20
    phi = 0.5
    clump_mass = 1.0
    dt = 1e-2
    e_int = 1.0

    temp = 1e-5
    can_rotate = True
    subtract_drift = True
    n_steps = 5

    # run for various mu and alpha, 10x each, taking 5 steps

    state, system = create_bidisperse_ga_clumps_2d(N_clumps, mu_eff, min_nv, phi, aspect_ratio, clump_mass, dt, e_int, body_type='solid')

    for i in range(n_steps):
        system = dataclasses.replace(
            system,
            linear_integrator=LinearIntegrator.create("verlet"),
            rotation_integrator=RotationIntegrator.create("verletspiral"),
        )

        phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
        state, system = jd.utils.packingUtils.scale_to_packing_fraction(state, system, phi - 1e-3)
        state = jd.utils.thermal.set_temperature(state, temp, can_rotate, subtract_drift, np.random.randint(0, 1e9))
        state, system = system.step(state, system, n=100_000)

        system = dataclasses.replace(
            system,
            linear_integrator=LinearFIRE.Create(),
            rotation_integrator=RotationFIRE.Create(),
        )

        state, system, final_pf, final_pe = jd.utils.jamming.bisection_jam(state, system)

        phi = jd.utils.packingUtils.compute_packing_fraction(state, system)

        # save the final data
        data_root = f'/home/mmccraw/dev/data/26-01-01/grant/specific-heat/mu-{mu_eff}-alpha-{aspect_ratio}/{run_id}/{i}/jamming'
        os.makedirs(data_root, exist_ok=True)
        with jd.CheckpointWriter(directory=data_root) as writer:
            writer.save(state, system)
        
        jax.clear_caches()