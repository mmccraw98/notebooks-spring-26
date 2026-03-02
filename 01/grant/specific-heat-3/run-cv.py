import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jaxdem.integrators import LinearIntegrator, RotationIntegrator
import jaxdem as jd
import numpy as np
import dataclasses
import os

from jaxdem_scripts.specific_heat import run_1, run_3, JobConfig as cv_config

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root

    cfg = cv_config(
        seed=np.random.randint(0, 1e9),
        n_phi_steps=10,
        n_temperature_steps=10,
        can_rotate=True,
        subtract_drift=True,
        temp_min=1e-5,
        temp_max=2e-5,
        delta_phi_min=1e-6,
    )

    with jd.CheckpointLoader(directory=input_root) as loader:
        state, system = loader.load()

    system = dataclasses.replace(
        system,
        linear_integrator=LinearIntegrator.create("verlet"),
        rotation_integrator=RotationIntegrator.create("verletspiral"),
    )

    run_3(state, system, output_root, cfg)