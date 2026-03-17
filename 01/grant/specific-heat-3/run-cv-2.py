import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jaxdem.integrators import LinearIntegrator, RotationIntegrator
import jaxdem as jd
import numpy as np
import dataclasses
import os

from jaxdem_scripts.specific_heat import run_4, JobConfig as cv_config

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    args = parser.parse_args()

    input_root = args.input_root
    output_root = args.output_root

    # cfg = cv_config(
    #     seed=np.random.randint(0, 1e9),
    #     n_phi_steps=10,
    #     n_temperature_steps=10,
    #     can_rotate=True,
    #     subtract_drift=True,
    #     temp_min=1e-5,
    #     temp_max=2e-5,
    #     delta_phi_min=1e-6,
    # )

    cfg_0 = cv_config(
        seed=np.random.randint(0, 1e9),
        n_phi_steps=2,
        n_temperature_steps=10,
        can_rotate=True,
        subtract_drift=True,
        temp_min=1e-10,
        temp_max=2e-10,
        delta_phi_min=1e-16,
    )
    cfg = cfg_0

    with jd.CheckpointLoader(directory=input_root) as loader:
        state, system = loader.load()

    system = dataclasses.replace(
        system,
        linear_integrator=LinearIntegrator.create("verlet"),
        rotation_integrator=RotationIntegrator.create("verletspiral"),
    )

    state, system, rattler_ids, non_rattler_ids = jd.utils.contacts.get_clump_rattler_ids(
        state,
        system,
    )

    np.savez(
        os.path.join(output_root, 'rattler_ids.npz'),
        rattler_ids=rattler_ids,
        non_rattler_ids=non_rattler_ids,
    )

    run_4(state, system, output_root, cfg)