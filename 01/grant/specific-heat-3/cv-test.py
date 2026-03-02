import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jaxdem.integrators import LinearIntegrator, RotationIntegrator
import jaxdem as jd
import numpy as np
import dataclasses
import os

from jaxdem_scripts.specific_heat import run_1, run_3, JobConfig as cv_config

cfg = cv_config(
    seed=np.random.randint(0, 1e9),
    n_phi_steps=10,
    n_temperature_steps=10,
    can_rotate=True,
    subtract_drift=True,
    temp_min=1e-5,
    temp_max=2e-5,
)

for i in [0, 3]:
    # load the input data
    data_root = f'/home/mmccraw/dev/data/26-01-01/grant/specific-heat-testing/{i}'
    with jd.CheckpointLoader(directory=os.path.join(data_root, 'final')) as loader:
        state, system = loader.load()

    system = dataclasses.replace(
        system,
        linear_integrator=LinearIntegrator.create("verlet"),
        rotation_integrator=RotationIntegrator.create("verletspiral"),
    )

    root = os.path.join('cv-data', str(i))
    os.makedirs(root, exist_ok=True)
    run_3(state, system, root, cfg)