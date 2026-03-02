import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jaxdem.integrators import LinearIntegrator, RotationIntegrator
import jaxdem as jd
import numpy as np
import dataclasses
import os


target_temp = 1e-5
can_rotate = True
subtract_drift = True
n_steps = 5
save_stride = 100
n_steps = 100_000

for i in range(5):
    # load the input data
    data_root = f'/home/mmccraw/dev/data/26-01-01/grant/specific-heat-testing/{i}'
    if not os.path.exists(data_root):
        continue
    with jd.CheckpointLoader(directory=os.path.join(data_root, 'final')) as loader:
        state, system = loader.load()

    system = dataclasses.replace(
        system,
        linear_integrator=LinearIntegrator.create("verlet"),
        rotation_integrator=RotationIntegrator.create("verletspiral"),
    )

    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
    state, system = jd.utils.packingUtils.scale_to_packing_fraction(state, system, phi - 0)
    state = jd.utils.thermal.set_temperature(state, target_temp, can_rotate, subtract_drift, np.random.randint(0, 1e9))

    def save_fn(st, sy):
        return (
            jd.utils.thermal.compute_potential_energy(st, sy),
            jd.utils.thermal.compute_translational_kinetic_energy(st),
            jd.utils.thermal.compute_rotational_kinetic_energy(st),
            jd.utils.thermal.compute_temperature(st, can_rotate, subtract_drift),
        )

    state, system, logged = system.trajectory_rollout(
        state, system,
        n=n_steps // save_stride,
        stride=save_stride,
        save_fn=save_fn,
    )

    pe, ke, ke_r, temp = logged

    import matplotlib.pyplot as plt
    plt.plot(pe, label=i)
plt.legend()
plt.savefig(f'energies.png')
plt.close()
