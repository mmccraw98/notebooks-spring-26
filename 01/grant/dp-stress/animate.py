import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os
from jaxdem.forces.force_manager import ForceManager
from functools import partial
import numpy as np
from tqdm import tqdm

from jaxdem.utils import load_legacy_simulation

data_root = '/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-4/med'

n_steps = 10_000
save_stride = 1

for path in tqdm(os.listdir(data_root)):
    input_path = os.path.join(data_root, path, 'final')
    # state = jd.utils.h5.load(os.path.join(input_path, 'state.h5'))
    # system = jd.utils.h5.load(os.path.join(input_path, 'system.h5'))
    # dp = jd.utils.h5.load(os.path.join(input_path, 'dp.h5'))
    # # dp.ec = None  # TURN OFF CONTENT ENERGY
    # dp_force, dp_energy = dp.create_force_energy_functions(dp)
    # system.force_manager = ForceManager.create(
    #     state_shape=state.shape,
    #     gravity=None,
    #     force_functions=[(dp_force, dp_energy, False)],
    # )
    state, system = load_legacy_simulation(
        state_path=os.path.join(input_path, 'state.h5'),
        system_path=os.path.join(input_path, 'system.h5'),
        dp_path=os.path.join(input_path, 'dp.h5'),
    )

    N_dps = int(jnp.max(state.bond_id) + 1)

    n_steps = 10_000
    save_stride = 100

    state, system, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=n_steps // save_stride, stride=save_stride,
    )

    exit()

    jax.clear_caches()