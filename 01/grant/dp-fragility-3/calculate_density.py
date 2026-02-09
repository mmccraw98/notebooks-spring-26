import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxdem as jd
import os
from jaxdem.forces.force_manager import ForceManager

if __name__ == "__main__":
    root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-density'
    
    input_path = '/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-3/floppy/phi-0.537377'

    # load the data
    data_root = os.path.dirname(input_path)
    state = jd.utils.h5.load(os.path.join(input_path, 'final', 'state.h5'))
    system = jd.utils.h5.load(os.path.join(input_path, 'final', 'system.h5'))
    dp = jd.utils.h5.load(os.path.join(input_path, 'final', 'dp.h5'))
    dp_force, dp_energy = dp.create_force_energy_functions(dp)
    system.force_manager = ForceManager.create(
        state_shape=state.shape,
        gravity=None,
        force_functions=[(dp_force, dp_energy, False)],
    )

    n_steps = 10_000
    save_stride = 100
    n_snapshots = n_steps // save_stride
    state, system, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=n_snapshots, stride=save_stride
    )

    density = os.path.basename(input_path)
    which = os.path.basename(os.path.dirname(input_path))

    output_path = os.path.join(root, which, density)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    jd.utils.h5.save(state_traj, os.path.join(output_path, 'state_traj.h5'))
    jd.utils.h5.save(system_traj, os.path.join(output_path, 'system_traj.h5'))