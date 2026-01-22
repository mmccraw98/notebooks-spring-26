import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
from tqdm import tqdm

from resources import *

def run_nvt_compression(state, system, delta_phi, target_temperatures, n_steps, nve_block_length):
    """
    (De)Compress the states by a set amount while running NVT using a rescaling thermostat
    Performs packing fraction increment and temperature rescaling in a single step
    Runs NVE dynamics for following steps
    state: State
    system: System
    delta_phi: float - packing fraction increment to (de)compress by
    target_temperatures: np.ndarray - target temperature for each state
    n_steps: int - total number of steps to run the compression for
    nve_block_length: int - number of steps to run NVE after compression (rescaling / increment frequency)
    """
    n_blocks = n_steps // nve_block_length
    delta_phi_per_block = delta_phi / n_blocks
    for block in tqdm(range(n_blocks), desc='Running NVT Compression'):
        state, system = increment_phi(state, system, delta_phi_per_block)
        state = scale_temps(state, target_temperatures)
        state, system, (state_traj, system_traj) = system.trajectory_rollout(
            state, system, n=1, stride=nve_block_length
        )
    return state, system
