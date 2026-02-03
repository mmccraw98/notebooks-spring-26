import numpy as np
from scipy.optimize import minimize_scalar, brentq
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
from jaxdem.utils.geometricAsperityCreation import generate_ga_deformable_state
from functools import partial
from tqdm import tqdm
import os


def render(state, system, path, id_name='clump_ID', frame=None):
    import subprocess
    import h5py
    import os
    if frame is None:
        pos = np.asarray(state.pos)
        rad = np.asarray(state.rad)
        pid = np.asarray(getattr(state, id_name))
        box_size = np.asarray(system.domain.box_size)
    else:
        pos = np.asarray(state.pos[frame])
        rad = np.asarray(state.rad[frame])
        pid = np.asarray(getattr(state, id_name)[frame])
        box_size = np.asarray(system.domain.box_size[frame])
    with h5py.File('config.h5', 'w') as f:
        f.create_dataset("pos", data=pos)
        f.create_dataset("rad", data=rad)
        f.create_dataset("ID",  data=pid)
        f.create_dataset("box_size", data=box_size)
    run_render = "/home/mmccraw/dev/analysis/fall-25/12/testing-jaxdem-scripts/rigid-particle-creation/run_render.sh"
    # run_render = "/Users/marshallmccraw/Projects/yale/analysis/fall-25/12/testing-jaxdem-scripts/rigid-particle-creation/run_render.sh"
    subprocess.run([
        str(run_render),
        "config.h5",
        path,
        "1000",
    ], check=True)
    os.remove("config.h5")

def animate(traj_state, traj_system, path, frames=100, fps=15, id_name='clump_ID'):
    import subprocess
    from pathlib import Path
    import h5py
    with h5py.File("traj.h5", "w") as f:
        f.create_dataset("pos", data=np.asarray(traj_state.pos))
        f.create_dataset("rad", data=np.asarray(traj_state.rad))
        f.create_dataset("ID", data=np.asarray(getattr(traj_state, id_name)))
        f.create_dataset("box_size", data=np.asarray(traj_system.domain.box_size))

    # --- Optional: generate a GIF animation (requires ParaView pvbatch) ---
    script_dir = Path(__file__).resolve().parent
    run_animation = "/home/mmccraw/dev/analysis/fall-25/12/testing-jaxdem-scripts/animation/run_animation.sh"
    # run_animation = "/Users/marshallmccraw/Projects/yale/analysis/fall-25/12/testing-jaxdem-scripts/animation/run_animation.sh"
    subprocess.run(
        [
            str(run_animation),
            "traj.h5",
            path,
            str(frames),   # num_frames (evenly sampled if traj has more)
            "1000",  # base_pixels
            str(fps),    # fps
        ],
        check=True,
    )