from functools import partial
import jaxdem as jd
import jax
import jax.numpy as jnp
from bump_utils import animate, render
import os
import h5py
import numpy as np
import subprocess
from pathlib import Path

root = '/Users/marshallmccraw/Projects/yale/data/s-26/grant/dp-fragility/2d-soft'
for name in os.listdir(root)[::-1]:
    path = os.path.join(root, name)

    anim_path = f'anims/{name}-anim.gif'
    if os.path.exists(anim_path):
        continue

    state = jd.utils.h5.load(os.path.join(path, 'final', 'state.h5'))
    system = jd.utils.h5.load(os.path.join(path, 'final', 'system.h5'))

    lut = np.empty(state.unique_ID.size, dtype=state.deformable_ID.dtype)
    lut[state.unique_ID] = state.deformable_ID

    from file_management import load_arrs
    data = load_arrs(os.path.join(path, 'traj', 'data.h5'))
    pos = data['pos']
    unique_ID = data['unique_ID']
    n_frames = pos.shape[0]
    rad = np.array([state.rad for _ in range(n_frames)])
    deformable_ID = lut[unique_ID]
    box_size = np.array([system.domain.box_size for _ in range(n_frames)])

    M = int(jnp.max(deformable_ID[0]) + 1)  # or known constant
    total_pos = jax.vmap(lambda p, d: jax.ops.segment_sum(p, d, num_segments=M))(pos, deformable_ID)  # (S, M, dim)
    dp_counts = jax.vmap(lambda d: jax.ops.segment_sum(jnp.ones((d.shape[0],), dtype=pos.dtype), d, num_segments=M))(deformable_ID)  # (S, M)
    dp_com = total_pos / jnp.maximum(dp_counts[..., None], 1.0)  # (S, M, dim)
    import matplotlib.pyplot as plt
    for dpid in range(M):
        plt.plot(dp_com[:, dpid, 0], dp_com[:, dpid, 1])
    plt.savefig(f'figs/{name}-traj.png')
    plt.close()

    with h5py.File("traj.h5", "w") as f:
        f.create_dataset("pos", data=pos)
        f.create_dataset("rad", data=rad)
        f.create_dataset("deformable_ID", data=deformable_ID)
        f.create_dataset("box_size", data=box_size)

    # --- Optional: generate a GIF animation (requires ParaView pvbatch) ---
    script_dir = Path(__file__).resolve().parent
    # run_animation = "/home/mmccraw/dev/analysis/fall-25/12/testing-jaxdem-scripts/animation/run_animation.sh"
    run_animation = "/Users/marshallmccraw/Projects/yale/analysis/fall-25/12/testing-jaxdem-scripts/animation/run_animation.sh"
    subprocess.run(
        [
            str(run_animation),
            "traj.h5",
            anim_path,
            str(100),   # num_frames (evenly sampled if traj has more)
            "1000",  # base_pixels
            str(15),    # fps
        ],
        check=True,
    )
