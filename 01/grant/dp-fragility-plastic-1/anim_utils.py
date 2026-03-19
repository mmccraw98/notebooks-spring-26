def render(state, system, path, id_name='clump_ID'):
    import subprocess
    import numpy as np
    import h5py
    import os
    with h5py.File('config.h5', 'w') as f:
        f.create_dataset("pos", data=np.asarray(state.pos))
        f.create_dataset("rad", data=np.asarray(state.rad))
        f.create_dataset("ID",  data=np.asarray(getattr(state, id_name)))
        f.create_dataset("box_size", data=np.asarray(system.domain.box_size))
    # run_render = "/home/mmccraw/dev/analysis/fall-25/12/testing-jaxdem-scripts/rigid-particle-creation/run_render.sh"
    run_render = "/Users/marshallmccraw/Projects/yale/analysis/fall-25/12/testing-jaxdem-scripts/rigid-particle-creation/run_render.sh"
    subprocess.run([
        str(run_render),
        "config.h5",
        path,
        "1000",
    ], check=True)
    os.remove("config.h5")

def animate(pos, rad, pid, box_size, path, frames=100, fps=15):
    import subprocess
    import numpy as np
    import h5py
    with h5py.File("traj.h5", "w") as f:
        f.create_dataset("pos", data=pos)
        f.create_dataset("rad", data=rad)
        f.create_dataset("ID", data=pid)
        f.create_dataset("box_size", data=box_size)

    # run_animation = "/home/mmccraw/dev/analysis/fall-25/12/testing-jaxdem-scripts/animation/run_animation.sh"
    run_animation = "/Users/marshallmccraw/Projects/yale/analysis/fall-25/12/testing-jaxdem-scripts/animation/run_animation.sh"
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