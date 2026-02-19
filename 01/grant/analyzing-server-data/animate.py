import numpy as np
import h5py

def load_arrs(path):
    with h5py.File(path, 'r') as f:
        return {name: f[name][()] for name in f.keys()}

def render(state, system, path, id_name='clump_ID'):
    import subprocess
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

def animate(traj_state, traj_system, path, frames=100, fps=15, id_name='clump_ID'):
    animate_given_data(
        pos=np.asarray(traj_state.pos),
        rad=np.asarray(traj_state.rad),
        pid=np.asarray(getattr(traj_state, id_name)),
        box_size=np.asarray(traj_system.domain.box_size),
        path=path,
        frames=frames,
        fps=fps,
    )

def animate_given_data(pos, rad, pid, box_size, path, frames=100, fps=15):
    import subprocess
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

import jaxdem as jd
import os

dp_data = np.load('data/dp-dynamics-data-3-longer-2.npz', allow_pickle=True)

for ptype, eb in zip(['floppy', 'hard'], [None, 1.0]):
    for phi in ['phi-0.707377', 'phi-0.737377', 'phi-0.777377', 'phi-0.807377', 'phi-0.847377', 'phi-0.907377']:
        phi_float = float(phi.split('-')[-1])
        mask = (dp_data['eb'] == eb) & (dp_data['phi'] == phi_float)
        phi_true = dp_data['phi_true'][mask][0]
        
        # OLD WAY:
        # root = f'/Users/marshallmccraw/Projects/yale/data/s-26/grant/server-data/dp-data-for-animation/{ptype}/{phi}/traj-for-density'
        # state_traj = jd.utils.h5.load(os.path.join(root, 'state_traj.h5'))
        # system_traj = jd.utils.h5.load(os.path.join(root, 'system_traj.h5'))
        # animate(state_traj, system_traj, f'animations/{ptype}-{phi_true}.gif')
        
        # NEW WAY:
        root = f'/Users/marshallmccraw/Projects/yale/data/s-26/grant/server-data/new-dp-data-for-animation/{ptype}/{phi}'
        data = load_arrs(os.path.join(root, 'anim-data.h5'))
        
        pos = data['pos']
        n_frames = pos.shape[0]
        box_size = np.array([data['box_size'] for _ in range(n_frames)])
        rad = np.array([data['rad'] for _ in range(n_frames)])
        dpid = np.array([data['dp_ids'] for _ in range(n_frames)])

        run_length = 5  # seconds

        animate_given_data(
            pos,
            rad,
            dpid,
            box_size,
            os.path.join('new-animations', f"{ptype}-{data['phi_true'][0]}-relaxed-{data['too_long']}.gif"),
            frames=n_frames,
            fps=n_frames / run_length,
        )
