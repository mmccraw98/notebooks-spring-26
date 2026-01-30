import numpy as np

def render(state, system, path, id_name='clump_ID'):
    import subprocess
    import h5py
    import os
    with h5py.File('config.h5', 'w') as f:
        f.create_dataset("pos", data=np.asarray(state.pos))
        f.create_dataset("rad", data=np.asarray(state.rad))
        f.create_dataset("ID",  data=np.asarray(getattr(state, id_name)))
        f.create_dataset("box_size", data=np.asarray(system.domain.box_size))
    run_render = "/home/mmccraw/dev/analysis/fall-25/12/testing-jaxdem-scripts/rigid-particle-creation/run_render.sh"
    subprocess.run([
        str(run_render),
        "config.h5",
        path,
        "1000",
    ], check=True)
    os.remove("config.h5")

def animate(traj_state, traj_system, path, frames=100, fps=15, id_name='clump_ID'):
    import subprocess
    import h5py
    with h5py.File("traj.h5", "w") as f:
        f.create_dataset("pos", data=np.asarray(traj_state.pos))
        f.create_dataset("rad", data=np.asarray(traj_state.rad))
        f.create_dataset("ID", data=np.asarray(getattr(traj_state, id_name)))
        f.create_dataset("box_size", data=np.asarray(traj_system.domain.box_size))

    # --- Optional: generate a GIF animation (requires ParaView pvbatch) ---
    run_animation = "/home/mmccraw/dev/analysis/fall-25/12/testing-jaxdem-scripts/animation/run_animation.sh"
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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize



def animate_flocks_2d(
    traj_state,
    traj_system=None,
    *,
    box_size=None,
    anchor=None,
    stride: int = 1,
    interval_ms: int = 30,
    fps: float | None = None,
    alpha: float = 0.9,
    cmap="hsv",
    radius_scale: float = 1.0,   # visual scale factor for radii
    save_path: str | None = None,
    dpi: int = 150,
    # If save_path ends with .mp4, we use ffmpeg/libx264 by default.
    ffmpeg_crf: int = 28,        # higher -> smaller file (typical 18-35)
    ffmpeg_preset: str = "slow", # slower -> smaller file
):
    """
    High-performance 2D flock animation.

    - Positions are wrapped into the periodic box.
    - View window is fixed to [anchor, anchor + box_size].
    - Marker sizes reflect `traj_state.rad` (exact radii, scaled by radius_scale).
    - Colors encode orientation from velocity angle using a periodic colormap.

    Note: Matplotlib scatter size `s` is in points^2, not data units. We convert
    radii in data-units to points using the current axes transform each frame.
    For best performance, we compute sizes once (fixed axis limits).
    """
    # Resolve box/anchor
    if box_size is None:
        if traj_system is None:
            raise ValueError("Provide box_size=... or pass traj_system=... to read domain.box_size.")
        box_size = np.asarray(traj_system.domain.box_size)
        if box_size.ndim == 2:
            box_size = box_size[0]
    box_size = np.asarray(box_size, dtype=float).reshape(2,)

    if anchor is None:
        if traj_system is not None and hasattr(traj_system.domain, "anchor"):
            anchor = np.asarray(traj_system.domain.anchor)
            if anchor.ndim == 2:
                anchor = anchor[0]
        else:
            anchor = np.zeros(2, dtype=float)
    anchor = np.asarray(anchor, dtype=float).reshape(2,)

    # Convert to NumPy once
    pos = np.asarray(traj_state.pos)[::stride]   # (T, N, 2)
    vel = np.asarray(traj_state.vel)[::stride]   # (T, N, 2)
    rad = np.asarray(traj_state.rad)
    if rad.ndim == 2:   # (T, N) possible in some stacked states
        rad = rad[::stride]
    if rad.ndim == 1:   # (N,)
        rad0 = rad
    elif rad.ndim == 2: # (T, N) -> assume constant in time; use first frame
        rad0 = rad[0]
    else:
        raise ValueError(f"Unexpected rad shape {rad.shape}")

    if pos.ndim != 3 or pos.shape[-1] != 2:
        raise ValueError(f"Expected 2D pos with shape (T,N,2); got {pos.shape}")

    # Wrap positions into the periodic box
    pos = (pos - anchor[None, None, :]) % box_size[None, None, :] + anchor[None, None, :]

    # Orientation -> scalar in [0, 1)
    theta = np.arctan2(vel[..., 1], vel[..., 0])          # (T, N)
    h = (theta + np.pi) / (2.0 * np.pi)
    h = np.mod(h, 1.0).astype(np.float32)

    T = pos.shape[0]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(anchor[0], anchor[0] + box_size[0])
    ax.set_ylim(anchor[1], anchor[1] + box_size[1])

    # Convert radii (data units) -> points^2 for scatter size `s`
    # With fixed limits, this is constant across frames.
    fig.canvas.draw()  # ensure transforms are ready
    p0 = ax.transData.transform(np.column_stack([np.zeros_like(rad0), np.zeros_like(rad0)]))
    p1 = ax.transData.transform(np.column_stack([rad0 * radius_scale, np.zeros_like(rad0)]))
    r_pix = np.linalg.norm(p1 - p0, axis=1)          # pixel radius
    r_pt = r_pix * 72.0 / fig.dpi                    # points radius
    sizes = np.pi * (r_pt ** 2)                      # points^2 area

    norm = Normalize(0.0, 1.0)
    sc = ax.scatter(
        pos[0, :, 0],
        pos[0, :, 1],
        s=sizes,
        c=h[0],
        cmap=cmap,
        norm=norm,
        alpha=alpha,
        linewidths=0,
    )

    def update(t):
        sc.set_offsets(pos[t])
        sc.set_array(h[t])
        return (sc,)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        interval=interval_ms,
        blit=True,
        repeat=False,
    )

    if save_path is not None:
        save_path_str = str(save_path)
        ext = save_path_str.lower().rsplit(".", 1)[-1] if "." in save_path_str else ""
        if fps is None:
            fps = 1000.0 / float(interval_ms)

        # Prefer MP4/H.264 when requested; it is dramatically smaller than GIF
        # for large-N scatter animations.
        if ext in {"mp4", "m4v"}:
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError(
                    "Saving .mp4 requires ffmpeg, but Matplotlib can't find an ffmpeg writer. "
                    "Install ffmpeg (e.g., `sudo apt-get install ffmpeg`) or save as .gif instead."
                )
            writer = animation.FFMpegWriter(
                fps=fps,
                codec="libx264",
                extra_args=[
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    str(int(ffmpeg_crf)),
                    "-preset",
                    str(ffmpeg_preset),
                ],
            )
            ani.save(save_path_str, dpi=dpi, writer=writer)
        else:
            ani.save(save_path_str, dpi=dpi)

    return ani