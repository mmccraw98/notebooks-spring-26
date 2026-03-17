import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxdem as jd
import os

from utils import create_ga_clumps_2d, step

from anim_utils import animate

from config import config_010_10, config_010_12, config_010_15, config_010_20, config_050_10, config_050_12, config_050_15, config_050_20, config_100_10, config_100_12, config_100_15, config_100_20

if __name__ == "__main__":
    for which, cfg in zip(
        # ['010_10', '010_12', '010_15', '010_20', '050_10', '050_12', '050_15', '050_20', '100_10', '100_12', '100_15', '100_20'],
        # [config_010_10, config_010_12, config_010_15, config_010_20, config_050_10, config_050_12, config_050_15, config_050_20, config_100_10, config_100_12, config_100_15, config_100_20],
        ['100_10'],
        [config_100_10],
    ):
        data_root = f'/home/mmccraw/dev/data/26-01-01/grant/ga-fragilitiy/version-4/{which}/dynamic-rollout'
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        
        state, system = create_ga_clumps_2d(cfg)

        temp = 1e-5
        n_steps = 100_000
        n_stride = 1000

        state = jd.utils.thermal.set_temperature(
            state,
            temp,
            can_rotate=cfg.can_rotate,
            subtract_drift=cfg.subtract_drift
        )

        n_frames = n_steps // n_stride
        n_final_frames = int(n_frames * 0.2)

        state, system, (state_traj, system_traj) = jd.utils.dynamicsRoutines.control_nvt_density_rollout(
            state,
            system,
            n=n_frames,
            stride=n_stride,
            rescale_every=1,
            packing_fraction_delta=0.25,
            temperature_delta=0.0,
            can_rotate=cfg.can_rotate,
            subtract_drift=cfg.subtract_drift,
        )
        
        state, system, (state_traj2, system_traj2) = jd.utils.dynamicsRoutines.control_nvt_density_rollout(
            state,
            system,
            n=n_final_frames,
            stride=n_stride,
            rescale_every=1,
            packing_fraction_delta=0.0,
            temperature_delta=0.0,
            can_rotate=cfg.can_rotate,
            subtract_drift=cfg.subtract_drift,
        )

        import numpy as np

        rad = np.concatenate([np.asarray(state_traj.rad), np.asarray(state_traj2.rad)], axis=0)
        cid = np.concatenate([np.asarray(state_traj.clump_id), np.asarray(state_traj2.clump_id)], axis=0)
        box_size = np.concatenate([np.asarray(system_traj.domain.box_size), np.asarray(system_traj2.domain.box_size)], axis=0)
        pos_all = np.concatenate([np.asarray(state_traj.pos), np.asarray(state_traj2.pos)], axis=0)
        pos = np.mod(pos_all, box_size[:, None, :])

        np.savez(
            'anim-data.npz',
            pos=pos,
            rad=rad,
            cid=cid,
            box_size=box_size,
        )

        animate(
            pos,
            rad,
            cid,
            box_size,
            'compress.gif',
            frames=n_frames + n_final_frames,
        )
