import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxdem as jd
import os

from utils import create_ga_dps_2d, step
from utils_old import create_ga_dps_2d as create_ga_dps_2d_old, step as step_old

from config import config2d_floppy, config2d_hard, config2d_soft, config2d_med

if __name__ == "__main__":
    for which, cfg in zip(
        # ['floppy', 'hard', 'soft', 'med'],
        # [config2d_floppy, config2d_hard, config2d_soft, config2d_med],
        ['hard'],
        [config2d_hard],
    ):
        data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-4/{which}'
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        
        state, system, dp = create_ga_dps_2d(cfg)
        step(
            cfg,
            input_path=data_root,  # use input path as the data root on initialization
            state=state,  # define state and system on initialization
            system=system,
            dp=dp,
        )
        jax.clear_caches()

        # data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-4-old/{which}'
        # if not os.path.exists(data_root):
        #     os.makedirs(data_root)

        # state, system, dp = create_ga_dps_2d_old(cfg)
        # step_old(
        #     cfg,
        #     input_path=data_root,  # use input path as the data root on initialization
        #     state=state,  # define state and system on initialization
        #     system=system,
        #     dp=dp,
        # )
        # jax.clear_caches()
