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
        ['med'],
        [config2d_med],
    ):
        data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-4/{which}'
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        
        state, system, dp = create_ga_dps_2d(cfg)

        phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
        print(phi)
        
        state, system, dp, pe, run_root = step(
            cfg,
            input_path=data_root,  # use input path as the data root on initialization
            state=state,  # define state and system on initialization
            system=system,
            dp=dp,
        )
        jax.clear_caches()

        while phi < 0.95:
            phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
            print(phi)

            state, system, dp, pe, run_root = step(
                cfg,
                input_path=run_root,
            )
            jax.clear_caches()