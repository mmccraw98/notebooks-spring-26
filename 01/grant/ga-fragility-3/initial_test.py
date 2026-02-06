import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxdem as jd
import os

from utils import create_ga_clumps_2d, step

from config import config_010_10, config_010_12, config_010_15, config_010_20, config_050_10, config_050_12, config_050_15, config_050_20, config_100_10, config_100_12, config_100_15, config_100_20

if __name__ == "__main__":
    for which, cfg in zip(
        ['010_10', '010_12', '010_15', '010_20', '050_10', '050_12', '050_15', '050_20', '100_10', '100_12', '100_15', '100_20'],
        [config_010_10, config_010_12, config_010_15, config_010_20, config_050_10, config_050_12, config_050_15, config_050_20, config_100_10, config_100_12, config_100_15, config_100_20],
    ):
        data_root = f'/home/mmccraw/dev/data/26-01-01/grant/ga-fragilitiy/version-3/{which}'
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        
        state, system = create_ga_clumps_2d(cfg)
        state, system, pe, run_root = step(
            cfg,
            input_path=data_root,  # use input path as the data root on initialization
            state=state,  # define state and system on initialization
            system=system,
            use_dynamic_rollout=False
        )

        # on all subsequent runs use:
        state, system, pe, run_root = step(
            cfg,
            input_path=run_root  # begin with the data from the prior run
        )
