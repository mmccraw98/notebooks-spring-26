import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jaxdem as jd
import os

from utils import create_spheres, step

from config import config_2d_1, config_2d_2, config_2d_3, config_2d_wca_1, config_2d_wca_2, config_2d_wca_3

if __name__ == "__main__":
    for which, cfg in zip(
        ['1', '2', '3', 'wca_1', 'wca_2', 'wca_3'],
        [config_2d_1, config_2d_2, config_2d_3, config_2d_wca_1, config_2d_wca_2, config_2d_wca_3],
    ):
        data_root = f'/home/mmccraw/dev/data/26-01-01/grant/sphere-fragilitiy/version-3/{which}'
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        
        state, system = create_spheres(cfg)
        state, system, _, _, pe, run_root = step(
            cfg,
            input_path=data_root,  # use input path as the data root on initialization
            state=state,  # define state and system on initialization
            system=system
        )

        # # on all subsequent runs use:
        # state, system, _, _, pe, run_root = step(
        #     cfg,
        #     input_path=run_root  # begin with the data from the prior run
        # )