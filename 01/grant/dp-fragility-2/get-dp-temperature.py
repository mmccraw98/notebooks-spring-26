import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
from bump_utils import create_dps_2d, step

temperatures = np.logspace(-6, -4, 20)

if __name__ == "__main__":

    # for which in ['2d-soft', '2d-med', '2d-hard', '2d-floppy']:
    for which in ['2d-floppy']:

        for temperature in temperatures:

            data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-temperature/{which}/{temperature}'

            if not os.path.exists(data_root):
                os.makedirs(data_root)

            if which == '2d-soft':
                from config import config2d_soft as cfg
            elif which == '2d-med':
                from config import config2d_med as cfg
            elif which == '2d-hard':
                from config import config2d_hard as cfg
            elif which == '2d-floppy':
                from config import config2d_floppy as cfg
            else:
                raise ValueError(f'Which {which} is unknown')

            seed = np.random.randint(0, 1e9)

            state, system, dp = create_dps_2d(
                phi=cfg.phi,
                N=cfg.N,
                mu_eff=cfg.mu_eff,
                aspect_ratio=1.0,
                min_nv=cfg.nv,
                mass=cfg.mass,
                eb=cfg.eb,
                el=cfg.el,
                ec=cfg.ec
            )
            key = jax.random.PRNGKey(seed)
            dp_vels = jax.random.normal(key, (cfg.N, state.dim))
            dp_vels -= jnp.mean(dp_vels, axis=0, keepdims=True)
            ke = 0.5 * jnp.sum(dp_vels ** 2) * cfg.mass
            temp = ke * 2 / (state.dim * state.N)
            dp_vels *= jnp.sqrt(temperature / temp)
            state.vel = dp_vels[state.deformable_ID]

            # run thermalization without compressing the states
            print('Running NVT...')
            state, system = jd.utils.control_nvt_density(
                state,
                system,
                n=cfg.n_dynamics_steps // 10,
                rescale_every=100,
                temperature_delta=temperature,  # maintain temperature
                packing_fraction_delta=0.0,  # do not compress on the first run
                can_rotate=False,
                subtract_drift=True,
            )
            print('Done')

            # save the initial data
            phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
            run_root = os.path.join(data_root, f'phi-{phi:.6f}')

            # run the dynamics, calculate quantities of interest, and save the data
            step(state, system, dp, cfg, run_root)
