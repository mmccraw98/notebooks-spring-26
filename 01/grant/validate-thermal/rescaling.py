import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import os
from functools import partial

if __name__ == "__main__":
    root = '/home/mmccraw/dev/data/26-01-01/grant/validate-thermal'
    p_dim_type = '2d-spheres'
    data_root = os.path.join(root, 'initial-data', p_dim_type)

    def test_nvt_1():
        """
        Set a temperature delta and test if it is reached
        Uses the default linear scheduler
        """
        initial_temperature = 1e-5
        temperature_delta = 1e-5
        can_rotate = 'spheres' not in data_root
        subtract_drift = True
        state = jd.utils.h5.load(os.path.join(data_root, 'state.h5'))
        system = jd.utils.h5.load(os.path.join(data_root, 'system.h5'))
        state = jd.utils.thermal.set_temperature(state, initial_temperature, can_rotate=can_rotate, subtract_drift=subtract_drift, seed=0)
        state, system = jd.utils.control_nvt_density(
            state,
            system,
            n=10_000,
            rescale_every=100,
            temperature_delta=temperature_delta,
            can_rotate=can_rotate,
            subtract_drift=subtract_drift,
        )
        final_temperature = jd.utils.thermal.compute_temperature(state, can_rotate=False, subtract_drift=True)
        assert jnp.isclose(final_temperature, initial_temperature + temperature_delta)

    def test_density_1():
        """
        Set a packing fraction delta and test if it is reached
        Uses the default linear scheduler
        """
        initial_temperature = 1e-5
        packing_fraction_delta = 0.01
        can_rotate = 'spheres' not in data_root
        subtract_drift = True
        state = jd.utils.h5.load(os.path.join(data_root, 'state.h5'))
        system = jd.utils.h5.load(os.path.join(data_root, 'system.h5'))
        state = jd.utils.thermal.set_temperature(state, initial_temperature, can_rotate=can_rotate, subtract_drift=subtract_drift, seed=0)
        initial_packing_fraction = jd.utils.packingUtils.compute_packing_fraction(state, system)
        state, system = jd.utils.control_nvt_density(
            state,
            system,
            n=10_000,
            rescale_every=100,
            packing_fraction_delta=packing_fraction_delta,
            can_rotate=can_rotate,
            subtract_drift=subtract_drift,
        )
        final_packing_fraction = jd.utils.packingUtils.compute_packing_fraction(state, system)
        assert jnp.isclose(final_packing_fraction, initial_packing_fraction + packing_fraction_delta)

    def test_density_2():
        """
        Compress by a packing fraction delta, keeping temperature constant
        Check if both are reached
        This is a typical use for the function
        Uses the default linear scheduler
        """
        initial_temperature = 1e-5
        packing_fraction_delta = 0.01
        can_rotate = 'spheres' not in data_root
        subtract_drift = True
        state = jd.utils.h5.load(os.path.join(data_root, 'state.h5'))
        system = jd.utils.h5.load(os.path.join(data_root, 'system.h5'))
        state = jd.utils.thermal.set_temperature(state, initial_temperature, can_rotate=can_rotate, subtract_drift=subtract_drift, seed=0)
        initial_packing_fraction = jd.utils.packingUtils.compute_packing_fraction(state, system)
        state, system = jd.utils.control_nvt_density(
            state,
            system,
            n=10_000,
            rescale_every=100,
            temperature_delta=0.0,
            packing_fraction_delta=packing_fraction_delta,
            can_rotate=can_rotate,
            subtract_drift=subtract_drift,
        )
        final_packing_fraction = jd.utils.packingUtils.compute_packing_fraction(state, system)
        final_temperature = jd.utils.thermal.compute_temperature(state, can_rotate=can_rotate, subtract_drift=subtract_drift)
        assert jnp.isclose(final_packing_fraction, initial_packing_fraction + packing_fraction_delta)
        assert jnp.isclose(final_temperature, initial_temperature)


    def test_rollout_1():
        """
        Compress by a packing fraction delta, while decreasing the temperature by a delta
        Use a rollout to get access to the trajectory data
        Uses the default linear scheduler
        """
        initial_temperature = 1e-4
        packing_fraction_delta = 0.1
        temperature_delta = -1e-5
        n_steps = 10_000
        save_stride = 10
        n_snapshots = int(n_steps) // int(save_stride)
        can_rotate = 'spheres' not in data_root
        subtract_drift = True
        state = jd.utils.h5.load(os.path.join(data_root, 'state.h5'))
        system = jd.utils.h5.load(os.path.join(data_root, 'system.h5'))
        state = jd.utils.thermal.set_temperature(state, initial_temperature, can_rotate=can_rotate, subtract_drift=subtract_drift, seed=0)
        state, system, (traj_state, traj_system) = jd.utils.control_nvt_density_rollout(
            state,
            system,
            n=n_snapshots,
            stride=save_stride,
            rescale_every=100,
            temperature_delta=temperature_delta,
            packing_fraction_delta=packing_fraction_delta,
            can_rotate=can_rotate,
            subtract_drift=subtract_drift,
        )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, constrained_layout=True, sharex=True)
        ax[0].plot(jax.vmap(partial(jd.utils.thermal.compute_temperature, can_rotate=can_rotate, subtract_drift=subtract_drift))(traj_state))
        ax[0].set_ylabel(r'$T$', fontsize=16)
        ax[1].plot(jax.vmap(partial(jd.utils.packingUtils.compute_packing_fraction))(traj_state, traj_system))
        ax[1].set_ylabel(r'$\phi$', fontsize=16)
        ax[2].plot(jax.vmap(partial(jd.utils.thermal.compute_potential_energy))(traj_state, traj_system))
        ax[2].set_ylabel(r'$PE$', fontsize=16)
        for a in ax:
            a.set_xlabel(r'$t$', fontsize=16)
        plt.savefig(f'figures/{p_dim_type}-rollout-1.png')
        plt.close()


    def test_rollout_2():
        """
        Modulate the density using a custom sine wave scheduler while keeping the temperature uncontrolled
        Use a rollout to get access to the trajectory data
        """
        initial_temperature = 1e-4
        packing_fraction_amplitude = -0.05
        n_steps = 10_000
        save_stride = 10
        n_snapshots = int(n_steps) // int(save_stride)
        can_rotate = 'spheres' not in data_root
        subtract_drift = True
        state = jd.utils.h5.load(os.path.join(data_root, 'state.h5'))
        system = jd.utils.h5.load(os.path.join(data_root, 'system.h5'))
        state = jd.utils.thermal.set_temperature(state, initial_temperature, can_rotate=can_rotate, subtract_drift=subtract_drift, seed=0)

        def sine_dens_schedule(k, K, start, target):
            x = k / jnp.maximum(K, 1)  # 0..1
            return start + packing_fraction_amplitude * jnp.sin(2.0 * jnp.pi * x)  # one full period

        state, system, (traj_state, traj_system) = jd.utils.control_nvt_density_rollout(
            state,
            system,
            n=n_snapshots,
            stride=save_stride,
            rescale_every=100,
            packing_fraction_delta=0.0,  # it needs to be defined to use the control, but is otherwise ignored
            density_schedule=sine_dens_schedule,
            can_rotate=can_rotate,
            subtract_drift=subtract_drift,
        )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, constrained_layout=True, sharex=True)
        ax[0].plot(jax.vmap(partial(jd.utils.thermal.compute_temperature, can_rotate=can_rotate, subtract_drift=subtract_drift))(traj_state))
        ax[0].set_ylabel(r'$T$', fontsize=16)
        ax[1].plot(jax.vmap(partial(jd.utils.packingUtils.compute_packing_fraction))(traj_state, traj_system))
        ax[1].set_ylabel(r'$\phi$', fontsize=16)
        ax[2].plot(jax.vmap(partial(jd.utils.thermal.compute_potential_energy))(traj_state, traj_system))
        ax[2].set_ylabel(r'$PE$', fontsize=16)
        for a in ax:
            a.set_xlabel(r'$t$', fontsize=16)
        plt.savefig(f'figures/{p_dim_type}-rollout-2.png')
        plt.close()

    test_density_1()
    test_density_2()
    test_nvt_1()
    test_rollout_1()
    test_rollout_2()