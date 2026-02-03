import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
from bump_utils import create_dps_2d
import matplotlib.pyplot as plt

if __name__ == "__main__":
    phi = 0.67
    N = 10
    mu = 0.1
    nv = 20
    mass = 10.0

    eb = 1e-3
    el = 1e0
    ec = 1e3

    seed = np.random.randint(0, 1e9)

    target_temperature = 1e-5

    collider_type = "naive"

    max_dt = 1e-2
    # min_dt = 9e-3
    # n_dts = 1
    min_dt = 1e-3
    n_dts = 5
    base_n_steps = 10_000
    base_save_stride = 100
    
    dts = np.logspace(np.log10(min_dt), np.log10(max_dt), n_dts)
    te_fluc = np.zeros_like(dts)

    for j, dt in enumerate(dts):
        print(f'iteration {j} of {len(dts)}')
        n_steps = int(max_dt / dt * base_n_steps)
        save_stride = int(max_dt / dt * base_save_stride)

        state, system, dp = create_dps_2d(
            phi=phi,
            N=N,
            mu_eff=mu,
            aspect_ratio=1.0,
            min_nv=nv,
            mass=mass,
            eb=eb,
            el=el,
            ec=ec,
            dt=dt,
            collider_type=collider_type
        )
        key = jax.random.PRNGKey(seed)
        dp_vels = jax.random.normal(key, (N, state.dim))
        dp_vels -= jnp.mean(dp_vels, axis=0, keepdims=True)
        ke = 0.5 * jnp.sum(dp_vels ** 2) * mass
        temp = ke * 2 / (state.dim * state.N)
        dp_vels *= jnp.sqrt(target_temperature / temp)
        state.vel = dp_vels[state.deformable_ID]

        n_snapshots = n_steps // save_stride
        state, system, (state_traj, system_traj) = system.trajectory_rollout(
            state, system, n=n_snapshots, stride=save_stride
            # state, system, n=n_steps, stride=1
        )
        pe = jax.vmap(jd.utils.thermal.compute_potential_energy)(state_traj, system_traj)
        ke = jax.vmap(jd.utils.compute_translational_kinetic_energy)(state_traj)
        
        # path = f'/home/mmccraw/dev/data/26-01-01/grant/dp-energy-conservation/dt-{dt}'
        # if not os.path.exists(path):
        #     os.makedirs(path)
        # jd.utils.h5.save(state_traj, os.path.join(path, 'state_traj.h5'))
        # jd.utils.h5.save(system_traj, os.path.join(path, 'system_traj.h5'))
        # jd.utils.h5.save(state, os.path.join(path, 'state.h5'))
        # jd.utils.h5.save(dp, os.path.join(path, 'dp.h5'))
        # jd.utils.h5.save(system, os.path.join(path, 'system.h5'))
        # np.savez(os.path.join(path, 'energies.npz'), pe=pe, ke=ke)

        te_fluc[j] = np.std(pe + ke) / np.mean(pe + ke)
        fig, ax = plt.subplots(1, 2, constrained_layout=True)
        ax[0].plot(pe)
        ax[0].plot(ke)
        ax[0].plot(pe + ke)
        ax[1].plot(pe + ke)
        plt.savefig(f'dt-{dt}.png')
        plt.close()

    plt.plot(dts, te_fluc)
    plt.plot(dts, dts ** 2)
    a, b = np.polyfit(np.log10(dts), np.log10(te_fluc), deg=1)
    print(a, b)
    plt.title(f'n={a}')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig('energies.png')
    plt.close()