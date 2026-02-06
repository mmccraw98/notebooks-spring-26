import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import jaxdem as jd
from file_management import save_arrs, make_data_dir
import os
from jaxdem.analysis import LagBinsPseudoLog, evaluate_binned
from jaxdem.analysis.kernels import isf_self_isotropic_kernel, msd_kernel
from jaxdem.utils.randomSphereConfiguration import random_sphere_configuration

@jax.jit
def reorder_state(state):
    ids = state.unique_ID  # (N,), permutation of 0..N-1
    inv = jnp.empty_like(ids)              # inv[id] = current_index
    inv = inv.at[ids].set(jnp.arange(ids.shape[0], dtype=ids.dtype))
    perm = inv  # canonical order is id=0,1,2,...,N-1
    def reorder_leaf(x):
        if hasattr(x, "ndim") and x.ndim >= 1 and x.shape[0] == perm.shape[0]:
            return x[perm]
        return x
    return jax.tree_util.tree_map(reorder_leaf, state)

def step(cfg, input_path, state = None, system = None):
    first_iteration = False
    if state is None and system is None:
        # load the data
        data_root = os.path.dirname(input_path)
        state = jd.utils.h5.load(os.path.join(input_path, 'final', 'state.h5'))
        system = jd.utils.h5.load(os.path.join(input_path, 'final', 'system.h5'))
    elif state is not None and system is not None:
        data_root = input_path
        first_iteration = True
    else:
        raise ValueError('State and System objects both need to be defined!')

    # compress and maintain temperature
    print('Running NVT...')
    state, system = jd.utils.control_nvt_density(
        state,
        system,
        n=cfg.n_dynamics_steps // 20,
        rescale_every=100,
        temperature_target=cfg.target_temperature,  # maintain temperature
        packing_fraction_delta=cfg.delta_phi * (not first_iteration),  # compress
        can_rotate=cfg.can_rotate,
        subtract_drift=cfg.subtract_drift,
    )
    # maintain temperature
    state, system = jd.utils.control_nvt_density(
        state,
        system,
        n=cfg.n_dynamics_steps // 20,
        rescale_every=100,
        temperature_target=cfg.target_temperature,  # maintain temperature
        packing_fraction_delta=0.0,  # maintain density
        can_rotate=cfg.can_rotate,
        subtract_drift=cfg.subtract_drift,
    )
    print('Done')

    # create the directories
    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
    run_root = os.path.join(data_root, f'phi-{phi:.6f}')
    run_root_paths = make_data_dir(run_root)

    # save initial data
    jd.utils.h5.save(state, os.path.join(run_root_paths['init'], 'state.h5'))
    jd.utils.h5.save(system, os.path.join(run_root_paths['init'], 'system.h5'))
    
    # run dynamics
    print('Running dynamics...')
    save_steps = jnp.asarray(jd.utils.make_save_steps_pseudolog(
        num_steps=cfg.n_dynamics_steps,
        reset_save_decade=cfg.reset_save_decade,
        min_save_decade=cfg.min_save_decade,
        decade=10,
        include_step0=True,
    ))

    state, system, (state_traj, system_traj) = system.trajectory_rollout_at_steps(
       state, system, save_steps=save_steps,
    )
    print('Done')

    # calculate energies
    pe = jax.vmap(jd.utils.thermal.compute_potential_energy)(state_traj, system_traj)
    ke = jax.vmap(jd.utils.compute_translational_kinetic_energy)(state_traj)
    unpermuted_state_traj = jax.vmap(reorder_state)(state_traj)  # un-permute the indices

    # get the positions and velocities
    pos = unpermuted_state_traj.pos
    vel = unpermuted_state_traj.vel

    # calculate correlation functions
    step_ids = np.array(system_traj.step_count)
    T = pos.shape[0]
    bins = LagBinsPseudoLog(
        T,
        dt_min=1,
        dt_max=int(step_ids[-1] - step_ids[0]),
        timestep=step_ids,
    )
    msd_res = evaluate_binned(msd_kernel, {"pos": pos}, bins)
    msd = np.array(msd_res.mean)
    k = 2.0 * jnp.pi / jnp.min(2 * state.rad)
    isf_res = evaluate_binned(isf_self_isotropic_kernel, {"pos": pos}, bins, kernel_kwargs={"k": k})
    isf = np.array(isf_res.mean)
    tau_steps = bins.values()
    t = tau_steps * float(system.dt)

    # save the trajectory
    save_arrs(
        [pe, ke, pos, vel, msd, isf, t],
        ['pe', 'ke', 'pos', 'vel', 'msd', 'isf', 't'],
        os.path.join(run_root_paths['traj'], 'data.h5')
    )

    # save the final state
    jd.utils.h5.save(state, os.path.join(run_root_paths['final'], 'state.h5'))
    jd.utils.h5.save(system, os.path.join(run_root_paths['final'], 'system.h5'))
    return state, system, state_traj, system_traj, pe, run_root

def create_spheres(cfg):
    seed = np.random.randint(0, 1e9)
    rad = jd.utils.dispersity.get_polydisperse_radii(cfg.N)
    pos, box_size = random_sphere_configuration(rad, cfg.phi, cfg.dim)
    state = jd.State.create(
        pos=pos,
        rad=rad,
        mass=jnp.ones(pos.shape[0])
    )
    if cfg.force_model == 'spring':
        mats = [jd.Material.create("elastic", young=cfg.e_int, poisson=0.5, density=1.0)]
    elif cfg.force_model == 'wca_shifted':
        mean_sigma = 2.0 * jnp.mean(rad)
        eps_wca = cfg.e_int * mean_sigma**2 / 456.0
        mats = [jd.Material.create("lj", epsilon=eps_wca, density=1.0)]
    else:
        raise ValueError(f'force_model {cfg.force_model} unknown')
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    system = jd.System.create(
        state_shape=state.shape,
        dt=cfg.dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type=cfg.force_model,
        collider_type="neighborlist",
        collider_kw=dict(
            state=state,
            cutoff=2.0 * jnp.max(state.rad),
            skin=0.05,
            safety_factor=2.0,
        ),
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )
    state = jd.utils.thermal.set_temperature(
        state,
        cfg.target_temperature,
        can_rotate=cfg.can_rotate,
        subtract_drift=cfg.subtract_drift,
        seed=seed
    )
    return state, system
