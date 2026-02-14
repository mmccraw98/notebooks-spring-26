import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy.optimize import minimize_scalar, brentq
import jax.numpy as jnp
import jaxdem as jd
from file_management import save_arrs, make_data_dir
import os
from jaxdem.analysis import LagBinsPseudoLog, evaluate_binned
from jaxdem.analysis.kernels import isf_self_isotropic_kernel, msd_kernel, unwrap_angles_2d, msad_kernel_2d, isf_angular_kernel_2d

from jaxdem.utils.geometricAsperityCreation import generate_ga_clump_state

def calc_mu_eff(vertex_radius, outer_radius, num_vertices):
    return 1 / np.sqrt(((2 * vertex_radius) / ((outer_radius - vertex_radius) * np.sin(np.pi / num_vertices))) ** 2 - 1)

def find_num_vertices_for_target_mu_eff(
    target_mu_eff: float,
    vertex_radius: float,
    outer_radius: float,
    num_vertices_min: int = 3,
    num_vertices_max: int = 100):
    best_nv = None
    best_mu = np.nan
    best_err = np.inf
    for nv in range(int(num_vertices_min), int(num_vertices_max) + 1):
        try:
            mu = float(calc_mu_eff(vertex_radius, outer_radius, nv))
        except (ValueError, ZeroDivisionError, FloatingPointError, OverflowError, TypeError):
            continue
        if not np.isfinite(mu):
            continue
        err = abs(mu - target_mu_eff)
        if err < best_err:
            best_nv, best_mu, best_err = nv, mu, err
    return best_nv, best_mu, best_err

def get_closest_vertex_radius_for_mu_eff(mu_eff, outer_radius, num_vertices):
    # Calculate mathematically valid bounds
    sin_term = np.sin(np.pi / num_vertices)
    min_vertex_radius = outer_radius * sin_term / (2 + sin_term) + 1e-12
    max_vertex_radius = outer_radius - 1e-12
    
    # Check if target mu_eff is achievable
    max_mu_eff = calc_mu_eff(min_vertex_radius, outer_radius, num_vertices)
    min_mu_eff = calc_mu_eff(max_vertex_radius, outer_radius, num_vertices)
    
    if mu_eff > max_mu_eff or mu_eff < min_mu_eff:
        # Target mu_eff is outside achievable range
        return np.nan
    try:
        # Use root finding since we want calc_mu_eff(vertex_radius) = mu_eff
        def objective(vertex_radius):
            return calc_mu_eff(vertex_radius, outer_radius, num_vertices) - mu_eff
        
        # Brent's method is robust for this monotonic function
        result = brentq(objective, min_vertex_radius, max_vertex_radius, xtol=1e-12)
        return result
        
    except (ValueError, RuntimeError, ZeroDivisionError):
        # Fallback to bounded scalar minimization if root finding fails
        def obj_squared(vertex_radius):
            try:
                return (calc_mu_eff(vertex_radius, outer_radius, num_vertices) - mu_eff) ** 2
            except (ValueError, RuntimeError, ZeroDivisionError):
                return np.inf
        
        result = minimize_scalar(obj_squared, bounds=(min_vertex_radius, max_vertex_radius), method='bounded')
        return result.x if result.success else np.nan


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

def step(cfg, input_path, state = None, system = None, use_dynamic_rollout = False):
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
    if use_dynamic_rollout:
        save_steps = jnp.asarray(jd.utils.make_save_steps_pseudolog(
            num_steps=cfg.n_dynamics_steps,
            reset_save_decade=cfg.reset_save_decade,
            min_save_decade=cfg.min_save_decade,
            decade=10,
            include_step0=True,
        ))

        def save_fn(st, sy):
            return (
                sy.step_count,
                st.pos_c,
                st.q.w,
                st.q.xyz,
                st.vel,
                st.angVel,
                st.clump_ID,
                st.unique_ID,
                jd.utils.thermal.compute_potential_energy(st, sy),
                jd.utils.thermal.compute_translational_kinetic_energy(st),
                jd.utils.thermal.compute_rotational_kinetic_energy(st),
            )

        state, system, logged = system.trajectory_rollout_at_steps(
            state, system, save_steps=save_steps, save_fn=save_fn,
        )
        print("Done")

        (step_ids, pos, q_w, q_xyz, vel, angVel, clump_ID_traj, unique_ID_traj, pe, ke, ke_r) = logged

        @jax.jit
        def ids_to_perm(ids):
            inv = jnp.empty_like(ids)  # inv[id] = current_index
            inv = inv.at[ids].set(jnp.arange(ids.shape[0], dtype=ids.dtype))
            return inv  # perm mapping current_index <- canonical id order

        perm_traj = jax.vmap(ids_to_perm)(unique_ID_traj)

        def reorder_traj(x):
            return jax.vmap(lambda x_t, p_t: x_t[p_t], in_axes=(0, 0))(x, perm_traj)

        pos_u = reorder_traj(pos)
        q_w_u = reorder_traj(q_w)
        q_xyz_u = reorder_traj(q_xyz)
        vel_u = reorder_traj(vel)
        angVel_u = reorder_traj(angVel)
        clump_u = reorder_traj(clump_ID_traj)

        # ---- dedupe: keep one representative particle per clump (use frame 0 after reorder) ----
        _, offsets = jnp.unique(clump_u[0], return_index=True)

        pos_u = pos_u[:, offsets, :]
        q_w_u = q_w_u[:, offsets, :]
        q_xyz_u = q_xyz_u[:, offsets, :]
        vel_u = vel_u[:, offsets, :]
        angVel_u = angVel_u[:, offsets, :]

        # calculate correlation functions
        step_ids = np.asarray(jax.device_get(step_ids))

        # correlation functions should use pos_u
        T = pos_u.shape[0]
        bins = LagBinsPseudoLog(
            T,
            dt_min=1,
            dt_max=int(step_ids[-1] - step_ids[0]),
            timestep=step_ids,
        )
        tau_steps = bins.values()
        t = tau_steps * float(system.dt)

    else:
        n_chunks = 10
        pos = []
        vel = []
        angVel = []
        q_w = []
        q_xyz = []
        pe = []
        ke = []
        ke_r = []
        for i in range(n_chunks):  # break up the dynamics to save on memory
            n_steps = cfg.n_dynamics_steps // n_chunks
            n_snapshots = (n_steps // cfg.min_save_decade)
            state, system, (state_traj, system_traj) = system.trajectory_rollout(
                state, system, n=n_snapshots, stride=cfg.min_save_decade
            )
            pe.append(jax.vmap(jd.utils.thermal.compute_potential_energy)(state_traj, system_traj))
            ke.append(jax.vmap(jd.utils.compute_translational_kinetic_energy)(state_traj))
            ke_r.append(jax.vmap(jd.utils.compute_rotational_kinetic_energy)(state_traj))
            unpermuted_state_traj = jax.vmap(reorder_state)(state_traj)  # un-permute the indices

            # get the clump positions and angles
            cids, offsets = jnp.unique(unpermuted_state_traj.clump_ID[0], return_index=True)
            pos.append(unpermuted_state_traj.pos_c[:, offsets])
            vel.append(unpermuted_state_traj.vel[:, offsets])
            angVel.append(unpermuted_state_traj.angVel[:, offsets])
            q_w.append(unpermuted_state_traj.q.w[:, offsets])
            q_xyz.append(unpermuted_state_traj.q.xyz[:, offsets])
        pos_u = jnp.concatenate(pos)
        vel_u = jnp.concatenate(vel)
        angVel_u = jnp.concatenate(angVel)
        q_w_u = jnp.concatenate(q_w)
        q_xyz_u = jnp.concatenate(q_xyz)
        pe = jnp.concatenate(pe)
        ke_r = jnp.concatenate(ke_r)
        ke = jnp.concatenate(ke)
        print('Done')

        T = pos_u.shape[0]
        bins = LagBinsPseudoLog(T, dt_min=1, dt_max=T-1)  # pseudo-log lags using time-indices
        tau_steps = bins.values()
        t = tau_steps * float(system.dt) * cfg.min_save_decade

    # calculate the correlation functions for each size
    corrs = {}
    _, nv = jnp.unique(state.clump_ID, return_counts=True)
    for _nv, name, diam in zip([min(nv), max(nv)], ['small', 'large'], [1.0, 1.4]):
        mask = nv == _nv

        msd_res = evaluate_binned(msd_kernel, {"pos": pos_u[:, mask]}, bins)
        msd = np.asarray(msd_res.mean)
        corrs.update({f'msd_{name}': msd})

        k = 2.0 * jnp.pi / diam  # wave vector for the particle diameter
        isf_res = evaluate_binned(isf_self_isotropic_kernel, {"pos": pos_u[:, mask]}, bins, kernel_kwargs={"k": k})
        isf = np.asarray(isf_res.mean)
        corrs.update({f'isf_{name}': isf})

        theta = unwrap_angles_2d(q_w_u[:, mask], q_xyz_u[:, mask])
        theta_0 = 2 * np.pi / (2 * np.pi / _nv)  # angular period for the particle symmetry angle

        msad_res = evaluate_binned(msad_kernel_2d, {"theta": theta}, bins)
        msad = np.asarray(msad_res.mean)
        corrs.update({f'msad_{name}': msad})

        aisf_res = evaluate_binned(isf_angular_kernel_2d,{"theta": theta}, bins, kernel_kwargs={"theta_0": theta_0})
        aisf = np.asarray(aisf_res.mean)
        corrs.update({f'aisf_{name}': aisf})

    save_arrs(
        [pe, ke, ke_r, pos_u, q_w_u, q_xyz_u, vel_u, angVel_u, t] + [v for v in corrs.values()],
        ['pe', 'ke', 'ke_r', 'pos', 'q_w', 'q_xyz', 'vel', 'angVel', 't'] + [k for k in corrs.keys()],
        os.path.join(run_root_paths['traj'], 'data.h5')
    )

    # save the final state
    jd.utils.h5.save(state, os.path.join(run_root_paths['final'], 'state.h5'))
    jd.utils.h5.save(system, os.path.join(run_root_paths['final'], 'system.h5'))
    return state, system, pe, run_root

def create_ga_clumps_2d(cfg):
    dim = 2
    particle_radii = jd.utils.dispersity.get_polydisperse_radii(cfg.N)
    asperity_radius = get_closest_vertex_radius_for_mu_eff(cfg.mu_eff, min(particle_radii), cfg.min_nv)
    max_nv, max_mu_eff, err = find_num_vertices_for_target_mu_eff(cfg.mu_eff, asperity_radius, max(particle_radii))
    vertex_counts = np.ones_like(particle_radii).astype(int) * cfg.min_nv
    vertex_counts[particle_radii == max(particle_radii)] = max_nv
    state, box_size = generate_ga_clump_state(
        particle_radii,
        vertex_counts,
        cfg.phi,
        dim,
        asperity_radius,
        body_type='solid',
        aspect_ratio=cfg.aspect_ratio,
        use_uniform_mesh=True,
        mass=cfg.mass,
        seed=np.random.randint(0, 1e9),
    )
    mats = [jd.Material.create("elastic", young=cfg.e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    system = jd.System.create(
        state_shape=state.shape,
        dt=cfg.dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="verletspiral",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="neighborlist",
        collider_kw=dict(
            state=state,
            cutoff=2.0 * jnp.max(state.rad),
            skin=0.05,
            safety_factor=5.0,
        ),
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )
    return state, system
