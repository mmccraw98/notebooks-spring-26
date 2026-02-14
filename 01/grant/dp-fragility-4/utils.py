import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy.optimize import minimize_scalar, brentq
import jax.numpy as jnp
import jaxdem as jd
from file_management import save_arrs, make_data_dir
import os
from functools import partial
from jaxdem.analysis import LagBinsPseudoLog, evaluate_binned
from jaxdem.analysis.kernels import isf_self_isotropic_kernel, msd_kernel
from jaxdem.utils.geometricAsperityCreation import generate_ga_deformable_state
from jaxdem.forces.force_manager import ForceManager

@partial(jax.jit, static_argnames=("N_dps",))
def get_com_pos(st, N_dps):
    total_pos = jax.ops.segment_sum(st.pos_c, st.deformable_ID, num_segments=N_dps)
    dp_counts = jax.ops.segment_sum(
        jnp.ones((st.N,), dtype=st.pos_c.dtype),
        st.deformable_ID,
        num_segments=N_dps,
    )
    return total_pos / jnp.maximum(dp_counts[:, None], 1.0)

@partial(jax.jit, static_argnames=("N_dps",))
def get_com_vel(st, N_dps):
    total_vel = jax.ops.segment_sum(st.vel, st.deformable_ID, num_segments=N_dps)
    dp_counts = jax.ops.segment_sum(
        jnp.ones((st.N,), dtype=st.vel.dtype),
        st.deformable_ID,
        num_segments=N_dps,
    )
    return total_vel / jnp.maximum(dp_counts[:, None], 1.0)

def compute_com(arr, deformable_ID, N_dps):
    """Compute center of mass from component array (positions or velocities)."""
    total = jax.ops.segment_sum(arr, deformable_ID, num_segments=N_dps)
    counts = jax.ops.segment_sum(
        jnp.ones(arr.shape[0], dtype=arr.dtype),
        deformable_ID,
        num_segments=N_dps,
    )
    return total / jnp.maximum(counts[:, None], 1.0)

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

def step(cfg, input_path, state = None, system = None, dp = None):
    first_iteration = False
    if state is None and system is None:
        # load the data
        data_root = os.path.dirname(input_path)
        state = jd.utils.h5.load(os.path.join(input_path, 'final', 'state.h5'))
        system = jd.utils.h5.load(os.path.join(input_path, 'final', 'system.h5'))
        dp = jd.utils.h5.load(os.path.join(input_path, 'final', 'dp.h5'))
        dp_force, dp_energy = dp.create_force_energy_functions(dp)
        system.force_manager = ForceManager.create(
            state_shape=state.shape,
            gravity=None,
            force_functions=[(dp_force, dp_energy, False)],
        )
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
    jd.utils.h5.save(dp, os.path.join(run_root_paths['init'], 'dp.h5'))
    
    # run dynamics
    print('Running dynamics...')
    N_dps = int(jax.device_get(jnp.max(state.deformable_ID))) + 1

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
            st.pos,
            st.vel,
            st.deformable_ID,
            st.unique_ID,
            jd.utils.thermal.compute_potential_energy(st, sy),
            jd.utils.thermal.compute_translational_kinetic_energy(st),
        )

    state, system, logged = system.trajectory_rollout_at_steps(
        state, system, save_steps=save_steps, save_fn=save_fn,
    )
    print('Done')

    (step_ids, pos, vel, deformable_ID, unique_ID, pe, ke) = logged

    # reorder the vertex data
    def reorder_frame(pos_t, uid_t):
        return pos_t[jnp.argsort(uid_t)]

    pos = jax.vmap(reorder_frame)(pos, unique_ID)
    vel = jax.vmap(reorder_frame)(vel, unique_ID)
    deformable_ID = jax.vmap(lambda did, uid: did[jnp.argsort(uid)])(deformable_ID, unique_ID)

    # Reconstruct COM positions and velocities from per-frame component trajectories
    pos_dp = jax.vmap(lambda p, did: compute_com(p, did, N_dps))(pos, deformable_ID)
    vel_dp = jax.vmap(lambda v, did: compute_com(v, did, N_dps))(vel, deformable_ID)

    # correlation functions
    step_ids = np.asarray(jax.device_get(step_ids))
    T = pos_dp.shape[0]
    bins = LagBinsPseudoLog(
        T,
        dt_min=1,
        dt_max=int(step_ids[-1] - step_ids[0]),
        timestep=step_ids,
    )
    tau_steps = bins.values()
    t = tau_steps * float(system.dt)
    
    corrs = {}
    _, nv = jnp.unique(state.deformable_ID, return_counts=True)
    for _nv, name, diam in zip([min(nv), max(nv)], ['small', 'large'], [1.0, 1.4]):
        mask_dp = nv == _nv
        mask = jnp.isin(deformable_ID[0], jnp.where(mask_dp)[0])

        msd_res = evaluate_binned(msd_kernel, {"pos": pos_dp[:, mask_dp]}, bins)
        msd = np.array(msd_res.mean)
        corrs.update({f'msd_{name}': msd})

        msd_res = evaluate_binned(msd_kernel, {"pos": pos[:, mask]}, bins, chunk_size=1_000)
        msd_vertex = np.array(msd_res.mean)
        corrs.update({f'msd_vertex_{name}': msd_vertex})

        k = 2.0 * jnp.pi / diam
        isf_res = evaluate_binned(isf_self_isotropic_kernel, {"pos": pos_dp[:, mask_dp]}, bins, kernel_kwargs={"k": k})
        isf = np.array(isf_res.mean)
        corrs.update({f'isf_{name}': isf})

        isf_res = evaluate_binned(isf_self_isotropic_kernel, {"pos": pos[:, mask]}, bins, kernel_kwargs={"k": k}, chunk_size=1_000)
        isf_vertex = np.array(isf_res.mean)
        corrs.update({f'isf_vertex_{name}': isf_vertex})

    pe, ke, pos, vel, pos_dp, vel_dp = jax.device_get((pe, ke, pos, vel, pos_dp, vel_dp))
    save_arrs(
        [pe, ke, pos, vel, pos_dp, vel_dp, t] + [v for v in corrs.values()],
        ['pe', 'ke', 'pos', 'vel', 'pos_dp', 'vel_dp', 't'] + [k for k in corrs.keys()],
        os.path.join(run_root_paths['traj'], 'data.h5')
    )

    # save the final state
    jd.utils.h5.save(state, os.path.join(run_root_paths['final'], 'state.h5'))
    jd.utils.h5.save(system, os.path.join(run_root_paths['final'], 'system.h5'))
    jd.utils.h5.save(dp, os.path.join(run_root_paths['final'], 'dp.h5'))
    return state, system, dp, pe, run_root


def create_ga_dps_2d(cfg):
    dim = 2
    particle_radii = jd.utils.dispersity.get_polydisperse_radii(cfg.N)
    asperity_radius = get_closest_vertex_radius_for_mu_eff(cfg.mu_eff, min(particle_radii), cfg.min_nv)
    max_nv, max_mu_eff, err = find_num_vertices_for_target_mu_eff(cfg.mu_eff, asperity_radius, max(particle_radii))
    vertex_counts = np.ones_like(particle_radii).astype(int) * cfg.min_nv
    vertex_counts[particle_radii == max(particle_radii)] = max_nv

    state, dp, box_size = generate_ga_deformable_state(
        particle_radii,
        vertex_counts,
        cfg.phi,
        dim,
        asperity_radius,
        aspect_ratio=cfg.aspect_ratio,
        use_uniform_mesh=True,
        mass=cfg.mass,
        seed=np.random.randint(0, 1e9),
        ec=cfg.ec,
        eb=cfg.eb,
        el=cfg.el,
        em=cfg.em,
    )

    mats = [jd.Material.create("elastic", young=cfg.e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

    dp_force, dp_energy = dp.create_force_energy_functions(dp)

    system = jd.System.create(
        state_shape=state.shape,
        dt=cfg.dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="",
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
        force_manager_kw=dict(
            force_functions=(
                (dp_force, dp_energy),
            ),
        ),
    )
    return state, system, dp
