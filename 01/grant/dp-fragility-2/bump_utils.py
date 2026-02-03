import numpy as np
from scipy.optimize import minimize_scalar, brentq
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
from jaxdem.utils.geometricAsperityCreation import generate_ga_deformable_state
from functools import partial
from tqdm import tqdm
from file_management import save_arrs, make_data_dir
import os

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

@jax.jit
def msd_for_lags(dp_com: jax.Array, lags: jax.Array) -> jax.Array:
    T = dp_com.shape[0]
    M = dp_com.shape[1]
    t = jnp.arange(T, dtype=lags.dtype)  # (T,)

    def msd_one(lag: jax.Array) -> jax.Array:
        # y[t] = dp_com[t + lag] (mod T) with a validity mask to drop wrapped terms
        idx = (t + lag) % T  # (T,)
        diff = dp_com[idx] - dp_com  # (T, M, dim)
        sq = jnp.sum(diff * diff, axis=-1)  # (T, M)

        valid = (t < (T - lag)).astype(dp_com.dtype)  # (T,)
        numer = jnp.sum(sq * valid[:, None])
        denom = (T - lag) * M
        return numer / denom

    return jax.vmap(msd_one)(lags)

def compute_msd(pos, batch_size = 256):
    # increase batch size for more throughput, decrease for lower peak memory
    # Compute MSD with a JIT-compiled kernel. To avoid doing "all at once", run lags in batches.
    t = np.arange(1, pos.shape[0] - 1, dtype=np.int32)
    msd = np.empty((t.size,), dtype=np.float64)

    for s in tqdm(range(0, t.size, batch_size)):
        lags = jnp.asarray(t[s : s + batch_size])
        msd[s : s + batch_size] = np.asarray(msd_for_lags(pos, lags))
    return msd, t

def step(state, system, dp, cfg, run_root, save_trajectory=False):
    run_root_paths = make_data_dir(run_root)
    jd.utils.h5.save(state, os.path.join(run_root_paths['init'], 'state.h5'))
    jd.utils.h5.save(system, os.path.join(run_root_paths['init'], 'system.h5'))
    jd.utils.h5.save(dp, os.path.join(run_root_paths['init'], 'dp.h5'))
    
    # run dynamics
    print('Running dynamics...')
    n_snapshots = cfg.n_dynamics_steps // cfg.save_stride
    state, system, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=n_snapshots, stride=cfg.save_stride
    )
    pe = jax.vmap(jd.utils.thermal.compute_potential_energy)(state_traj, system_traj)
    ke = jax.vmap(jd.utils.compute_translational_kinetic_energy)(state_traj)
    state_traj = jax.vmap(reorder_state)(state_traj)  # un-permute the indices
    if save_trajectory:
        jd.utils.h5.save(state_traj, os.path.join(run_root_paths['traj'], 'state_traj.h5'))
        jd.utils.h5.save(system_traj, os.path.join(run_root_paths['traj'], 'system_traj.h5'))
    print('Done')

    # compute the com pos, vel, and the msd
    N_dps = int(jax.device_get(jnp.max(state_traj.deformable_ID))) + 1
    pos_dp = jax.vmap(lambda st: get_com_pos(st, N_dps=N_dps))(state_traj)
    vel_dp = jax.vmap(lambda st: get_com_vel(st, N_dps=N_dps))(state_traj)
    msd, t = compute_msd(pos_dp)

    # save the trajectory
    save_arrs([state_traj.pos, state_traj.vel], ['pos', 'vel'], os.path.join(run_root_paths['traj'], 'vertex_data.h5'))
    # save the com data
    save_arrs(
        [pe, ke, pos_dp, vel_dp, msd, t],
        ['pe', 'ke', 'pos_dp', 'vel_dp', 'msd', 't_dimless'],
        os.path.join(run_root_paths['traj'], 'data.h5')
    )

    # save the final state
    jd.utils.h5.save(state, os.path.join(run_root_paths['final'], 'state.h5'))
    jd.utils.h5.save(system, os.path.join(run_root_paths['final'], 'system.h5'))
    jd.utils.h5.save(dp, os.path.join(run_root_paths['final'], 'dp.h5'))

    return state, system, dp, state_traj, system_traj

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

def create_dps_2d(phi, N, mu_eff, aspect_ratio, min_nv, mass, eb, el, ec):
    dim = 2
    particle_radii = jd.utils.dispersity.get_polydisperse_radii(N)
    asperity_radius = get_closest_vertex_radius_for_mu_eff(mu_eff, min(particle_radii), min_nv)
    max_nv, max_mu_eff, err = find_num_vertices_for_target_mu_eff(mu_eff, asperity_radius, max(particle_radii))
    vertex_counts = np.ones_like(particle_radii).astype(int) * min_nv
    vertex_counts[particle_radii == max(particle_radii)] = max_nv

    state, dp, box_size = generate_ga_deformable_state(
        particle_radii,
        vertex_counts,
        phi,
        dim,
        asperity_radius,
        aspect_ratio=aspect_ratio,
        use_uniform_mesh=True,
        mass=mass,
        seed=np.random.randint(0, 1e9),
        ec=ec,
        eb=eb,
        el=el,
        em=None
    )

    e_int = 1.0
    dt = 1e-2

    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

    dp_force, dp_energy = dp.create_force_energy_functions(dp)

    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        # collider_type="neighborlist",
        # collider_kw=dict(
        #     state=state,
        #     cutoff=2.0 * jnp.max(state.rad),
        #     skin=0.03,
        # ),
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


def create_dps_3d(phi, N, asperity_radius, aspect_ratio, min_nv, mass, eb, el, ec, em):
    dim = 3
    particle_radii = jd.utils.dispersity.get_polydisperse_radii(N, [1.0], [1.0])
    vertex_counts = np.ones_like(particle_radii).astype(int) * min_nv

    state, dp, box_size = generate_ga_deformable_state(
        particle_radii,
        vertex_counts,
        phi,
        dim,
        asperity_radius,
        aspect_ratio=aspect_ratio,
        use_uniform_mesh=True,
        mass=mass,
        seed=np.random.randint(0, 1e9),
        eb=eb,
        el=el,
        ec=ec,
        em=em,
    )

    e_int = 1.0
    dt = 1e-2

    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)

    dp_force, dp_energy = dp.create_force_energy_functions(dp)

    cutoff = 2.0 * jnp.max(state.rad)
    number_density = state.N / jnp.prod(box_size)

    system = jd.System.create(
        state_shape=state.shape,
        dt=dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="neighborlist",
        collider_kw=dict(
            state=state,
            cutoff=2.0 * jnp.max(state.rad),
            skin=0.03,
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


def render(state, system, path, id_name='clump_ID'):
    import subprocess
    import h5py
    import os
    with h5py.File('config.h5', 'w') as f:
        f.create_dataset("pos", data=np.asarray(state.pos))
        f.create_dataset("rad", data=np.asarray(state.rad))
        f.create_dataset("ID",  data=np.asarray(getattr(state, id_name)))
        f.create_dataset("box_size", data=np.asarray(system.domain.box_size))
    run_render = "/home/mmccraw/dev/analysis/fall-25/12/testing-jaxdem-scripts/rigid-particle-creation/run_render.sh"
    # run_render = "/Users/marshallmccraw/Projects/yale/analysis/fall-25/12/testing-jaxdem-scripts/rigid-particle-creation/run_render.sh"
    subprocess.run([
        str(run_render),
        "config.h5",
        path,
        "1000",
    ], check=True)
    os.remove("config.h5")

def animate(traj_state, traj_system, path, frames=100, fps=15, id_name='clump_ID'):
    import subprocess
    from pathlib import Path
    import h5py
    with h5py.File("traj.h5", "w") as f:
        f.create_dataset("pos", data=np.asarray(traj_state.pos))
        f.create_dataset("rad", data=np.asarray(traj_state.rad))
        f.create_dataset("ID", data=np.asarray(getattr(traj_state, id_name)))
        f.create_dataset("box_size", data=np.asarray(traj_system.domain.box_size))

    # --- Optional: generate a GIF animation (requires ParaView pvbatch) ---
    script_dir = Path(__file__).resolve().parent
    run_animation = "/home/mmccraw/dev/analysis/fall-25/12/testing-jaxdem-scripts/animation/run_animation.sh"
    # run_animation = "/Users/marshallmccraw/Projects/yale/analysis/fall-25/12/testing-jaxdem-scripts/animation/run_animation.sh"
    subprocess.run(
        [
            str(run_animation),
            "traj.h5",
            path,
            str(frames),   # num_frames (evenly sampled if traj has more)
            "1000",  # base_pixels
            str(fps),    # fps
        ],
        check=True,
    )