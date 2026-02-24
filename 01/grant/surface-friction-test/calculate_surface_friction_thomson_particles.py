import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import jaxdem as jd
from jaxdem.utils.geometricAsperityCreation import generate_ga_clump_state
from jaxdem.utils.quaternion import Quaternion
import trimesh
import math
from tqdm import tqdm
import os
import argparse

from jaxdem.utils.geometricAsperityCreation import generate_mesh, compute_mesh_properties

# TODO: add docstrings and type hints to all functions

CENTRAL_ID = 0
TRACER_ID = 1

def make_q_base(theta_base, phi_base):
    """
    Build q_base so that the body-frame direction (theta_base, phi_base)
    ends up facing the central particle.
    
    Parameters
    ----------
    theta_base : float   polar angle on tracer body sphere  [0, π]
    phi_base   : float   azimuthal angle                    [-π, π]
    """
    # body-frame direction we want to face the central particle
    probe_dir = jnp.array([
        jnp.sin(theta_base) * jnp.cos(phi_base),
        jnp.sin(theta_base) * jnp.sin(phi_base),
        jnp.cos(theta_base),
    ])

    # quat_from_x_to(-probe_dir) rotates [1,0,0] → -probe_dir
    # its inverse rotates probe_dir → [-1,0,0]  (which q_dir then aims at central)
    q_fwd_raw = quat_from_x_to(-probe_dir)                          # (4,)
    q_fwd = Quaternion(w=q_fwd_raw[0:1][None, :],                   # (1,1)
                       xyz=q_fwd_raw[1:4][None, :])                  # (1,3)
    return Quaternion.inv(q_fwd)

def quat_from_x_to(d):
    """Unit quaternion (w, x, y, z) rotating [1, 0, 0] to unit vector d.
    
    Uses the half-way quaternion formula:
        q = normalize([1 + dot([1,0,0], d),  cross([1,0,0], d)])
          = normalize([1 + dx,  0,  -dz,  dy])
    """
    q = jnp.array([1.0 + d[0], 0.0, -d[2], d[1]])
    norm = jnp.linalg.norm(q)
    # Antiparallel fallback (d ≈ [-1,0,0]): 180° about y-axis
    return jnp.where(norm < 1e-8, jnp.array([0.0, 0.0, 1.0, 0.0]), q / norm)

def random_points_on_sphere(key, N, S=1):
    """Generate n random points uniformly distributed on the unit sphere."""
    points = jax.random.normal(key, shape=(S, N, 3))
    norms = jnp.linalg.norm(points, axis=-1, keepdims=True)
    return (points / norms).squeeze()

def riesz_energy(pos, alpha):
    """Riesz energy kernel.  alpha=1 reduces to the Thomson problem.  alpha=\infty reduces to the packing problem"""
    r_ij = pos[:, None, :] - pos[None, :, :]
    # squared distances (no gradient issue here)
    d_sq = jnp.sum(r_ij**2, axis=-1)
    # fill diagonal with 1.0 BEFORE sqrt, so grad(sqrt(1.0)) = 0.5, not inf
    n = pos.shape[0]
    d_sq = d_sq.at[jnp.diag_indices(n)].set(1.0)
    d_ij = jnp.sqrt(d_sq)
    e_ij = 1.0 / d_ij ** alpha
    # zero out the diagonal so self-interactions don't contribute
    e_ij = e_ij.at[jnp.diag_indices(n)].set(0.0)
    return jnp.sum(jnp.triu(e_ij, k=1))

def project_to_tangent(grad, pos):
    """Remove the radial component of the gradient (project onto tangent plane of sphere)."""
    # For unit sphere, the normal at each point is just the position itself
    radial = jnp.sum(grad * pos, axis=-1, keepdims=True) * pos
    return grad - radial

def minimize_on_sphere(pos, alpha, lr=0.01, steps=1000):
    energy_grad = jax.grad(riesz_energy)
    def step(pos, _):
        g = energy_grad(pos, alpha)
        g_tangent = project_to_tangent(g, pos)
        pos = pos - lr * g_tangent
        # retract back to the sphere (normalize each point)
        pos = pos / jnp.linalg.norm(pos, axis=-1, keepdims=True)
        return pos, riesz_energy(pos, alpha)
    pos, energies = jax.lax.scan(step, pos, None, length=steps)
    return pos, energies

def create_state_and_system_thomson(
    particle_radii,
    vertex_counts,
    asperity_radius,
    body_type,
    N_steps,
):
    states = []
    for rad, nv in zip(particle_radii, vertex_counts):
        mass = 1.0
        nv = int(nv)
        add_core = body_type in ['true-solid', 'solid']

        key = jax.random.PRNGKey(np.random.randint(0, 1e9))
        asperity_pos_init = random_points_on_sphere(key, N=nv)
        if N_steps > 0:
            asperity_pos, _ = minimize_on_sphere(
                asperity_pos_init, alpha=1, lr=0.01 / nv, steps=N_steps
            )
        else:
            asperity_pos = asperity_pos_init

        core_radius = rad - asperity_radius
        asperity_pos = asperity_pos * core_radius
        asperity_radii = jnp.ones(nv) * asperity_radius

        if add_core:
            asperity_pos = jnp.concatenate([asperity_pos, jnp.zeros((1, 3))])
            asperity_radii = jnp.concatenate([asperity_radii, jnp.array([core_radius])])

        mesh = generate_mesh(
            asperity_positions=asperity_pos,
            asperity_radii=asperity_radii,
            subdivisions=6,
        )
        pos_c, q, inertia_dimensionless, volume = compute_mesh_properties(mesh, mass=1.0)

        if add_core and body_type == 'solid':
            asperity_pos = asperity_pos[:-1]
            asperity_radii = asperity_radii[:-1]

        n = asperity_pos.shape[0]
        Q = Quaternion.create(
            w=jnp.full((n, 1), q[0]),
            xyz=jnp.tile(q[1:], (n, 1)),
        )
        pos_c_tiled = jnp.tile(pos_c, (n, 1))

        state = jd.State.create(
            pos=asperity_pos,
            rad=asperity_radii,
            clump_ID=jnp.zeros(n),
            volume=jnp.ones(n) * volume,
            mass=jnp.ones(n) * mass,
            inertia=jnp.tile(inertia_dimensionless * mass, (n, 1)),
            q=Q,
        )
        state.pos_c = pos_c_tiled
        state.pos_p = Quaternion.rotate_back(Q, asperity_pos - pos_c_tiled)
        states.append(state)
    
    state = states[0]
    for s in states[1:]:
        state = jd.State.merge(state, s)

    box_size = jnp.ones(state.dim) * (jnp.sum(particle_radii) * 3)

    mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    system = jd.System.create(
        state_shape=state.shape,
        dt=1e-2,
        linear_integrator_type="",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
            anchor=jnp.zeros_like(box_size),
        ),
    )

    return state, system, box_size

@jax.jit
def find_contact_point(
    state,
    system,
    max_separation,
    min_separation,
    tracer_mask,
    offsets,
    pe_target,
    separation_tolerance
):
    # the jax code below is equivalent to this:
    # while ((separation_high - separation_low) > separation_tolerance):
    #     # check if the tracer is too far or too close based on the potential energy
    #     pe = jnp.sum(system.collider.compute_potential_energy(state, system))
    #     too_far = (pe < pe_target)
    #     too_close = (1 - too_far)
    #     # if too far, separation_high = separation
    #     separation_high = separation * (too_far) + separation_high * (too_close)
    #     # if too close, separation_low = separation
    #     separation_low = separation * (too_close) + separation_low * (too_far)

    #     # move the tracer according to the new mid-point
    #     new_separation = (separation_high + separation_low) / 2
    #     delta = separation - new_separation
    #     state.pos_c += delta * tracer_mask * direction  # shift along direction vector

    #     # update the separation
    #     separation = new_separation
    
    separation_high = max_separation
    separation_low = min_separation

    # calculate center of mass distance between each clump
    pos_c = state.pos_c[offsets]
    r_ij = pos_c[0] - pos_c[1]
    separation = jnp.linalg.norm(r_ij)
    direction = r_ij / separation

    # use this as the condition to continue running the while loop
    # while the brackets are too far away, continue running the while loop body
    def loop_condition(values):
        separation_high, separation_low, _, _ = values
        return (separation_high - separation_low) > separation_tolerance

    # run this function as the body of the while loop
    # it grows / shrinks the separation between the particles based on the potential energy
    def loop_body(values):
        separation_high, separation_low, separation, state = values
        # check if the tracer is too far or too close based on the potential energy
        pe = jnp.sum(system.collider.compute_potential_energy(state, system))
        too_far = (pe < pe_target)
        too_close = (1 - too_far)
        # if too far, separation_high = separation
        separation_high = separation * (too_far) + separation_high * (too_close)
        # if too close, separation_low = separation
        separation_low = separation * (too_close) + separation_low * (too_far)

        # move the tracer according to the new mid-point
        new_separation = (separation_high + separation_low) / 2
        delta = separation - new_separation
        state.pos_c += delta * tracer_mask[:, None] * direction  # shift along direction vector

        # update the separation
        separation = new_separation
        return separation_high, separation_low, separation, state

    # initial values for the while loop
    initial_values = (
        jnp.asarray(separation_high),
        jnp.asarray(separation_low),
        jnp.asarray(separation),
        state,
    )

    final_values = jax.lax.while_loop(
        loop_condition,
        loop_body,
        initial_values,
    )

    separation_high, separation_low, separation, state = final_values
    return separation, state

@jax.jit
def get_friction_for_config(
    state,
    system,
    tracer_position,
    quat,
    max_separation,
    min_separation,
    tracer_mask,
    offsets,
    pe_target,
    separation_tolerance
):
    state.pos_c = jnp.broadcast_to(system.domain.box_size / 2, state.pos_c.shape).copy()
    state.pos_c = state.pos_c + tracer_position * tracer_mask[:, None]
    state.q.w = jnp.where(tracer_mask[:, None], quat[0:1], 1.0)  # rotate
    state.q.xyz = jnp.where(tracer_mask[:, None], quat[1:4], 0.0)

    # find minimal separation point
    separation, state = find_contact_point(state, system, max_separation, min_separation, tracer_mask, offsets, pe_target, separation_tolerance)
    state, system = system.collider.compute_force(state, system)

    pos_c = state.pos_c[offsets]
    r_ij = pos_c[CENTRAL_ID] - pos_c[TRACER_ID]
    separation = jnp.linalg.norm(r_ij)
    direction = r_ij / separation
    force = jnp.sum(state.force * tracer_mask[:, None], axis=0)
    force_n_mag = jnp.sum(force * direction)
    force_t_mag = jnp.linalg.norm(force - force_n_mag * direction)
    return jnp.abs(force_t_mag / force_n_mag), separation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Surface friction test")
    parser.add_argument("--output_directory", type=str, help="Path containing the output file")
    parser.add_argument("--nv", type=int, default=100, help="Number of vertices in both particles")
    parser.add_argument("--particle_radius", type=float, default=0.5, help="Outer radius of both particles")
    parser.add_argument("--asperity_radius", type=float, default=0.125, help="Radius of all surface asperities on both particles")
    parser.add_argument("--n_theta", type=int, default=1, help="How many orientation theta angles to sample for the tracer")
    parser.add_argument("--n_phi", type=int, default=1, help="How many orientation theta angles to sample for the tracer")
    parser.add_argument("--batch_size", type=int, default=10_000, help="Compute friction coefficients for this many configurations at once")
    parser.add_argument("--subdivisions", type=int, default=3, help="Number of mesh subdivisions, larger means higher resolution")
    parser.add_argument("--N_steps", type=int, default=0, help="Number of steps in Thomson problem particle minimization.  Use 0 for random.  Larger values give more ordered particles")
    args = parser.parse_args()

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    nv = args.nv
    particle_radius = args.particle_radius
    asperity_radius = args.asperity_radius
    exterior_subdivisions = args.subdivisions
    batch_size = args.batch_size
    n_theta = args.n_theta
    n_phi = args.n_phi
    N_steps = args.N_steps

    pe_target = 1e-16
    separation_tolerance = 1e-10
    particle_radii = np.array([particle_radius, particle_radius])
    vertex_counts = np.array([nv, nv])
    core_type = "true"

    # set guessed bounds
    scale = 1.1
    max_separation = np.sum(particle_radii) * scale
    min_separation = (max_separation - 2 * asperity_radius) / scale
    med_separation = (max_separation - min_separation) / 2

    # Precompute all quaternions and positions
    sphere_mesh = trimesh.creation.icosphere(
        subdivisions=exterior_subdivisions,
        radius=1.0
    )

    # create inputs
    state, system, box_size = create_state_and_system_thomson(
        particle_radii=particle_radii,
        vertex_counts=vertex_counts,
        asperity_radius=asperity_radius,
        core_type=core_type,
        N_steps=N_steps,
    )

    # get the first id in each clump
    cids, offsets = jnp.unique(state.clump_ID, return_index=True)

    # move particle 0 to center, particle 1 to midpoint
    tracer_mask = (state.clump_ID == TRACER_ID)
    core_mask = (state.clump_ID == CENTRAL_ID)
    state.pos_c = box_size / 2  # center both particles
    state.pos_c += (med_separation) * tracer_mask[:, None] * jnp.array([1.0, 0.0, 0.0])  # offset the tracer

    # zero out the orientations
    state.q.w = jnp.ones_like(state.q.w)
    state.q.xyz = jnp.zeros_like(state.q.xyz)

    directions = jnp.array(sphere_mesh.vertices)
    quats = jax.vmap(quat_from_x_to)(directions)
    tracer_positions = directions * med_separation
    n_directions = directions.shape[0]

    compute_batch = jax.jit(jax.vmap(
        get_friction_for_config,
        in_axes=(None, None, 0, 0, None, None, None, None, None, None)
    ))

    theta_surface = jnp.arccos(directions[:, 2])
    phi_surface = jnp.arctan2(directions[:, 1], directions[:, 0])

    # Sweep over tracer base orientations
    theta_bases = np.linspace(0, np.pi / 2, n_theta, endpoint=True)
    phi_bases = np.linspace(0, np.pi / 2, n_phi, endpoint=True)

    for tracer_theta in tqdm(theta_bases, desc='theta_base'):
        for tracer_phi in tqdm(phi_bases, desc='phi_base', leave=False):
            q_base = make_q_base(tracer_theta, tracer_phi)

            q_dir = Quaternion(w=quats[:, 0:1], xyz=quats[:, 1:4])
            q_total = q_dir @ q_base
            q_total = Quaternion.unit(q_total)
            quats_composed = jnp.concatenate([q_total.w, q_total.xyz], axis=-1)

            mu = np.zeros(n_directions)
            separation = np.zeros(n_directions)

            n_batches = math.ceil(n_directions / batch_size)
            for batch_id in range(n_batches):
                start = batch_id * batch_size
                end = min(start + batch_size, n_directions)
                _mu, _separation = compute_batch(
                    state, system,
                    tracer_positions[start:end],
                    quats_composed[start:end],
                    max_separation, min_separation,
                    tracer_mask, offsets,
                    pe_target, separation_tolerance
                )
                mu[start:end] = _mu
                separation[start:end] = _separation

            name = (f'nv-{state.N // 2}-arad-{asperity_radius}'
                    f'-theta-{tracer_theta:.4f}-phi-{tracer_phi:.4f}.npz')
            np.savez(
                os.path.join(args.output_directory, name),
                theta=theta_surface,
                phi=phi_surface,
                mu=mu,
                separation=separation,
                tracer_theta=tracer_theta,
                tracer_phi=tracer_phi,
            )
    # SAVE POS, RAD, AND PID TO ANIMATE