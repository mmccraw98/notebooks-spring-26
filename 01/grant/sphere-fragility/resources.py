import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd

def compute_ke(state):
    return 0.5 * state.mass * jnp.sum(state.vel ** 2, axis=-1)

def compute_temp(state):
    dof = (state.N - 1) * state.dim
    ke = compute_ke(state)
    return 2 * jnp.sum(ke, axis=-1) / dof

def scale_temps(state, target_temperatures):
    state.vel -= jnp.mean(state.vel, axis=-2, keepdims=True)
    temperature = compute_temp(state)
    scale = jnp.sqrt(target_temperatures / temperature)
    state.vel *= scale[:, None, None]
    return state

def compute_phi(state, system):
    return jnp.sum(state.volume, axis=-1) / jnp.prod(system.domain.box_size, axis=-1)

def increment_phi(state, system, delta_phi):
    phi = compute_phi(state, system)
    scale = (phi / (phi + delta_phi)) ** (1 / state.dim)
    state.pos_c *= scale[:, None, None]
    system.domain.box_size *= scale[:, None]
    return state, system

def create(pos, rad, box_size, e_int, dt):
    state = jd.State.create(
        pos=pos,
        rad=rad,
        mass=jnp.ones(pos.shape[0])
    )
    mats = [jd.Material.create("elastic", young=e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
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
        #     cutoff=jnp.max(rad)
        # ),
        mat_table=mat_table,
        domain_kw=dict(
            box_size=box_size,
        ),
    )
    return state, system