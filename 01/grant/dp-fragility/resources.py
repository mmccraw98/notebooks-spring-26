import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd

def compute_ke(state):
    return 0.5 * state.mass * jnp.sum(state.vel ** 2, axis=-1)

def compute_dp_ke(state):
    N_dps = jnp.unique(state.deformable_ID).size
    dpids, nv = jnp.unique(state.deformable_ID, return_counts=True)
    vel_dp = jax.ops.segment_sum(state.vel, state.deformable_ID, N_dps) / nv[:, None]
    mass_dp = jax.ops.segment_sum(state.mass, state.deformable_ID, N_dps)
    return 0.5 * mass_dp * jnp.sum(vel_dp ** 2, axis=-1)

def compute_temp(state):
    dof = (state.N - 1) * state.dim
    ke = compute_ke(state)
    return 2 * jnp.sum(ke, axis=-1) / dof

def compute_dp_temp(state):  # TODO: need to measure the rotational kinetic energy!
    N_dps = jnp.unique(state.deformable_ID).size
    dof = (N_dps - 1) * state.dim
    ke = compute_dp_ke(state)
    return 2 * jnp.sum(ke, axis=-1) / dof

def scale_temps(state, target_temperatures):
    state.vel -= jnp.mean(state.vel, axis=-2, keepdims=True)
    temperature = compute_temp(state)
    scale = jnp.sqrt(target_temperatures / temperature)
    state.vel *= scale[:, None, None]
    return state

def scale_dp_temps(state, target_temperatures):
    N_dps = jnp.unique(state.deformable_ID).size
    dpids, nv = jnp.unique(state.deformable_ID, return_counts=True)
    vel_dp = jax.ops.segment_sum(state.vel, state.deformable_ID, N_dps) / nv[:, None]
    vel_dp -= jnp.mean(vel_dp, axis=-2, keepdims=True)
    temperature = compute_dp_temp(state)
    scale = jnp.sqrt(target_temperatures / temperature)
    state.vel *= scale

def compute_phi(state, system):
    return jnp.sum(state.volume, axis=-1) / jnp.prod(system.domain.box_size, axis=-1)

def increment_phi(state, system, delta_phi):
    phi = compute_phi(state, system)
    scale = (phi / (phi + delta_phi)) ** (1 / state.dim)
    state.pos_c *= scale[:, None, None]
    system.domain.box_size *= scale[:, None]
    return state, system
