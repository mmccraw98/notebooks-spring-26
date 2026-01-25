from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from functools import partial
from typing import Tuple

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

import jaxdem as jd
from jaxdem.utils import Quaternion
from jaxdem.utils import geometricAsperityCreation

jax.config.update("jax_enable_x64", True)


def euler_xyz_to_quaternion(euler_xyz: jnp.ndarray) -> Quaternion:
    q_xyzw = Rotation.from_euler("xyz", euler_xyz).as_quat()
    return Quaternion.create(w=q_xyzw[..., 3:4], xyz=q_xyzw[..., 0:3])


def spherical_unit(theta: float, phi: float) -> jnp.ndarray:
    st = jnp.sin(theta)
    return jnp.array([st * jnp.cos(phi), st * jnp.sin(phi), jnp.cos(theta)])


def set_clump_pose(state: jd.State, clump_id: int, pos_c: jnp.ndarray, q: Quaternion) -> jd.State:
    mask = state.clump_ID == clump_id
    mask_v = mask[..., None]
    pos_c_new = jnp.where(mask_v, pos_c, state.pos_c)
    w_full = jnp.broadcast_to(q.w, state.q.w.shape)
    xyz_full = jnp.broadcast_to(q.xyz, state.q.xyz.shape)
    q_new = Quaternion(
        w=jnp.where(mask_v, w_full, state.q.w),
        xyz=jnp.where(mask_v, xyz_full, state.q.xyz),
    )
    return replace(state, pos_c=pos_c_new, q=q_new)


def translate_clump(state: jd.State, clump_id: int, delta: jnp.ndarray) -> jd.State:
    mask = (state.clump_ID == clump_id)[..., None]
    return replace(state, pos_c=state.pos_c + mask * delta)


def clump_force(state: jd.State, clump_id: int) -> jnp.ndarray:
    idx = jnp.argmax((state.clump_ID == clump_id).astype(jnp.int32))
    return state.force[idx]


def make_single_clump(cfg: "SweepConfig", *, particle_radius: float, nv: int, particle_center: jnp.ndarray) -> jd.State:
    return geometricAsperityCreation.make_single_particle_3d(
        asperity_radius=cfg.asperity_radius,
        particle_radius=particle_radius,
        target_num_vertices=nv,
        aspect_ratio=cfg.mesh_aspect_ratio,
        add_core=cfg.add_core,
        particle_center=particle_center,
        mass=1.0,
        mesh_type=cfg.mesh_type,
    )


@partial(jax.jit, static_argnames=("tracer_id",))
def tracer_force(state: jd.State, system: jd.System, *, tracer_id: int) -> jnp.ndarray:
    st, _ = system.collider.compute_force(state, system)
    return clump_force(st, tracer_id)


@partial(jax.jit, static_argnames=("tracer_id",))
def normal_force_at_rad(
    state: jd.State,
    system: jd.System,
    *,
    tracer_id: int,
    base_com: jnp.ndarray,
    n_hat: jnp.ndarray,
    rad: jax.Array,
) -> jax.Array:
    mask_v = (state.clump_ID == tracer_id)[..., None]
    pos_c_new = jnp.where(mask_v, base_com + rad * n_hat, state.pos_c)
    st = replace(state, pos_c=pos_c_new)
    F = tracer_force(st, system, tracer_id=tracer_id)
    return jnp.dot(F, n_hat)


@partial(jax.jit, static_argnames=("tracer_id", "max_bracket_iter", "max_bisect_iter"))
def find_separation_for_normal_force(
    state: jd.State,
    system: jd.System,
    *,
    tracer_id: int,
    base_com: jnp.ndarray,
    n_hat: jnp.ndarray,
    rad0: float,
    fn_target: float,
    rad_rtol: float,
    max_bracket_iter: int,
    max_bisect_iter: int,
) -> float:
    n_hat = n_hat / (jnp.linalg.norm(n_hat) + 1e-30)

    def fn_at(rad):
        return normal_force_at_rad(state, system, tracer_id=tracer_id, base_com=base_com, n_hat=n_hat, rad=rad)

    rad_high = jnp.asarray(rad0, dtype=float)
    rad_low = jnp.asarray(rad0, dtype=float)
    step = jnp.asarray(1e-3, dtype=float)

    def bracket_cond(carry):
        rad_low, step, k = carry
        return jnp.logical_and(fn_at(rad_low) < fn_target, k < max_bracket_iter)

    def bracket_body(carry):
        rad_low, step, k = carry
        rad_low = jnp.maximum(rad_low - step, 0.0)
        return rad_low, step * 2.0, k + 1

    rad_low, _, _ = jax.lax.while_loop(bracket_cond, bracket_body, (rad_low, step, 0))

    def bisect_cond(carry):
        rad_low, rad_high, k = carry
        rel = (rad_high - rad_low) / jnp.maximum(rad_high, 1e-30)
        return jnp.logical_and(k < max_bisect_iter, rel > rad_rtol)

    def bisect_body(carry):
        rad_low, rad_high, k = carry
        mid = 0.5 * (rad_low + rad_high)
        rad_low, rad_high = jax.lax.cond(
            fn_at(mid) >= fn_target,
            lambda _: (mid, rad_high),
            lambda _: (rad_low, mid),
            operand=None,
        )
        return rad_low, rad_high, k + 1

    rad_low, _, _ = jax.lax.while_loop(bisect_cond, bisect_body, (rad_low, rad_high, 0))
    return rad_low


@dataclass
class SweepConfig:
    asperity_radius: float = 0.2
    add_core: bool = True
    base_radius: float = 0.5
    tracer_radius: float = 0.5
    base_nv: int = 20
    tracer_nv: int = 20
    mesh_aspect_ratio: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    e_int: float = 1.0
    dt: float = 1e-2
    rad_rtol: float = 1e-10
    max_bracket_iter: int = 80
    max_bisect_iter: int = 80
    fn_target: float = 1e-5
    mesh_type: str = "ico"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-theta", type=int, default=9)
    ap.add_argument("--n-phi", type=int, default=18)
    ap.add_argument("--theta-min", type=float, default=1e-3)
    ap.add_argument("--theta-max", type=float, default=np.pi - 1e-3)
    ap.add_argument("--out", type=str, default="surface_friction_sweep.npz")
    ap.add_argument("--fn-target", type=float, default=1e-2)
    ap.add_argument("--base-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--tracer-euler", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--tracer-euler-in-base-frame", action="store_true")
    args = ap.parse_args()

    cfg = SweepConfig(fn_target=args.fn_target)

    base = make_single_clump(cfg, particle_radius=cfg.base_radius, nv=cfg.base_nv, particle_center=jnp.zeros(3))
    tracer = make_single_clump(cfg, particle_radius=cfg.tracer_radius, nv=cfg.tracer_nv, particle_center=jnp.zeros(3))
    base = replace(base, fixed=jnp.ones(base.pos.shape[:-1], dtype=bool))
    tracer = translate_clump(tracer, 0, jnp.array([cfg.base_radius + cfg.tracer_radius + 0.5, 0.0, 0.0]))

    state = jd.State.merge(base, tracer)

    mats = [jd.Material.create("elastic", young=cfg.e_int, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    system = jd.System.create(
        state_shape=state.shape,
        dt=cfg.dt,
        linear_integrator_type="verlet",
        rotation_integrator_type="verletspiral",
        domain_type="free",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
    )

    q_base = euler_xyz_to_quaternion(jnp.array(args.base_euler))
    q_tracer_rel = euler_xyz_to_quaternion(jnp.array(args.tracer_euler))
    q_tracer = q_tracer_rel if not args.tracer_euler_in_base_frame else (q_base @ q_tracer_rel)

    base_id = 0
    tracer_id = 1

    base_com = state.pos_c[jnp.argmax((state.clump_ID == base_id).astype(jnp.int32))]
    tracer_com0 = state.pos_c[jnp.argmax((state.clump_ID == tracer_id).astype(jnp.int32))]

    state = set_clump_pose(state, base_id, pos_c=base_com, q=q_base)
    state = set_clump_pose(state, tracer_id, pos_c=tracer_com0, q=q_tracer)

    thetas = np.linspace(args.theta_min, args.theta_max, args.n_theta)
    phis = np.linspace(0.0, 2.0 * np.pi, args.n_phi, endpoint=False)

    mu_out = np.zeros((args.n_theta, args.n_phi), dtype=float)
    fn_out = np.zeros_like(mu_out)
    ft_out = np.zeros_like(mu_out)
    rad_out = np.zeros_like(mu_out)

    rad_guess = float(cfg.base_radius + cfg.tracer_radius)

    for it, theta in enumerate(thetas):
        for ip, phi in enumerate(phis):
            n_hat = q_base.rotate(q_base, spherical_unit(theta, phi))

            rad_star = find_separation_for_normal_force(
                state,
                system,
                tracer_id=tracer_id,
                base_com=base_com,
                n_hat=n_hat,
                rad0=rad_guess,
                fn_target=cfg.fn_target,
                rad_rtol=cfg.rad_rtol,
                max_bracket_iter=cfg.max_bracket_iter,
                max_bisect_iter=cfg.max_bisect_iter,
            )
            rad_guess = float(rad_star)

            mask_v = (state.clump_ID == tracer_id)[..., None]
            pos_c_new = jnp.where(mask_v, base_com + rad_star * n_hat, state.pos_c)
            st = replace(state, pos_c=pos_c_new)

            F = tracer_force(st, system, tracer_id=tracer_id)
            Fn = jnp.dot(F, n_hat)
            Ft = jnp.linalg.norm(F - Fn * n_hat)
            mu = Ft / jnp.maximum(jnp.abs(Fn), 1e-30)

            mu_out[it, ip] = float(np.asarray(mu))
            fn_out[it, ip] = float(np.asarray(Fn))
            ft_out[it, ip] = float(np.asarray(Ft))
            rad_out[it, ip] = float(np.asarray(rad_star))

    np.savez(
        args.out,
        thetas=thetas,
        phis=phis,
        mu=mu_out,
        Fn=fn_out,
        Ft=ft_out,
        rad=rad_out,
        base_euler=np.asarray(args.base_euler),
        tracer_euler=np.asarray(args.tracer_euler),
        tracer_euler_in_base_frame=np.asarray(bool(args.tracer_euler_in_base_frame)),
        cfg=cfg.__dict__,
    )
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()