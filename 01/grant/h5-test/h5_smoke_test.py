# SPDX-License-Identifier: BSD-3-Clause
"""
Smoke test for jaxdem.utils.h5.

Run:
  python tests/h5_smoke_test.py
"""

from __future__ import annotations

import os
import tempfile
import warnings

try:
    import jax  # type: ignore
    import jax.numpy as jnp  # type: ignore
except ModuleNotFoundError as e:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _JAX_IMPORT_ERROR = e
else:
    _JAX_IMPORT_ERROR = None

try:
    import h5py  # type: ignore
except ModuleNotFoundError as e:
    h5py = None  # type: ignore[assignment]
    _H5PY_IMPORT_ERROR = e
else:
    _H5PY_IMPORT_ERROR = None

if jax is not None and h5py is not None:
    import jaxdem as jd
    from jaxdem.colliders import NeighborList
    from jaxdem.utils import h5


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def make_state_system():
    # Small system with nontrivial neighbor interactions
    pos = jnp.array(
        [
            [0.00, 0.00],
            [0.20, 0.00],
            [0.00, 0.20],
            [0.20, 0.20],
            [0.10, 0.10],
        ],
        dtype=float,
    )
    state = jd.State.create(pos=pos, rad=jnp.ones((pos.shape[0],), dtype=float) * 0.05)

    system = jd.System.create(
        state_shape=state.shape,
        dt=0.01,
        domain_type="free",
        collider_type="naive",
        force_model_type="spring",
    )

    # Swap in NeighborList collider (needs state at construction time)
    system.collider = NeighborList.Create(
        state,
        cutoff=0.5,
        box_size=system.domain.box_size,
        skin=0.1,
        max_neighbors=10,
    )

    # Add a non-serializable callable (should be skipped with a warning)
    system.force_manager = jd.ForceManager.create(
        state.shape,
        force_functions=(
            # typical "closure/lambda" user force
            (lambda pos, st, sys: (jnp.zeros_like(st.force), jnp.zeros_like(st.torque))),
        ),
    )

    # Should be runnable pre-save
    state, system = system.step(state, system, n=1)
    return state, system


def test_round_trip_system_state(tmp_path: str) -> None:
    state, system = make_state_system()

    path = os.path.join(tmp_path, "bundle.h5")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        h5.save((state, system), path)
        _assert(any("skipping callable" in str(x.message) for x in w), "expected callable-skip warning")

    loaded = h5.load(path)
    _assert(isinstance(loaded, tuple) and len(loaded) == 2, "expected tuple(state, system)")
    st2, sys2 = loaded

    _assert(isinstance(st2.pos_c, jax.Array), "expected loaded State arrays as jax.Array")

    # Critical: static dataclass field must be a Python int (not a jax scalar)
    _assert(isinstance(sys2.collider.max_neighbors, int), "NeighborList.max_neighbors must be a Python int")

    # Callable field should have been skipped => defaulted to empty tuple on load
    _assert(
        getattr(sys2.force_manager, "force_functions", ()) == (),
        "expected force_functions to be empty tuple after skipping callables",
    )

    # Should be runnable post-load (and not crash due to non-hashable static args)
    st3, sys3 = sys2.step(st2, sys2, n=1)
    _assert(st3.pos_c.shape == st2.pos_c.shape, "step should preserve state shape")
    _assert(isinstance(sys3.collider.max_neighbors, int), "max_neighbors must remain Python int after stepping")


def test_schema_warnings_unknown_and_missing(tmp_path: str) -> None:
    if h5py is None or jax is None:
        raise RuntimeError(
            "jax and h5py are required to run this smoke test. Install them "
            "(e.g. `pip install jax h5py`) and re-run."
        ) from (_H5PY_IMPORT_ERROR or _JAX_IMPORT_ERROR)

    # Save a State, then mutate the HDF5 to create:
    # - an unknown field
    # - a missing field
    state = jd.State.create(pos=jnp.zeros((3, 2), dtype=float))

    path = os.path.join(tmp_path, "state_mutated.h5")
    h5.save(state, path)

    with h5py.File(path, "a") as f:
        root = f["root"]
        # add unknown field
        ds = root.create_dataset("extra_field_not_in_class", data=123)
        ds.attrs["__kind__"] = "scalar"
        # delete a known field to force missing-field default
        if "vel" in root:
            del root["vel"]

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        st2 = h5.load(path, warn_missing=True, warn_unknown=True)

        msgs = "\n".join(str(x.message) for x in w)
        _assert("unknown saved fields" in msgs, "expected unknown-field warning")
        _assert("missing saved fields" in msgs, "expected missing-field warning")

    _assert(st2.is_valid, "loaded State should still be valid after missing-field defaulting")


def test_dp_round_trip_manual_reattach(tmp_path: str) -> None:
    """
    DP (DeformableParticleContainer) test with the intended workflow:
    - DP container is saved explicitly
    - Force/energy callables are *not* serialized (skipped with warning)
    - After load, recreate callables from the loaded container and reattach
    - Forces/torques should match
    """
    if h5py is None or jax is None:
        raise RuntimeError(
            "jax and h5py are required to run this smoke test. Install them "
            "(e.g. `pip install jax h5py`) and re-run."
        ) from (_H5PY_IMPORT_ERROR or _JAX_IMPORT_ERROR)

    # Reference (stress-free) configuration for the DP container.
    ref_pos = jnp.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )

    # Deformed configuration used in the State (so DP edges are stretched).
    cur_pos = ref_pos.at[1, 0].set(1.05)  # pull one vertex in +x

    state = jd.State.create(pos=cur_pos, rad=jnp.ones((cur_pos.shape[0],), dtype=float) * 0.01)

    # Use a collider/model that contributes ~0 contact force for this configuration,
    # so we isolate DP forces.
    system = jd.System.create(
        state_shape=state.shape,
        dt=0.01,
        domain_type="free",
        collider_type="naive",
        force_model_type="spring",
    )

    # Build a simple edge-spring DP container from the reference configuration.
    # In 2D, "elements" would be segments; for edge-length energy we only need edges.
    edges = jnp.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=int)
    dp = jd.DeformableParticleContainer.create(
        vertices=ref_pos,
        edges=edges,
        edges_ID=jnp.zeros((edges.shape[0],), dtype=int),
        el=jnp.array([10.0], dtype=float),
    )

    dp_force_fn, dp_energy_fn = jd.DeformableParticleContainer.create_force_energy_functions(dp)

    # Attach DP as an external force/energy term (callables are intentionally not serialized by h5)
    system.force_manager = jd.ForceManager.create(
        state.shape,
        force_functions=((dp_force_fn, dp_energy_fn, False),),
    )

    # Compute forces once (collider -> force_manager)
    stA, sysA = system.collider.compute_force(state, system)
    stA, sysA = sysA.force_manager.apply(stA, sysA)
    FA, TA = stA.force, stA.torque
    EA = sysA.force_manager.compute_potential_energy(stA, sysA)

    # Ensure this isn't a trivial "all zeros" test: DP must produce a measurable signal.
    # (We deform one vertex of a square wireframe, so edge-length energy should be > 0.)
    _assert(jnp.max(jnp.abs(FA)) > 1e-10, "DP force unexpectedly ~0; test is not exercising DP mechanics")
    _assert(jnp.max(jnp.abs(EA)) > 1e-12, "DP energy unexpectedly ~0; test is not exercising DP mechanics")

    # Save bundle including DP container (callables will be skipped in the System)
    path = os.path.join(tmp_path, "dp_bundle.h5")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # IMPORTANT: compute_force/apply are JIT'd with donate_argnames, so the original
        # (state, system) inputs may be invalidated. Always save the returned objects.
        h5.save((stA, sysA, dp), path)
        _assert(any("skipping callable" in str(x.message) for x in w), "expected callable-skip warning")

    loaded = h5.load(path)
    _assert(isinstance(loaded, tuple) and len(loaded) == 3, "expected tuple(state, system, dp)")
    st2, sys2, dp2 = loaded

    # Re-attach DP forces manually after load (expected workflow)
    dp_force_fn2, dp_energy_fn2 = jd.DeformableParticleContainer.create_force_energy_functions(dp2)
    sys2.force_manager = jd.ForceManager.create(
        st2.shape,
        force_functions=((dp_force_fn2, dp_energy_fn2, False),),
    )

    stB, sysB = sys2.collider.compute_force(st2, sys2)
    stB, sysB = sysB.force_manager.apply(stB, sysB)
    FB, TB = stB.force, stB.torque
    EB = sysB.force_manager.compute_potential_energy(stB, sysB)

    # Forces should match (within numerical precision).
    _assert(jnp.allclose(FA, FB, atol=1e-8, rtol=1e-7), "DP forces mismatch after save/load + reattach")
    _assert(jnp.allclose(TA, TB, atol=1e-8, rtol=1e-7), "DP torques mismatch after save/load + reattach")
    _assert(jnp.allclose(EA, EB, atol=1e-10, rtol=1e-8), "DP energy mismatch after save/load + reattach")

    # And remain non-trivial after reattach.
    _assert(jnp.max(jnp.abs(FB)) > 1e-10, "DP force after reattach unexpectedly ~0")
    _assert(jnp.max(jnp.abs(EB)) > 1e-12, "DP energy after reattach unexpectedly ~0")


def main() -> None:
    if jax is None:
        raise SystemExit(
            "jax is not installed in this environment. Install it (e.g. `pip install jax`) "
            "and re-run this test."
        )
    if h5py is None:
        raise SystemExit(
            "h5py is not installed in this environment. Install it (e.g. `pip install h5py`) "
            "and re-run this test."
        )
    with tempfile.TemporaryDirectory() as td:
        test_round_trip_system_state(td)
        test_schema_warnings_unknown_and_missing(td)
        test_dp_round_trip_manual_reattach(td)
    print("h5_smoke_test: OK")


if __name__ == "__main__":
    main()

