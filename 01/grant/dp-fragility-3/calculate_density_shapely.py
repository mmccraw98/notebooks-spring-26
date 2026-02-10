import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import jaxdem as jd
import os
from jaxdem.forces.force_manager import ForceManager

from shapely.ops import unary_union
from shapely import Point, Polygon

from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

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

def compute_dp_area(args):
    pos_masked, rad_masked, quad_segs = args
    shape = unary_union(
        [Point(v).buffer(distance=r, quad_segs=quad_segs) for v, r in zip(pos_masked, rad_masked)]
        + [Polygon(pos_masked)]
    )
    return shape.area

def get_area_for_frame(pos, rad, deformable_ID, N, box_size, quad_segs=1e4):
    dp_args = []
    for dpid in range(N):
        mask = deformable_ID == dpid
        dp_args.append((np.asarray(pos[mask]), np.asarray(rad[mask]), quad_segs))
    with ProcessPoolExecutor() as pool:
        area = sum(tqdm(pool.map(compute_dp_area, dp_args), total=N))
    return area / jnp.prod(box_size)


if __name__ == "__main__":
    input_path = '/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-3/floppy/phi-0.537377'

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

    n_steps = 10_000
    save_stride = 500
    n_snapshots = n_steps // save_stride
    state, system, (state_traj, system_traj) = system.trajectory_rollout(
        state, system, n=n_snapshots, stride=save_stride
    )
    state_traj = jax.vmap(reorder_state)(state_traj)

    N = int(max(state.deformable_ID) + 1)

    phis = []
    for fid in range(state_traj.pos.shape[0]):
        phis.append(get_area_for_frame(
            state_traj.pos[fid],
            state_traj.rad[fid],
            state_traj.deformable_ID[fid],
            N,
            system_traj.domain.box_size[fid],
            quad_segs=1e4,
        ))

    import matplotlib.pyplot as plt
    plt.plot(phis)
    plt.savefig('phis.png')
    plt.close()

    np.savez(
        os.path.join(input_path, 'traj', 'density.npz'),
        phi=phis,
    )