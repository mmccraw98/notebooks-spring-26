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

import matplotlib.pyplot as plt

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


def point_in_polygon(point, segments):
    """
    Test if a 2D point is inside a closed polygon defined by line segments.
    
    Parameters
    ----------
    point : jax.Array, shape (2,)
        The query point.
    segments : jax.Array, shape (M, 2, 2)
        Polygon edges. segments[i, 0] and segments[i, 1] are the two endpoints.
    
    Returns
    -------
    bool
        True if the point is inside the polygon.
    """
    px, py = point[0], point[1]
    
    # Start and end of each edge
    x1, y1 = segments[:, 0, 0], segments[:, 0, 1]
    x2, y2 = segments[:, 1, 0], segments[:, 1, 1]
    
    # Does a horizontal ray from (px, py) to the right cross this edge?
    # Condition 1: the edge straddles the y-coordinate of the point
    cond_y = (y1 <= py) != (y2 <= py)
    
    # Condition 2: the intersection x-coordinate is to the right of the point
    # x_intersect = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
    t = (py - y1) / jnp.where(y2 == y1, 1.0, y2 - y1)
    x_intersect = x1 + t * (x2 - x1)
    cond_x = px < x_intersect
    
    # Count crossings (odd = inside)
    crossings = jnp.sum(cond_y & cond_x)
    return crossings % 2 == 1



def compute_dp_area(args):
    pos_masked, rad_masked, quad_segs = args
    shape = unary_union(
        [Point(v).buffer(distance=r, quad_segs=quad_segs) for v, r in zip(pos_masked, rad_masked)]
        + [Polygon(pos_masked)]
    )
    return shape.area


def get_area_for_frame(pos, rad, deformable_ID, quad_segs=1e4):
    dp_args = []
    for dpid in range(N):
        mask = deformable_ID == dpid
        dp_args.append((np.asarray(pos[mask]), np.asarray(rad[mask]), quad_segs))
    with ProcessPoolExecutor() as pool:
        area = sum(tqdm(pool.map(compute_dp_area, dp_args), total=N))
    return area / jnp.prod(box_size)


if __name__ == "__main__":
    # root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-density'
    
    # input_path = '/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-3/floppy/phi-0.537377'

    # # load the data
    # data_root = os.path.dirname(input_path)
    # state = jd.utils.h5.load(os.path.join(input_path, 'final', 'state.h5'))
    # system = jd.utils.h5.load(os.path.join(input_path, 'final', 'system.h5'))
    # dp = jd.utils.h5.load(os.path.join(input_path, 'final', 'dp.h5'))
    # dp_force, dp_energy = dp.create_force_energy_functions(dp)
    # system.force_manager = ForceManager.create(
    #     state_shape=state.shape,
    #     gravity=None,
    #     force_functions=[(dp_force, dp_energy, False)],
    # )

    # n_steps = 10_000
    # save_stride = 100
    # n_snapshots = n_steps // save_stride
    # state, system, (state_traj, system_traj) = system.trajectory_rollout(
    #     state, system, n=n_snapshots, stride=save_stride
    # )

    # density = os.path.basename(input_path)
    # which = os.path.basename(os.path.dirname(input_path))

    # output_path = os.path.join(root, which, density)
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    # jd.utils.h5.save(state_traj, os.path.join(output_path, 'state_traj.h5'))
    # jd.utils.h5.save(system_traj, os.path.join(output_path, 'system_traj.h5'))

    root = '/Users/marshallmccraw/Projects/yale/data/s-26/grant/dp-density'
    state_traj = jd.utils.h5.load(os.path.join(root, 'state_traj.h5'))
    state_traj = jax.vmap(reorder_state)(state_traj)
    system_traj = jd.utils.h5.load(os.path.join(root, 'system_traj.h5'))
    dp = jd.utils.h5.load(os.path.join(root, 'dp.h5'))


    fid = 0
    pos = state_traj.pos[fid]
    rad = state_traj.rad[fid]
    box_size = system_traj.domain.box_size[0]

    S_total = int(1e4)
    chunk = int(1e4)
    bounds = box_size
    dim = state_traj.dim
    seed = np.random.randint(0, 1e9)
    key0 = jax.random.PRNGKey(seed)
    n_steps = S_total // chunk
    assert (S_total % chunk) == 0
    N = int(jnp.max(dp.elements_ID) + 1)
    rad2 = rad[:, None] ** 2

    system = jd.System.create(
        state_shape=pos.shape,
        domain_type="periodic",
        domain_kw=dict(
            box_size=box_size,
        ),
        # collider_type="neighborlist",
        # collider_kw=dict(
        #     state=state,
        #     cutoff=2.0 * jnp.max(state.rad),
        #     skin=0.05,
        #     safety_factor=5.0,
        # ),
    )

    # JUST USE SHAPELY
    phis = []
    print(state_traj.pos.shape[0])
    # for fid in range(state_traj.pos.shape[0]):
    for fid in range(20):
        phis.append(get_area_for_frame(
            state_traj.pos[fid],
            state_traj.rad[fid],
            state_traj.deformable_ID[fid],
            quad_segs=1e4
        ))
        print(phis[-1])
    plt.plot(phis)
    plt.savefig('densities.png')
    plt.close()


    # i = 0
    # hits = 0

    # key_i = jax.random.fold_in(key0, i)
    # samples = jax.random.uniform(key_i, shape=(chunk, dim)) * bounds

    # # check intersection with polygon interiors
    # def check_single_dp(dp_elements):
    #     segments = pos[dp_elements]
    #     return jax.vmap(point_in_polygon, in_axes=(0, None))(samples, segments)

    # inside_polygons = jnp.zeros(chunk, dtype=bool)
    # dpids, offsets, counts = jnp.unique(dp.elements_ID, return_index=True, return_counts=True)
    # for count in jnp.unique(counts):
    #     count = counts[0]
    #     mask = counts == count 
    #     batched_elements = jnp.array([
    #         dp.elements[o:o + count] for o in offsets[mask]
    #     ])
    #     inside_polygons += jnp.any(jax.vmap(check_single_dp)(batched_elements), axis=0)
    
    # # check intersection with sphere interiors
    # disp = system.domain.displacement(
    #     pos[:, None, :],
    #     samples[None, :, :],
    #     system
    # )
    # r2 = jnp.sum(disp * disp, axis=-1)
    # inside_spheres = jnp.any(r2 <= rad2, axis=0)

    # hits += jnp.sum(inside_spheres + inside_polygons)

    # print(hits)
