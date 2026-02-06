import jax.numpy as jnp
import jax
import jaxdem as jd
jax.config.update("jax_enable_x64", True)
import numpy as np
import os
import argparse
import sys
from file_management import load_arrs

if __name__ == "__main__":
    path = '/Users/marshallmccraw/Downloads/phi-0.537377'
    state = jd.utils.h5.load(os.path.join(path, 'final', 'state.h5'))
    system = jd.utils.h5.load(os.path.join(path, 'final', 'system.h5'))

    lut = np.empty(state.unique_ID.size, dtype=state.deformable_ID.dtype)
    lut[state.unique_ID] = state.deformable_ID

    data = load_arrs(os.path.join(path, 'traj', 'data.h5'))
    pos = data['pos']
    unique_ID = data['unique_ID']
    n_frames = pos.shape[0]
    rad = np.array([state.rad for _ in range(n_frames)])
    deformable_ID = lut[unique_ID]
    box_size = np.array([system.domain.box_size for _ in range(n_frames)])

    M = int(jnp.max(deformable_ID[0]) + 1)  # or known constant
    total_pos = jax.vmap(lambda p, d: jax.ops.segment_sum(p, d, num_segments=M))(pos, deformable_ID)  # (S, M, dim)
    dp_counts = jax.vmap(lambda d: jax.ops.segment_sum(jnp.ones((d.shape[0],), dtype=pos.dtype), d, num_segments=M))(deformable_ID)  # (S, M)
    dp_com = total_pos / jnp.maximum(dp_counts[..., None], 1.0)  # (S, M, dim)

    print(dp_com.shape)