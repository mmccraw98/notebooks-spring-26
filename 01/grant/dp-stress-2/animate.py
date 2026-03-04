import os

import numpy as np

import jaxdem as jd

import jax.numpy as jnp

from anim_utils import animate_from_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("name")
args = parser.parse_args()
name = args.name

root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-5/compression-3/{name}'

for phi_dir in os.listdir(root):
    data = np.load(os.path.join(root, phi_dir, 'traj.npz'))

    with jd.CheckpointLoader(directory=os.path.join(root, phi_dir, 'init')) as loader:
        state, system = loader.load()
    order = jnp.argsort(state.unique_id)

    n_frames = data['pos'].shape[0]
    rad = jnp.repeat(state.rad[order][None], n_frames, axis=0)
    dpid = jnp.repeat(state.bond_id[order][None], n_frames, axis=0)
    box_size = jnp.repeat(system.domain.box_size[None], n_frames, axis=0)

    os.makedirs(f'anims-3/{name}', exist_ok=True)
    animate_from_data(data['pos'], rad, dpid, box_size, f'anims-3/{name}/{phi_dir}.gif', frames=100, fps=15)
    