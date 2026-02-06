import os
import h5py

def make_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_data_dir(path):
    paths = {name: None for name in ['init', 'final', 'traj']}
    for name in paths.keys():
        loc = os.path.join(path, name)
        make_if_not_exists(loc)
        paths[name] = loc
    return paths

def save_arrs(arrs, arr_names, path):
    with h5py.File(path, 'w') as f:
        for arr, name in zip(arrs, arr_names):
            f.create_dataset(name, arr.shape, arr.dtype, arr)

def load_arrs(path):
    with h5py.File(path, 'r') as f:
        return {name: f[name][()] for name in f.keys()}
