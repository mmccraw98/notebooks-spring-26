import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import jax.numpy as jnp

from jaxdem.analysis import LagBinsPseudoLog, evaluate_binned, LagBinsLinear

from jaxdem_scripts.correlations import get_pseudo_log_bins_from_steps

phi_norm = plt.Normalize(0.6, 1.0)
cmap = plt.cm.viridis

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("name")
args = parser.parse_args()
name = args.name

dt = 1e-3

data_root = f'/home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-5/compression-4/{name}'

for phi_dir in os.listdir(data_root):
    corrs = np.load(os.path.join(data_root, phi_dir, 'corrs.npz'))
    phi_float = float(phi_dir.split('phi-')[-1])
    plt.plot(corrs['t'], corrs['isf_vertex_small'], c=cmap(phi_norm(phi_float)))

plt.xscale('log')
    
plt.tight_layout()
os.makedirs('corr-figures-4', exist_ok=True)
plt.savefig(f'corr-figures-4/isf.png')