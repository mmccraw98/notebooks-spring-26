import jaxdem as jd
import jax.numpy as jnp
import numpy as np
import os
from file_management import load_arrs

from utils import reorder_state

import matplotlib.pyplot as plt

for rollout_type in ['dynamic-rollout', 'linear-rollout']:
    root = f'/home/mmccraw/dev/data/26-01-01/grant/ga-fragilitiy/version-4/100_10/{rollout_type}/phi-0.557325'
    data = load_arrs(os.path.join(root, 'traj', 'data.h5'))
    plt.plot(data['t'], data['aisf_small'])
plt.xscale('log')
plt.savefig('isf.png')
plt.close()
