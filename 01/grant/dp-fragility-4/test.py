import jaxdem as jd
import jax.numpy as jnp
import numpy as np
import os
from file_management import load_arrs

from utils import reorder_state

import matplotlib.pyplot as plt

root = f'//home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-4-old/hard/phi-0.595423/'
data = load_arrs(os.path.join(root, 'traj', 'data.h5'))
plt.plot(data['t'], data['isf'])

root = f'//home/mmccraw/dev/data/26-01-01/grant/dp-fragilitiy/version-4/hard/phi-0.595423/'
data = load_arrs(os.path.join(root, 'traj', 'data.h5'))
plt.plot(data['t'], data['isf_small'])
plt.plot(data['t'], data['isf_vertex_small'])

plt.xscale('log')
plt.savefig('isf.png')
plt.close()
