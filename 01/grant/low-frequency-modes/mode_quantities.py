import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy as sp

data = np.load('hessian.npz')

H = data['H']
M = data['M']

N_c = H.shape[0] // 3

# vals, vecs = sp.linalg.eigh(H, M)  # mass orthonormalization

# DO NOT USE THE MASS ORTHONORMALIZATION FOR THE FOLLOWING:
vals, vecs = sp.linalg.eigh(H)
modes = vecs.T.reshape(-1, N_c, 3)

# verify the unit norm
assert jnp.allclose(np.sum(modes ** 2, axis=(-1, -2)), 1.0)

# calculate the translational and rotational content
trans = jnp.sum(modes[..., :-1] ** 2, axis=(-1, -2))
rot = 1 - trans

# calculate the participation ratios
P = 1.0 / (N_c * jnp.sum(jnp.sum(modes ** 2, axis=-1) ** 2, axis=-1))
P_x = jnp.sum(modes[..., 0] ** 2, axis=(-1)) ** 2 / (N_c * jnp.sum(modes[..., 0] ** 4, axis=(-1)))
P_y = jnp.sum(modes[..., 1] ** 2, axis=(-1)) ** 2 / (N_c * jnp.sum(modes[..., 1] ** 4, axis=(-1)))
P_t = jnp.sum(modes[..., 2] ** 2, axis=(-1)) ** 2 / (N_c * jnp.sum(modes[..., 2] ** 4, axis=(-1)))
P_xy = jnp.sum(modes[..., :-1] ** 2, axis=(-1, -2)) ** 2 / (N_c * jnp.sum(modes[..., :-1] ** 4, axis=(-1, -2)))

# calculate the optical order parameter
t = modes[..., -1]
Q_opt = jnp.sum(t[:, :, None] * t[:, None, :], axis=(-1, -2)) / (N_c * jnp.sum(modes[..., 2] ** 2, axis=(-1)))