import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
import jax.numpy as jnp
import os

import numpy as np
import matplotlib.pyplot as plt

root = os.path.join('data', '010_10')

delta_phis = np.logspace(-8, -1, 10)
pes = np.zeros_like(delta_phis)

for i, delta_phi in enumerate(delta_phis):
    state = jd.utils.h5.load(os.path.join(root, 'state.h5'))
    system = jd.utils.h5.load(os.path.join(root, 'system.h5'))

    phi = jd.utils.packingUtils.compute_packing_fraction(state, system)
    state, system = jd.utils.packingUtils.scale_to_packing_fraction(state, system, phi + delta_phi)
    state, system = system.collider.compute_force(state, system)
    pes[i] = jnp.sum(system.collider.compute_potential_energy(state, system))
    # pes[i] = jnp.sum(jnp.linalg.norm(state.force, axis=-1))

print(delta_phis)
print(pes)

plt.plot(delta_phis, delta_phis ** 2)

plt.plot(delta_phis, pes)
plt.xscale('log')
plt.yscale('log')
plt.savefig('figures/pe-scaling.png')
plt.close()