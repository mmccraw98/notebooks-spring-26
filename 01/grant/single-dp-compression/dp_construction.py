from functools import partial
import jax
import numpy as np
import jax.numpy as jnp
import jaxdem as jd
from jaxdem.forces.deformable_particle import angle_between_normals
from jaxdem.utils.geometricAsperityCreation import generate_asperities_2d, _randomize_orientation
import trimesh

jax.config.update("jax_enable_x64", True)

asperity_radius = 0.2
particle_radius = 0.5
num_vertices = 20

seed = None
random_orientation = True
particle_center = jnp.zeros(2)
aspect_ratio = 1.0
core_type = None
use_uniform_mesh = True

pts, rads = generate_asperities_2d(
    asperity_radius=asperity_radius,
    particle_radius=particle_radius,
    num_vertices=num_vertices,
    aspect_ratio=aspect_ratio,
    core_type=core_type,
    use_uniform_mesh=use_uniform_mesh,
)

pts = jnp.asarray(pts, dtype=float) + jnp.asarray(particle_center, dtype=float)
rads = jnp.asarray(rads, dtype=float)

if random_orientation:
    import numpy as np

    if seed is None:
        seed = int(np.random.randint(0, 1_000_000_000))
    pts = _randomize_orientation(pts, key=jax.random.PRNGKey(seed))

print(pts)