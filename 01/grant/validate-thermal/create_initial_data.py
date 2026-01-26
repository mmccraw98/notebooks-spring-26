import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import jaxdem as jd
from bump_utils import create_clumps_2d, create_clumps_3d, create_spheres
import os

if __name__ == "__main__":
    data_root = '/home/mmccraw/dev/data/26-01-01/grant/validate-thermal/initial-data'
    if not os.path.exists(data_root):
        os.makedirs(data_root)
    
    def make2d():
        path = os.path.join(data_root, '2d-clumps')
        if not os.path.exists(path):
            os.makedirs(path)
        N = 100
        nv = 20
        mu = 0.1
        phi = 0.7
        state, system = create_clumps_2d(phi, N, mu, 1.0, nv, 1.0)
        jd.utils.h5.save(state, os.path.join(path, 'state.h5'))
        jd.utils.h5.save(system, os.path.join(path, 'system.h5'))

    def make3d():
        path = os.path.join(data_root, '3d-clumps')
        if not os.path.exists(path):
            os.makedirs(path)
        N = 100
        nv = 20
        vrad = 0.3
        phi = 0.5
        state, system = create_clumps_3d(phi, N, vrad, 1.0, nv, 1.0)
        jd.utils.h5.save(state, os.path.join(path, 'state.h5'))
        jd.utils.h5.save(system, os.path.join(path, 'system.h5'))

    def make2d_spheres():
        path = os.path.join(data_root, '2d-spheres')
        if not os.path.exists(path):
            os.makedirs(path)
        N = 100
        phi = 0.7
        state, system = create_spheres(phi, N, 2, 1.0)
        jd.utils.h5.save(state, os.path.join(path, 'state.h5'))
        jd.utils.h5.save(system, os.path.join(path, 'system.h5'))

    def make3d_spheres():
        path = os.path.join(data_root, '3d-spheres')
        if not os.path.exists(path):
            os.makedirs(path)
        N = 100
        phi = 0.5
        state, system = create_spheres(phi, N, 3, 1.0)
        jd.utils.h5.save(state, os.path.join(path, 'state.h5'))
        jd.utils.h5.save(system, os.path.join(path, 'system.h5'))

    make2d()
    make3d()
    make2d_spheres()
    make3d_spheres()