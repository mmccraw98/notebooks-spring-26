import subprocess
import sys
import numpy as np
import os
from uuid import uuid4

root = "/home/mmccraw/dev/data/26-01-01/grant/friction-surface-thomson"

n_asperity_points = 100
particle_radius = 0.5
subdivisions = 6
n_theta = 10
n_phi = 10
batch_size = 10_000
N_steps = 0

for nv in [20, 50]:
    for asperity_radius in np.linspace(0.01, particle_radius, n_asperity_points + 1)[:-1]:
        cmd = [
            sys.executable, "calculate_surface_friction.py",
            "--output_directory", os.path.join(root, N_steps, str(nv), str(uuid4())),
            "--nv", str(nv),
            "--particle_radius", str(particle_radius),
            "--asperity_radius", str(asperity_radius),
            "--n_theta", str(n_theta),
            "--n_phi", str(n_phi),
            "--batch_size", str(batch_size),
            "--subdivisions", str(subdivisions),
            "--N_steps", N_steps,
        ]
        subprocess.run(cmd, check=True)
