import subprocess
import sys
import numpy as np
import os

root = "/home/mmccraw/dev/data/26-01-01/grant/friction-surface"

n_asperity_points = 100
particle_radius = 0.5
subdivisions = 6
n_theta = 10
n_phi = 10
batch_size = 10_000
mesh_type = "octa"

# for nv in [13, 43, 93]:  # for iso
for nv in [19, 39, 103]:  # for octa
    for asperity_radius in np.linspace(0.01, particle_radius, n_asperity_points + 1)[:-1]:
        cmd = [
            sys.executable, "calculate_surface_friction.py",
            "--output_directory", os.path.join(root, mesh_type, str(nv)),
            "--nv", str(nv),
            "--particle_radius", str(particle_radius),
            "--asperity_radius", str(asperity_radius),
            "--n_theta", str(n_theta),
            "--n_phi", str(n_phi),
            "--batch_size", str(batch_size),
            "--subdivisions", str(subdivisions),
            "--mesh_type", mesh_type,
        ]
        subprocess.run(cmd, check=True)
