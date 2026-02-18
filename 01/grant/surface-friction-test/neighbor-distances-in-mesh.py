from matplotlib.colors import LogNorm
import trimesh
import numpy as np
import matplotlib.pyplot as plt

p_rad = 0.5
n_arad_points = 1000
max_subs = 5

meshes = [trimesh.creation.icosphere(subs, radius=p_rad) for subs in range(max_subs)]
nvs = [mesh.vertices.shape[0] for mesh in meshes]

cmap = plt.cm.viridis
nv_norm = LogNorm(min(nvs), max(nvs))

for nv, mesh in zip(nvs, meshes):
    edge_lengths = np.linalg.norm(np.diff(mesh.vertices[mesh.edges], axis=1).squeeze(), axis=-1)
    plt.scatter(nv, np.mean(edge_lengths), c='k')

n = np.logspace(1, 3)
plt.plot(n, 1.5 / np.sqrt(n), c='k', ls='--', label=r'$\langle L \rangle \sim \sqrt{N_v}$')

plt.xlabel(r'$N_v$', fontsize=16)
plt.ylabel(r'$\langle L \rangle$', fontsize=16)

plt.xscale('log')
plt.yscale('log')
plt.savefig('figures-edge-length-scaling/ico-edges.png', dpi=600)
plt.close()