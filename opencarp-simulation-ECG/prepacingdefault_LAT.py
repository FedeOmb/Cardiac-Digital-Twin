# Genera LAT file di zeri per la mesh per inizializzare il prepacing
import numpy as np

pts_file = './sb301_meshes/sb301_fine500um_opencarp.pts'
with open(pts_file) as f:
    n_nodes = int(f.readline().strip())

lats = np.zeros(n_nodes)
np.savetxt('./sb301_meshes/sb301_latmap_zero.dat', lats, fmt='%.6f')
print(f"LAT file generato: {n_nodes} nodi, tutti a t=0 ms")