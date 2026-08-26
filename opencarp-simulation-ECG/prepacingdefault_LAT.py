# Genera LAT file di zeri per la mesh per inizializzare il prepacing
import os

import numpy as np

CASE_NAME = "sb4201"
pts_file = os.path.join(".", CASE_NAME+'_reg', CASE_NAME + '_meshes', "sb4201_500um_tagged_fastendo_reg.pts")
with open(pts_file) as file:
    n_nodes = int(file.readline().strip())

lats = np.zeros(n_nodes)
np.savetxt(os.path.join(".", CASE_NAME+'_reg', CASE_NAME + '_meshes', f'{CASE_NAME}_latmap_zero.dat'), lats, fmt='%.6f')
print(f"LAT file generato: {n_nodes} nodi, tutti a t=0 ms")