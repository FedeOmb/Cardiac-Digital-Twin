import os
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

merged_mesh_path = os.path.join(".", "sb3901","sb3901_torso", "sb3901_finalmeshum_torsobiv_v2.vtk")
merged_mesh = pv.read(merged_mesh_path)

# Estrai punti e tag delle celle
merged_pts = np.array(merged_mesh.points)            
cell_tags  = np.array(merged_mesh.cell_data["elemTag"]) 

# --- Filtra le celle miocardiche (tag 1,2,3,4) ---
myo_cell_ids = np.where((cell_tags >= 1) & (cell_tags <= 4))[0]

# Raccogli i nodi unici di quelle celle usando pyvista threshold
myo_submesh = merged_mesh.extract_cells(myo_cell_ids)
# extract_cells mantiene i punti originali → usa point_data per risalire agli indici globali
# Il mapping originale si ottiene dai punti della submesh vs merged_pts

# Costruisci KD-tree dei punti miocardici (è più sicuro lavorare direttamente
# sui punti della submesh e mappare per coordinate)
myo_pts   = np.array(myo_submesh.points)
tree_myo  = cKDTree(myo_pts)

# Indici globali dei nodi miocardici: cerca ogni punto myo in merged_pts
tree_merged = cKDTree(merged_pts)
_, global_idx = tree_merged.query(myo_pts)
# global_idx[i] = indice globale nella merged mesh del nodo i della submesh mio

# --- KD-tree sui nodi miocardici, risultato in indici globali ---
# Leggi i root nodes dalla mesh BIV originale
biv_mesh_path = os.path.join(".", "sb3901","sb3901_meshes", "sb3901_500um_taggedv2_fastendo.vtk")
biv_mesh    = pv.read(biv_mesh_path)
biv_pts     = np.array(biv_mesh.points)
root_nodes_path = os.path.join(".", "sb3901","sb3901_rootnodes","sb3901_fine500um_candidate_root_nodes.vtx")
root_idx    = np.loadtxt(root_nodes_path, dtype=int)
root_coords = biv_pts[root_idx]

distances, local_idx = tree_myo.query(root_coords)
new_indices = global_idx[local_idx]   # indici globali nella mesh merged

# --- Report ---
print(f"Root nodes totali: {len(new_indices)}")
print(f"Distanza max dal nodo originale: {distances.max():.4f} mm")
print(f"Distanza media: {distances.mean():.6f} mm")

problematic = np.where(distances > 0.5)[0]
if len(problematic) > 0:
    print(f"ATTENZIONE: {len(problematic)} nodi lontani >0.5 mm")
else:
    print("Tutti i root nodes mappati correttamente su tessuto miocardico ✓")

# --- Verifica tag dei nodi trovati ---
# Ogni nodo può appartenere a celle di tag diversi; controlla il tag dominante
'''
print("\nVerifica tag celle adiacenti ai root nodes:")
for j, (ni, dist) in enumerate(zip(new_indices[:5], distances[:5])):  # mostra primi 5
    # Trova celle che contengono questo nodo
    adj_cells = merged_mesh.extract_points([ni])
    adj_tags  = cell_tags[merged_mesh.find_cells_within_bounds(
                    [merged_pts[ni,0]-0.01, merged_pts[ni,0]+0.01,
                     merged_pts[ni,1]-0.01, merged_pts[ni,1]+0.01,
                     merged_pts[ni,2]-0.01, merged_pts[ni,2]+0.01])]
    print(f"  Root {j}: nodo globale {ni}, distanza {dist:.4f} mm")
'''
# --- Salva in formato .vtx per openCARP ---
with open(os.path.join(".", "sb3901","sb3901_rootnodes","sb3901_root_nodes_mapping_torsobiv_v2.vtx"), "w") as f:
    for i in new_indices:
        f.write(f"{i}\n")
#np.savetxt("sb301_root_nodes_mapping_torsobiv.dat", new_indices, fmt="%d")

print("\nFIle root nodes mappati salvato")

