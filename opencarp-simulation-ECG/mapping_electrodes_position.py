from scipy.spatial import cKDTree
import numpy as np
import os
import json
import pyvista as pv

def map_electrodes_coords(electrodes_coords,torso_mesh_pts):

    # KD-tree sulla superficie del torso target
    print("creazione kdtree sui nodi del torso...")
    kdtree = cKDTree(torso_mesh_pts)

    distances, target_electrode_indices = kdtree.query(
        electrodes_coords, k=1, workers=-1
    )

    electrode_coords_target = torso_mesh_pts[target_electrode_indices]
    print("Coordinate elettrodi target (um):")
    print(electrode_coords_target)

    print("indici target degli elettrodi:")
    print(target_electrode_indices)

    return target_electrode_indices, distances

# Mappatura standard elettrodi 12-lead (limb + precordiali)
ELECTRODES_LEAD_LABELS = ['LA', 'RA', 'LL', 'RL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def save_electrode_indices(target_indices, coords, labels, out_file):
    """Salva le coordinate dei nodi in formato testo"""
    with open(out_file + '.txt', 'w') as f:
        for coord in coords:
            f.write("{:.2f} {:.2f} {:.2f}\n".format(*coord))
    
    json_nodes = {}
    for i in range(len(labels)):
        json_nodes[labels[i]] = str(target_indices[i])

    with open(out_file + '.json', 'w', encoding='utf-8') as json_file:
        json.dump(json_nodes, json_file, indent=4)

if __name__ == "__main__":
    torso_input_folder = os.path.join(".", "sb3901", "sb3901_torso")
    PATH_TORSO_MESH = os.path.join(torso_input_folder, "sb3901_finalmesh_torsobiv_opencarpv2.pts")
    SOURCE_ELECTRODES_FILE = os.path.join(torso_input_folder, "sb3901_electrodesum.pts")

    mesh_nodes = np.loadtxt(PATH_TORSO_MESH, skiprows=1)    
    electrodes_coords = np.loadtxt(SOURCE_ELECTRODES_FILE, skiprows=1, delimiter=" ")
    print("Coordinate elettrodi sorgente (um):")
    print(electrodes_coords)
    print(electrodes_coords.shape)
    target_indices, dists = map_electrodes_coords(electrodes_coords, mesh_nodes)

    # Salva il file .vtx nel formato richiesto da openCARP
    out_vtx = os.path.join(torso_input_folder, 'sb3901_electrodes_ids_mapped_torsobiv.vtx')
    with open(out_vtx, 'w') as f:
        f.write(f"{len(target_indices)}\n")  
        f.write("extra\n") 
        for node in target_indices:
            f.write(f"{node}\n")
    
    # Generazione file .vtx singoli per ogni elettrodo
    vtx_out_dir = os.path.join(torso_input_folder, "sb3901_electrodes_vtx")
    os.makedirs(vtx_out_dir, exist_ok=True)
    for i in range(len(target_indices)):
        vtx_filename = os.path.join(vtx_out_dir, f'{ELECTRODES_LEAD_LABELS[i]}.vtx')
        with open(vtx_filename, 'w') as f:
            f.write(f"1\n") 
            f.write("extra\n") 
            f.write(f"{target_indices[i]}\n")