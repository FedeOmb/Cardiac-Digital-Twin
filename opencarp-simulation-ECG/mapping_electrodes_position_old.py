from scipy.spatial import cKDTree
import numpy as np
import os
import json
import pyvista as pv

def transfer_electrode_positions(
    source_pts,          # nodi della mesh sorgente (torso con atri)
    source_electrode_indices,  # array di int: indici dei nodi elettrodo nella mesh sorgente
    target_pts           # nodi della mesh torso target (mesh completa CEMRG)
):
    """
    Trasferisce la posizione degli elettrodi dalla mesh sorgente alla target
    tramite nearest-neighbor search con KD-tree.

    Returns:
        target_electrode_indices: indici dei nodi più vicini nella mesh target
        electrode_coords_source:  coordinate 3D nella mesh sorgente
        electrode_coords_target:  coordinate 3D nella mesh target
        distances:                distanza di trasferimento per ogni elettrodo (mm)
    """
    # Coordinate degli elettrodi nella mesh sorgente
    electrode_coords_source = source_pts[source_electrode_indices]

    # KD-tree sulla superficie del torso target
    # IMPORTANTE: usare solo nodi di superficie, non tutti i nodi del volume
    kdtree = cKDTree(target_pts)

    distances, target_electrode_indices = kdtree.query(
        electrode_coords_source, k=1, workers=-1
    )

    electrode_coords_target = target_pts[target_electrode_indices]
    print("Coordinate elettrodi target (um):")
    print(electrode_coords_target)

    print(f"Trasferimento elettrodi completato:")
    print(f"  Distanza media: {distances.mean():.2f} um")
    print(f"  Distanza max:   {distances.max():.2f} um")

    return target_electrode_indices, electrode_coords_source, electrode_coords_target, distances

# Mappatura standard elettrodi 12-lead (limb + precordiali)
ECG_12_LEAD_LABELS = [
    'ref', 'R', 'L', 'F',          # limb leads (Wilson)
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6'  # precordiali
]

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
    PATH_SOURCE_MESH = os.path.join(".", "torso_mesh_atria_models", "S1_atria_torso.vtk")
    PATH_TARGET_MESH = os.path.join(".", "KCL_torso1", "KCL_torso1.vtk")
    SOURCE_ELECTRODES_FILE = os.path.join(".", "torso_mesh_atria_models", "electrode_nodes.json")
    
    torso_source = pv.read(PATH_SOURCE_MESH)
    torso_target = pv.read(PATH_TARGET_MESH)

    source_surf_pts = np.array(torso_source.extract_surface().points)
    target_surf_pts = np.array(torso_target.extract_surface().points)
    
    with open(SOURCE_ELECTRODES_FILE, 'r', encoding='utf-8') as file:
        electrodes_json = json.load(file)

    print("Indici elettrodi sorgente:", electrodes_json)
    electrode_source_indices = list(electrodes_json.values())
    print(electrode_source_indices)

    target_indices, coords_source, coords_target, dists = transfer_electrode_positions(source_surf_pts, electrode_source_indices, target_surf_pts)
    save_electrode_indices(target_indices, coords_target, ECG_12_LEAD_LABELS,"electrode_indices_mapping")
