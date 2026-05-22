import os
import pyvista as pv
import numpy as np

def get_nodes_in_radius_by_tag(vtk_file, center_node, endo_tag, radius_um=1500):
    mesh = pv.read(vtk_file)

    # Seleziona solo le celle (elementi) con il tag endocardio
    # Il tag è solitamente in CellData, campo "elemTag" o "MaterialID"
    endo_mask = mesh.cell_data['elemTag'] == endo_tag
    endo_mesh = mesh.extract_cells(endo_mask)

    # Nodi del sottoinsieme endocardico
    endo_pts    = endo_mesh.points                # coordinate
    endo_pt_ids = endo_mesh.point_data['vtkOriginalPointIds']  # indici originali

    # Filtra per raggio
    center = mesh.points[center_node]
    dists  = np.linalg.norm(endo_pts - center, axis=1)
    mask   = dists <= radius_um

    return endo_pt_ids[mask]

mapped_rn_path = os.path.join(".","sb3901","sb3901_rootnodes","sb3901_root_nodes_mapping_torsobiv_v2.vtx")
nodes = np.loadtxt(mapped_rn_path, dtype=int)  
vtx_dir = os.path.join(".","sb3901","sb3901_rootnodes", 'sb3901_rootnodes_mapped_torsobiv_v2')
os.makedirs(vtx_dir, exist_ok=True)
torso_biv_mesh_path = os.path.join(".", "sb3901","sb3901_torso", "sb3901_finalmeshum_torsobiv_v2.vtk")
stim_cmds = ['-num_stim', len(nodes)]
for i, node in enumerate(nodes):
    nearby_nodes = get_nodes_in_radius_by_tag(torso_biv_mesh_path, node, endo_tag=3, radius_um=2000)
    vtx_filename = os.path.join(vtx_dir, f'RN{i+1}.vtx')
    with open(vtx_filename, 'w') as f:
        f.write(f"{len(nearby_nodes)}\n")  # Numero di nodi da stimolare
        f.write("intra\n")  # Header standard di openCARP per i vertici intracellulari
        for n in nearby_nodes:
            f.write(f"{n}\n")