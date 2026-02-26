from tabnanny import verbose

import numpy as np
import os
from cellular_models import StepFunctionUpstrokeEP
from geometry_functions import EikonalGeometry
from conduction_system import EmptyConductionSystem, PurkinjeSystemVC, select_random_root_nodes
from io_functions import write_purkinje_vtk, write_root_node_csv
from utils import get_xyz_name_list

geometric_data_dir = '/home/federico/Cardiac-Digital-Twin/cardiac-data/meta_data/geometric_data/'
source_resolution = 'coarse2'

def generate_purkinje_network(subject_name, output_dir):

    output_dir = os.path.join(geometric_data_dir, subject_name, f"{subject_name}_{source_resolution}", "simulation_files")
    # 1. Carica la Geometria
    # Coordinate Cobiveco calcolate
    vc_name_list = ['ab', 'rt', 'tv']
    resting_vm_value = 0.
    upstroke_vm_value = 1.
    cellular_model = StepFunctionUpstrokeEP(resting_vm_value=resting_vm_value, upstroke_vm_value=upstroke_vm_value,
                                                    verbose=verbose)
    celltype_vc_info = {}
    geometry = EikonalGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                                   conduction_system=EmptyConductionSystem(verbose=verbose),
                                   geometric_data_dir=geometric_data_dir, resolution=source_resolution,
                                   subject_name=subject_name, vc_name_list=vc_name_list, verbose=verbose)

    # 2. Genera la Rete di Purkinje (Candidate Network)
    random_node_distance = np.random.uniform(1.5, 4.0) # Esempio: tra 1.5mm e 4mm

    approx_djikstra_purkinje_max_path_len = 200
    lv_inter_root_node_distance = 2.5  
    rv_inter_root_node_distance = 2.5 
    
    conduction_system = PurkinjeSystemVC(
        approx_djikstra_purkinje_max_path_len=approx_djikstra_purkinje_max_path_len, 
        geometry=geometry,
        lv_inter_root_node_distance=lv_inter_root_node_distance,
        rv_inter_root_node_distance=rv_inter_root_node_distance,
        verbose=True
    )
    
    # Assign conduction_system to its geometry
    geometry.set_conduction_system(conduction_system)

    lv_candidate_purkinje_edge, rv__candidate_purkinje_edge = geometry.get_lv_rv_candidate_purkinje_edge()
    node_xyz = geometry.get_node_xyz()
    print("Node xyz shape:", node_xyz.shape)
    print("node example", node_xyz[:10])
    node_vc_list = [geometry.get_node_vc_field(vc_name=vc_name) for vc_name in vc_name_list]
    # 3. calcola le Purkinje Network e i Root Nodes (PMJ) candidati per entrambi i ventricoli
    # LV
    write_purkinje_vtk(edge_list=lv_candidate_purkinje_edge, filename=subject_name + '_candidate_LV_Purkinje', node_xyz=node_xyz,
                            verbose=verbose, visualisation_dir=output_dir)
    lv_candidate_root_node_index, rv_candidate_root_node_index = geometry.get_lv_rv_candidate_root_node_index()
    write_root_node_csv(filename=subject_name + '_candidate_LV_root_nodes.csv', node_vc_list=node_vc_list, node_xyz=node_xyz,
                    root_node_index_list=lv_candidate_root_node_index, vc_name_list=vc_name_list, verbose=verbose,
                    visualisation_dir=output_dir, xyz_name_list=get_xyz_name_list())
    # RV
    write_purkinje_vtk(edge_list=rv__candidate_purkinje_edge, filename=subject_name + '_candidate_RV_Purkinje', node_xyz=node_xyz,
                            verbose=verbose, visualisation_dir=output_dir)
    write_root_node_csv(filename=subject_name + '_candidate_RV_root_nodes.csv', node_vc_list=node_vc_list, node_xyz=node_xyz,
                    root_node_index_list=rv_candidate_root_node_index, vc_name_list=vc_name_list, verbose=verbose,
                    visualisation_dir=output_dir, xyz_name_list=get_xyz_name_list())
    
    # 4. Randomizzazione Attivazione root nodes
    all_candidate_root_nodes_index = conduction_system.get_all_candidate_root_node_index()
    total_candidate_root_nodes = len(all_candidate_root_nodes_index)
    random_active_percentage = np.random.uniform(0.5, 0.9)
    print(f"Attivando casualmente {random_active_percentage*100:.1f}% dei root nodes candidati ({total_candidate_root_nodes} totali)")
    nb_active_nodes = int(total_candidate_root_nodes * random_active_percentage)
    selection_mask = select_random_root_nodes(nb_active_nodes=nb_active_nodes, candidate_root_node_indexes=all_candidate_root_nodes_index)

    active_root_nodes = all_candidate_root_nodes_index[selection_mask]
    lv_purkinje_edge, rv_purkinje_edge = geometry.get_lv_rv_selected_purkinje_edge(root_node_meta_index=selection_mask)
    write_purkinje_vtk(edge_list=lv_purkinje_edge, filename=subject_name + '_selected_LV_Purkinje', node_xyz=node_xyz,
                            verbose=verbose, visualisation_dir=output_dir)
    write_purkinje_vtk(edge_list=rv_purkinje_edge, filename=subject_name + '_selected_RV_Purkinje', node_xyz=node_xyz,
                            verbose=verbose, visualisation_dir=output_dir)
    # 5. Esportazione per openCARP (.vtx)
    vtx_filename = os.path.join(output_dir, f"{subject_name}_active_root_nodes.vtx")
    np.savetxt(vtx_filename, active_root_nodes, fmt='%d')
    print(f"Generati {len(active_root_nodes)} root nodes e salvati per opencarp in {vtx_filename}")

#
#  generate_dataset_case('/path/to/mesh_folder/', 'Patient001', './output_dataset/')
if __name__ == "__main__":
    subject_name = 'kaggle503'
    output_dir = 'simulation_files/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    generate_purkinje_network(subject_name=subject_name, output_dir=output_dir)