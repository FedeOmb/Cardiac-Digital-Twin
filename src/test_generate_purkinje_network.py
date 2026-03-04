from tabnanny import verbose

import numpy as np
import os
from cellular_models import StepFunctionUpstrokeEP
from geometry_functions import EikonalGeometry
from conduction_system import EmptyConductionSystem, PurkinjeSystemVC, select_random_root_nodes
from io_functions import write_purkinje_vtk, write_root_node_csv, read_csv_file, get_node_xyz_filename
from path_config import get_path_mapping
from utils import get_xyz_name_list, map_indexes

#geometric_data_dir = '/home/federico/Cardiac-Digital-Twin/cardiac-data/meta_data/geometric_data/'
#geometric_data_dir = os.path.join(os.path.expanduser("~"), "Desktop", "digital-twin-framework-camps", "Cardiac-Digital-Twin", "cardiac-data", "meta_data", "geometric_data/")
source_resolution = 'coarse1500'

def generate_dummy_fiber_files(subject_name, geometric_data_dir, resolution):
    """Genera file dummy per fibre, sheet, normal e material se mancanti."""
    mesh_dir = os.path.join(geometric_data_dir, subject_name, f"{subject_name}_{resolution}")
    file_prefix = f"{subject_name}_{resolution}"
    
    if not os.path.exists(mesh_dir):
        print(f"Attenzione: La cartella {mesh_dir} non esiste.")
        return

    # 1. Leggi il numero di nodi dal file xyz
    xyz_filename = os.path.join(mesh_dir, f"{file_prefix}_xyz.csv")
    if not os.path.exists(xyz_filename):
        print(f"Errore: File coordinate non trovato: {xyz_filename}")
        return
    
    # Conta le righe (nodi)
    with open(xyz_filename, 'r') as f:
        n_nodes = sum(1 for _ in f)
    
    # 2. Genera file fibre (x, y, z unitari ortogonali) se mancanti
    vectors = {'fibre': [1, 0, 0], 'sheet': [0, 1, 0], 'normal': [0, 0, 1]}
    for name, vec in vectors.items():
        path = os.path.join(mesh_dir, f"{file_prefix}_nodefield_{name}.csv")
        if not os.path.exists(path):
            print(f"Generazione file dummy {name}: {path}")
            np.savetxt(path, np.tile(vec, (n_nodes, 1)), delimiter=',')

    # 3. Genera file material (se manca) - Richiesto da RawVCFibreCardiacGeoTet
    tetra_path = os.path.join(mesh_dir, f"{file_prefix}_tetra.csv")
    material_path = os.path.join(mesh_dir, f"{file_prefix}_material_tetra.csv")
    if os.path.exists(tetra_path) and not os.path.exists(material_path):
        with open(tetra_path, 'r') as f:
            n_tetra = sum(1 for _ in f)
        print(f"Generazione file dummy material: {material_path}")
        np.savetxt(material_path, np.ones((n_tetra, 1), dtype=int), delimiter=',', fmt='%d')

    # 4. Genera file elettrodi (se manca) - Richiesto da SimpleCardiacGeoTet
    electrode_path = os.path.join(geometric_data_dir, subject_name, f"{subject_name}_electrode_xyz.csv")
    if not os.path.exists(electrode_path):
        print(f"Generazione file dummy elettrodi: {electrode_path}")
        np.savetxt(electrode_path, np.array([[0.0, 0.0, 0.0]]), delimiter=',')

def generate_purkinje_network(subject_name, geometric_data_dir, resolution):

    output_dir = os.path.join(geometric_data_dir, subject_name, f"{subject_name}_{resolution}", "purkinje/")
    print(f"Output directory: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    verbose = True
    # 1. Carica la Geometria
    # Coordinate Cobiveco calcolate
    vc_name_list = ['ab_cut', 'rt']
    #vc_name_list = ['ab', 'rt']    
    resting_vm_value = 0.
    upstroke_vm_value = 1.
    cellular_model = StepFunctionUpstrokeEP(resting_vm_value=resting_vm_value, upstroke_vm_value=upstroke_vm_value,
                                                    verbose=verbose)
    celltype_vc_info = {}
    
    # Genera file ausiliari (fibre, materiali, elettrodi) se non esistono
    generate_dummy_fiber_files(subject_name, geometric_data_dir, resolution)
    
    geometry = EikonalGeometry(cellular_model=cellular_model, celltype_vc_info=celltype_vc_info,
                                   conduction_system=EmptyConductionSystem(verbose=verbose),
                                   geometric_data_dir=geometric_data_dir, resolution=resolution,
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
    write_purkinje_vtk(edge_list=lv_candidate_purkinje_edge, filename=subject_name + '_' + resolution + '_candidate_LV_Purkinje', node_xyz=node_xyz,
                            verbose=verbose, visualisation_dir=output_dir)
    lv_candidate_root_node_index, rv_candidate_root_node_index = geometry.get_lv_rv_candidate_root_node_index()
    write_root_node_csv(filename=subject_name + '_' + resolution + '_candidate_LV_root_nodes.csv', node_vc_list=node_vc_list, node_xyz=node_xyz,
                    root_node_index_list=lv_candidate_root_node_index, vc_name_list=vc_name_list, verbose=verbose,
                    visualisation_dir=output_dir, xyz_name_list=get_xyz_name_list())
    # RV
    write_purkinje_vtk(edge_list=rv__candidate_purkinje_edge, filename=subject_name + '_' + resolution + '_candidate_RV_Purkinje', node_xyz=node_xyz,
                            verbose=verbose, visualisation_dir=output_dir)
    write_root_node_csv(filename=subject_name + '_' + resolution + '_candidate_RV_root_nodes.csv', node_vc_list=node_vc_list, node_xyz=node_xyz,
                    root_node_index_list=rv_candidate_root_node_index, vc_name_list=vc_name_list, verbose=verbose,
                    visualisation_dir=output_dir, xyz_name_list=get_xyz_name_list())
    
    # 4. Randomizzazione Attivazione root nodes
    all_candidate_root_nodes_index = conduction_system.get_candidate_root_node_index()
    all_candidate_root_nodes_index = conduction_system.get_candidate_root_node_index()
    total_candidate_root_nodes = len(all_candidate_root_nodes_index)
    random_active_percentage = np.random.uniform(0.5, 0.9)
    print(f"Attivando casualmente {random_active_percentage*100:.1f}% dei root nodes candidati ({total_candidate_root_nodes} totali)")
    nb_active_nodes = int(total_candidate_root_nodes * random_active_percentage)
    selection_mask = select_random_root_nodes(nb_root_nodes=nb_active_nodes, candidate_root_node_indexes=all_candidate_root_nodes_index)

    active_root_nodes = all_candidate_root_nodes_index[selection_mask]
    lv_purkinje_edge, rv_purkinje_edge = geometry.get_lv_rv_selected_purkinje_edge(root_node_meta_index=selection_mask)
    write_purkinje_vtk(edge_list=lv_purkinje_edge, filename=subject_name + '_' + resolution + '_selected_LV_Purkinje', node_xyz=node_xyz,
                            verbose=verbose, visualisation_dir=output_dir)
    write_purkinje_vtk(edge_list=rv_purkinje_edge, filename=subject_name + '_' + resolution + '_selected_RV_Purkinje', node_xyz=node_xyz,
                            verbose=verbose, visualisation_dir=output_dir)
    # 5. Esportazione per openCARP (.vtx)
    vtx_filename = os.path.join(output_dir, f"{subject_name}_active_root_nodes.vtx")
    np.savetxt(vtx_filename, active_root_nodes, fmt='%d')
    print(f"Generati {len(active_root_nodes)} root nodes e salvati per opencarp in {vtx_filename}")
    
    return geometry, lv_purkinje_edge, rv_purkinje_edge, active_root_nodes

def map_purkinje_to_fine_old(subject_name, geometric_data_dir, coarse_resolution, fine_resolution, 
                         coarse_geometry, lv_pk_edge, rv_pk_edge, active_root_nodes):
    print(f"Mappatura della rete di Purkinje da {coarse_resolution} a {fine_resolution}...")
    
    # 1. Carica i nodi della mesh fine
    fine_xyz_filename = get_node_xyz_filename(subject_name, geometric_data_dir, fine_resolution)
    if not os.path.exists(fine_xyz_filename):
        print(f"Errore: File coordinate fine non trovato: {fine_xyz_filename}")
        return

    fine_node_xyz = read_csv_file(fine_xyz_filename)
    coarse_node_xyz = coarse_geometry.get_node_xyz()
    
    # 2. Calcola la mappa degli indici (Coarse -> Fine)
    print("Calcolo mapping nodi...")
    coarse_to_fine_mapping = map_indexes(points_to_map_xyz=coarse_node_xyz, reference_points_xyz=fine_node_xyz)
    
    # 3. Mappa gli edges (connettività)
    lv_pk_edge_fine = coarse_to_fine_mapping[lv_pk_edge]
    rv_pk_edge_fine = coarse_to_fine_mapping[rv_pk_edge]
    
    # 4. Mappa i root nodes
    active_root_nodes_fine = coarse_to_fine_mapping[active_root_nodes]
    
    # 5. Salva i risultati
    output_dir = os.path.join(geometric_data_dir, subject_name, f"{subject_name}_{fine_resolution}", "purkinje/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Salvataggio rete mappata in {output_dir}")
    write_purkinje_vtk(edge_list=lv_pk_edge_fine, filename=subject_name + '_' + fine_resolution + '_selected_LV_Purkinje', node_xyz=fine_node_xyz, verbose=True, visualisation_dir=output_dir)
    write_purkinje_vtk(edge_list=rv_pk_edge_fine, filename=subject_name + '_' + fine_resolution + '_selected_RV_Purkinje', node_xyz=fine_node_xyz, verbose=True, visualisation_dir=output_dir)
    vtx_filename = os.path.join(output_dir, f"{subject_name}_{fine_resolution}_active_root_nodes.vtx")
    np.savetxt(vtx_filename, active_root_nodes_fine, fmt='%d')
    print(f"Salvati {len(active_root_nodes_fine)} root nodes mappati in {vtx_filename}")

def map_purkinje_to_fine(subject_name, geometric_data_dir, coarse_resolution, fine_resolution, 
                         coarse_geometry, lv_pk_edge, rv_pk_edge, active_root_nodes):
    print(f"Mappatura della rete di Purkinje da {coarse_resolution} a {fine_resolution}...")
    
    # 1. Carica i nodi della mesh fine
    fine_xyz_filename = get_node_xyz_filename(subject_name, geometric_data_dir, fine_resolution)
    if not os.path.exists(fine_xyz_filename):
        print(f"Errore: File coordinate fine non trovato: {fine_xyz_filename}")
        return

    fine_node_xyz = read_csv_file(fine_xyz_filename)
    coarse_node_xyz = coarse_geometry.get_node_xyz()
    # Rilevamento scala mesh fine e conversione temporanea in cm per il mapping    
    max_fine = np.max(np.abs(fine_node_xyz))
    if max_fine > 5000: # Probabilmente micrometri (um) 
        print("Mesh fine rilevata in micrometri (um). Converto temporaneamente in cm per il mapping.")
        fine_to_cm_factor = 1e-4 # 1/10000           
    elif max_fine > 50: # Probabilmente millimetri (mm)
        print("Mesh fine rilevata in millimetri (mm). Converto temporaneamente in cm per il mapping.")
        fine_to_cm_factor = 0.1    
    else:
        print("Mesh fine rilevata in centimetri (cm). Nessuna conversione necessaria.")
        fine_to_cm_factor = 1.0
    
    fine_node_xyz_cm = fine_node_xyz * fine_to_cm_factor

# 2. Calcola la mappa degli indici (Coarse -> Fine)
    print("Calcolo mapping nodi...")
# Usiamo la versione in cm della mesh fine per il calcolo della distanza
    coarse_to_fine_mapping = map_indexes(points_to_map_xyz=coarse_node_xyz, reference_points_xyz=fine_node_xyz_cm)    
    # 3. Mappa gli edges (connettività)
    lv_pk_edge_fine = coarse_to_fine_mapping[lv_pk_edge]
    rv_pk_edge_fine = coarse_to_fine_mapping[rv_pk_edge]
    
    # 4. Mappa i root nodes
    active_root_nodes_fine = coarse_to_fine_mapping[active_root_nodes]
    
    # 5. Salva i risultati
    output_dir = os.path.join(geometric_data_dir, subject_name, f"{subject_name}_{fine_resolution}", "purkinje/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Se la mesh fine non era in micrometri, salva una versione extra in um per visualizzazione con openCARP
    if fine_to_cm_factor > 1e-4:
        print("Generazione file VTK extra in micrometri per compatibilità visualizzazione openCARP...")
        nodes_to_write_um = fine_node_xyz * (1e-4 / fine_to_cm_factor) # Converte in um
        write_purkinje_vtk(edge_list=lv_pk_edge_fine, filename=subject_name + '_selected_LV_Purkinje_mapped_um', node_xyz=nodes_to_write_um, verbose=True, visualisation_dir=output_dir)
        write_purkinje_vtk(edge_list=rv_pk_edge_fine, filename=subject_name + '_selected_RV_Purkinje_mapped_um', node_xyz=nodes_to_write_um, verbose=True, visualisation_dir=output_dir)    
    
    vtx_filename = os.path.join(output_dir, f"{subject_name}_{fine_resolution}_active_root_nodes.vtx")
    np.savetxt(vtx_filename, active_root_nodes_fine, fmt='%d')
    print(f"Salvati {len(active_root_nodes_fine)} root nodes mappati in {vtx_filename}")

if __name__ == "__main__":

    script_directory = os.path.dirname(os.path.realpath(__file__))
    print('Script directory:', script_directory)
    # Change the current working directory to the script dierctory
    os.chdir(script_directory)
    working_directory = os.getcwd()
    print('Working directory:', working_directory)
    if os.path.isfile('../.custom_config/.your_path_mapping.txt'):
        #path_dict = get_path_mapping('../.custom_config/.your_path_mapping_docker_vscode.txt')
        path_dict = get_path_mapping()
    else:
        raise 'Missing data and results configuration file at: ../.custom_config/.your_path_mapping.txt'    
    
    data_dir = path_dict["data_path"]
    geometric_data_dir = data_dir + 'geometric_data/'

    subject_name = 'kaggle502'
    # subject_name = 'DTI003'
    output_dir = 'purkinje/'
    coarse_resolution = 'coarse1500'
    fine_resolution = 'fine500'

    geometry, lv_pk, rv_pk, roots = generate_purkinje_network(subject_name=subject_name, geometric_data_dir=geometric_data_dir, resolution=coarse_resolution)
    
    # Mappatura su fine
    if os.path.exists(os.path.join(geometric_data_dir, subject_name, f"{subject_name}_{fine_resolution}")):
        map_purkinje_to_fine(subject_name, geometric_data_dir, coarse_resolution, fine_resolution, geometry, lv_pk, rv_pk, roots)