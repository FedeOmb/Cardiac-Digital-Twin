import sys
sys.path.insert(0, "../src")
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
source_resolution = 'coarse1500cm'

def select_physiological_root_nodes(geometry, all_candidates, lv_candidates, rv_candidates):
    """
    Seleziona 7 root nodes basandosi su coordinate fisiologiche Cobiveco (Mincholé et al. 2019).
    ab_cut: 0 (Apice) -> 1 (Base).
    rt: rotazione (0.85 = setto, 0.35 = parete libera, 0.6 = paraseptale ant, 0.1 = paraseptale post).
    """
    lv_targets = [
        [0.5, 0.85],   # 1. Left Anterior Fascicle (LAF)
        [0.8, 0.6],   # 2. Left Posterior Fascicle (LPF)
        [0.5, 0.1],  # 3. Left Septal Fascicle (LSF) / Apice
        [0.4, 0.2]   # 4. Left Free Wall (LFW) - Opzionale ma utile per mesh grandi
    ]
    
    rv_targets = [           #(ab,rt)
        [0.5, 0.85],  # 5. RV Mid Septum
        [0.3, 0.4],   # 6. RV Free Wall Anterior
        [0.7, 0.20]   # 7. RV Free Wall Posterior
    ]
    
    ab_field = geometry.get_node_vc_field('ab')
    rt_field = geometry.get_node_vc_field('rt')
    
    selected_nodes = []
    
    def find_closest(target, candidates):
        best_node = None
        min_dist = float('inf')
        for node in candidates:
            # Gestione della circolarità di rt (varia da 0 a 1)
            rt_dist = abs(rt_field[node] - target[1])
            rt_dist = min(rt_dist, 1.0 - rt_dist)
            
            dist = (ab_field[node] - target[0])**2 + rt_dist**2
            if dist < min_dist:
                min_dist = dist
                best_node = node
        return best_node
        
    for target in lv_targets:
        node = find_closest(target, lv_candidates)
        if node is not None and node not in selected_nodes:
            selected_nodes.append(node)
            
    for target in rv_targets:
        node = find_closest(target, rv_candidates)
        if node is not None and node not in selected_nodes:
            selected_nodes.append(node)
            
    selection_mask = np.isin(all_candidates, selected_nodes)
    return selection_mask

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


def generate_purkinje_network(subject_name, geometric_data_dir, resolution, lv_inter_node_dist=2.5, rv_inter_node_dist=2.5, output_dir_name="putkinje"):

    output_dir = os.path.join(geometric_data_dir, subject_name, f"{subject_name}_{resolution}", f"{output_dir_name}/")
    print(f"Output directory: {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    verbose = True
    # 1. Carica la Geometria
    # Coordinate Cobiveco calcolate
    vc_name_list = ['ab', 'rt']
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
    lv_inter_root_node_distance = lv_inter_node_dist 
    rv_inter_root_node_distance = rv_inter_node_dist 
    
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
    write_root_node_csv(filename=subject_name + '_' + resolution + '_candidate_LV_root_nodes_coords.csv', node_vc_list=node_vc_list, node_xyz=node_xyz,
                    root_node_index_list=lv_candidate_root_node_index, vc_name_list=vc_name_list, verbose=verbose,
                    visualisation_dir=output_dir, xyz_name_list=get_xyz_name_list())
    # RV
    write_purkinje_vtk(edge_list=rv__candidate_purkinje_edge, filename=subject_name + '_' + resolution + '_candidate_RV_Purkinje', node_xyz=node_xyz,
                            verbose=verbose, visualisation_dir=output_dir)
    write_root_node_csv(filename=subject_name + '_' + resolution + '_candidate_RV_root_nodes_coords.csv', node_vc_list=node_vc_list, node_xyz=node_xyz,
                    root_node_index_list=rv_candidate_root_node_index, vc_name_list=vc_name_list, verbose=verbose,
                    visualisation_dir=output_dir, xyz_name_list=get_xyz_name_list())
    vtx_lv_filename = os.path.join(output_dir, f"{subject_name}_candidate_LV_root_nodes_index.vtx")
    np.savetxt(vtx_lv_filename, lv_candidate_root_node_index, fmt='%d')
    vtx_rv_filename = os.path.join(output_dir, f"{subject_name}_candidate_RV_root_nodes_index.vtx")
    np.savetxt(vtx_rv_filename, rv_candidate_root_node_index, fmt='%d')
    # 4. Randomizzazione Attivazione root nodes
    all_candidate_root_nodes_index = geometry.get_candidate_root_node_index()
    num_candidate_root_nodes = len(all_candidate_root_nodes_index)
    #random_active_percentage = np.random.uniform(0.5, 0.9)
    #print(f"Attivando casualmente {random_active_percentage*100:.1f}% dei root nodes candidati ({total_candidate_root_nodes} totali)")
    #nb_active_nodes = int(num_candidate_root_nodes * random_active_percentage)
    #selection_mask = select_random_root_nodes(nb_root_nodes=nb_active_nodes, candidate_root_node_indexes=all_candidate_root_nodes_index)

    #active_root_nodes = all_candidate_root_nodes_index[selection_mask]
    #lv_purkinje_edge, rv_purkinje_edge = geometry.get_lv_rv_selected_purkinje_edge(root_node_meta_index=selection_mask)
    #write_purkinje_vtk(edge_list=lv_purkinje_edge, filename=subject_name + '_' + resolution + '_selected_LV_Purkinje', node_xyz=node_xyz,
    #                        verbose=verbose, visualisation_dir=output_dir)
    #write_purkinje_vtk(edge_list=rv_purkinje_edge, filename=subject_name + '_' + resolution + '_selected_RV_Purkinje', node_xyz=node_xyz,
    #                        verbose=verbose, visualisation_dir=output_dir)
   # active_root_nodes = all_candidate_root_nodes_index #attivazione di tutti i root nodes
  #  lv_purkinje_edge = lv_candidate_purkinje_edge
  #  rv_purkinje_edge = rv__candidate_purkinje_edge
    purkinje_speed = 0.3 #cm/ms
    activation_times_candidates = geometry.get_candidate_root_node_time(purkinje_speed=purkinje_speed)
    # 5. Esportazione per openCARP (.vtx) - candidate root nodes
    vtx_filename = os.path.join(output_dir, f"{subject_name}_candidate_root_nodes.vtx")
    np.savetxt(vtx_filename, all_candidate_root_nodes_index, fmt='%d')
    times_filename = os.path.join(output_dir, f"{subject_name}_candidate_root_nodes_times.csv")
    np.savetxt(times_filename, activation_times_candidates, fmt='%.4f', header="activation_time_ms", comments='')
    print(f"Generati {len(all_candidate_root_nodes_index)} root nodes candidati e salvati per opencarp in {vtx_filename}")
    
    # 6. Seleziona root nodes fisiologici basati sulle coordinate Cobiveco
    if num_candidate_root_nodes == 0:
        print("Nessun root node candidato trovato.")
           
    else:
        print("Selezionando root nodes in posizioni fisiologiche...")
        selection_mask = select_physiological_root_nodes(
            geometry=geometry, 
            all_candidates=all_candidate_root_nodes_index,
            lv_candidates=lv_candidate_root_node_index,
            rv_candidates=rv_candidate_root_node_index
        )
        active_root_nodes = all_candidate_root_nodes_index[selection_mask]
        print(f"Trovati {len(active_root_nodes)} root nodes fisiologici.")
        lv_purkinje_edge, rv_purkinje_edge = geometry.get_lv_rv_selected_purkinje_edge(root_node_meta_index=selection_mask)
        activation_times_selected = geometry.get_selected_root_node_time(root_node_meta_index=selection_mask, purkinje_speed=purkinje_speed)

    write_purkinje_vtk(edge_list=lv_purkinje_edge, filename=subject_name + '_' + resolution + '_selected_LV_Purkinje', node_xyz=node_xyz,
                            verbose=verbose, visualisation_dir=output_dir)
    write_purkinje_vtk(edge_list=rv_purkinje_edge, filename=subject_name + '_' + resolution + '_selected_RV_Purkinje', node_xyz=node_xyz,
                            verbose=verbose, visualisation_dir=output_dir)
    sel_vtx_filename = os.path.join(output_dir, f"{subject_name}_selected_root_nodes.vtx")
    np.savetxt(sel_vtx_filename, active_root_nodes, fmt='%d')
    sel_times_filename = os.path.join(output_dir, f"{subject_name}_selected_root_nodes_times.csv")
    np.savetxt(sel_times_filename, activation_times_selected, fmt='%.4f', header="activation_time_ms", comments='')
    print(f"Selezionati {len(active_root_nodes)} root nodes casuali e salvati in {sel_vtx_filename}")
    write_root_node_csv(filename=subject_name + '_' + resolution + '_selected_root_nodes_coords.csv', node_vc_list=node_vc_list, node_xyz=node_xyz,
                    root_node_index_list=active_root_nodes, vc_name_list=vc_name_list, verbose=verbose,
                    visualisation_dir=output_dir, xyz_name_list=get_xyz_name_list())
    return geometry, active_root_nodes, all_candidate_root_nodes_index, activation_times_selected, activation_times_candidates


def map_purkinje_to_fine(subject_name, geometric_data_dir, coarse_resolution, fine_resolution, 
                         coarse_geometry, active_root_nodes, candidate_root_nodes, activation_times_selected, activation_times_candidates, 
                         coarse_units="cm", fine_units="um", output_dir_name="purkinje"):
    print(f"Mappatura della rete di Purkinje da {coarse_resolution} a {fine_resolution}...")
    
    # 1. Carica i nodi della mesh fine
    fine_xyz_filename = get_node_xyz_filename(subject_name, geometric_data_dir, fine_resolution)
    if not os.path.exists(fine_xyz_filename):
        print(f"Errore: File coordinate fine non trovato: {fine_xyz_filename}")
        return

    fine_node_xyz = read_csv_file(fine_xyz_filename)
    coarse_node_xyz = coarse_geometry.get_node_xyz()

    # Rilevamento scala mesh fine e conversione temporanea in cm per il mapping    
    scale_factor = 1.0
    if (coarse_units == "cm") and (fine_units == "um"):
        print("Mesh fine in micrometri (um). Converto temporaneamente in cm per il mapping.")
        scale_factor = 1e-4 # 1/10000   conversione cm - um
    elif (coarse_units == "cm") and (fine_units == "mm"):
        print("Mesh fine in millimetri (mm). Converto temporaneamente in cm per il mapping.")
        scale_factor = 0.1 #   conversione cm - mm        
    elif (coarse_units == "cm") and (fine_units == "cm"):
        print("Mesh fine in centimetri (cm). Nessuna conversione necessaria")

    fine_node_xyz_cm = fine_node_xyz * scale_factor

# 2. Calcola la mappa degli indici (Coarse -> Fine)
    print("Calcolo mapping nodi...")
# Usiamo la versione in cm della mesh fine per il calcolo della distanza
    coarse_to_fine_mapping = map_indexes(points_to_map_xyz=coarse_node_xyz, reference_points_xyz=fine_node_xyz_cm)    
    # 3. Mappa gli edges (connettività)
   # lv_pk_edge_fine = coarse_to_fine_mapping[lv_pk_edge]
   # rv_pk_edge_fine = coarse_to_fine_mapping[rv_pk_edge]
    
    # 4. Mappa i root nodes
    active_root_nodes_fine = coarse_to_fine_mapping[active_root_nodes]
    candidate_root_nodes_fine = coarse_to_fine_mapping[candidate_root_nodes]
    # 5. Salva i risultati
    output_dir = os.path.join(geometric_data_dir, subject_name, f"{subject_name}_{fine_resolution}", output_dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    vtx_filename = os.path.join(output_dir, f"{subject_name}_{fine_resolution}_active_root_nodes.vtx")
    np.savetxt(vtx_filename, active_root_nodes_fine, fmt='%d')
    print(f"Salvati {len(active_root_nodes_fine)} root nodes mappati in {vtx_filename}")

    vtx_filename = os.path.join(output_dir, f"{subject_name}_{fine_resolution}_candidate_root_nodes.vtx")
    np.savetxt(vtx_filename, candidate_root_nodes_fine, fmt='%d')
    print(f"Salvati {len(candidate_root_nodes_fine)} candidate root nodes mappati in {vtx_filename}")

    sel_times_filename = os.path.join(output_dir, f"{subject_name}_selected_root_nodes_times.csv")
    np.savetxt(sel_times_filename, activation_times_selected, fmt='%.4f', header="activation_time_ms", comments='')
    cand_times_filename = os.path.join(output_dir, f"{subject_name}_candidate_root_nodes_times.csv")
    np.savetxt(cand_times_filename, activation_times_candidates, fmt='%.4f', header="activation_time_ms", comments='')

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

    subject_name = 'sb3101'
    output_dir = 'purkinje300'
    coarse_resolution = 'coarse1500cm'
    fine_resolution = 'fine500um'


    geometry, active_root_nodes, candidate_root_nodes, activation_times_selected, activation_times_candidates = generate_purkinje_network(
                                                    subject_name=subject_name, geometric_data_dir=geometric_data_dir, resolution=coarse_resolution, 
                                                    lv_inter_node_dist=2.5, rv_inter_node_dist=2.5, output_dir_name=output_dir)
    
    # Mappatura su fine
    if os.path.exists(os.path.join(geometric_data_dir, subject_name, f"{subject_name}_{fine_resolution}")):
        map_purkinje_to_fine(subject_name, geometric_data_dir, coarse_resolution, fine_resolution, geometry, 
                             active_root_nodes, candidate_root_nodes, activation_times_selected, activation_times_candidates, 
                             coarse_units="cm", fine_units="um", output_dir_name=output_dir)