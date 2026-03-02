import numpy as np
import pyvista as pv
import os
from scipy.spatial import cKDTree
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.util import numpy_support as VN

def save_vtk_vtu_to_csv(anatomy_subject_name, geometric_data_dir, target_resolution, vtk_filename):
    input_dir = geometric_data_dir + anatomy_subject_name + '/'
    output_dir = geometric_data_dir + anatomy_subject_name + '/' + anatomy_subject_name + '_' + target_resolution + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(input_dir + vtk_filename):
        reader = vtkUnstructuredGridReader()
        if vtk_filename.endswith('.vtu'):
            reader = vtkXMLUnstructuredGridReader()
        else:
            reader = vtkUnstructuredGridReader()
        reader.SetFileName(input_dir + vtk_filename)
        if not vtk_filename.endswith('.vtu'):
            reader.ReadAllVectorsOn()
            reader.ReadAllScalarsOn()
        reader.Update()
        data = reader.GetOutput()
        unprocessed_node_xyz = VN.vtk_to_numpy(data.GetPoints().GetData())
        print(unprocessed_node_xyz.shape)
        # Save node xyz coordinates
        np.savetxt( output_dir + anatomy_subject_name + '_'
                   + target_resolution + '_xyz.csv', unprocessed_node_xyz, delimiter=',', fmt='%.16f')
        n_tetra = data.GetNumberOfCells()
        unprocessed_tetra = np.reshape(VN.vtk_to_numpy(data.GetCells().GetConnectivityArray()), [n_tetra, 4])
        print(unprocessed_tetra.shape)
        # Save tetra
        np.savetxt(output_dir + anatomy_subject_name + '_' + target_resolution + '_tetra.csv', unprocessed_tetra, delimiter=',', fmt='%d')

def save_vtu_arrays_to_csv(anatomy_subject_name, geometric_data_dir, target_resolution, vtu_filename):
    input_dir = geometric_data_dir + anatomy_subject_name + '/'
    output_dir = geometric_data_dir + anatomy_subject_name + '/' + anatomy_subject_name + '_' + target_resolution + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(input_dir + vtu_filename)
    reader.Update()
    data = reader.GetOutput()

    # Point Data
    point_data = data.GetPointData()
    for i in range(point_data.GetNumberOfArrays()):
        array_name = point_data.GetArrayName(i)
        array = point_data.GetArray(i)
        np_array = VN.vtk_to_numpy(array)
        np.savetxt(os.path.join(output_dir, anatomy_subject_name + '_' + target_resolution + '_nodefield_' + array_name + '.csv'), np_array, delimiter=',')

    # Cell Data
""" cell_data = data.GetCellData()
    for i in range(cell_data.GetNumberOfArrays()):
        array_name = cell_data.GetArrayName(i)
        array = cell_data.GetArray(i)
        np_array = VN.vtk_to_numpy(array)
        np.savetxt(os.path.join(output_dir, array_name + '.csv'), np_array, delimiter=',')
     """

#Estrae i nodi delle superfici LV e RV da una mesh VTU.
def save_endo_nodes_to_csv(anatomy_subject_name, geometric_data_dir, target_resolution, vtu_filename, lv_tag, rv_tag, tag_array_name='class'):
    input_dir = geometric_data_dir + anatomy_subject_name + '/'
    vtu_path = input_dir + vtu_filename
    output_dir = geometric_data_dir + anatomy_subject_name + '/' + anatomy_subject_name + '_' + target_resolution + '/'
    print(f"Caricamento mesh: {vtu_path}...")
    mesh = pv.read(vtu_path)
    
    # 1. Estrai la superficie esterna mantenendo gli ID originali del volume
    # 'pass_pointid=True' crea l'array 'vtkOriginalPointIds'
    surface = mesh.extract_surface(pass_pointid=True, pass_cellid=True)
    
    # Controllo se il nome del tag esiste
    has_cell = tag_array_name in surface.cell_data
    has_point = tag_array_name in surface.point_data
    if not (has_cell or has_point):
        print(f"ERRORE: Campo '{tag_array_name}' non trovato. Campi cell_data: {list(surface.cell_data.keys())}, point_data: {list(surface.point_data.keys())}")
        return

    # 2. Estrai LV
    print(f"Estrazione LV (Tag={lv_tag})...")
    lv_surf = surface.threshold(value=[lv_tag, lv_tag], scalars=tag_array_name, preference='cell')
    if lv_surf.n_points > 0:
        lv_ids = np.unique(lv_surf['vtkOriginalPointIds'])
        np.savetxt(os.path.join(output_dir, anatomy_subject_name + '_' + target_resolution + '_boundarynodefield' + '_lvendo' + '.csv'), lv_ids, fmt='%d')
    else:
        print("Nessun nodo trovato per LV!")

    # 3. Estrai RV
    print(f"Estrazione RV (Tag={rv_tag})...")
    rv_surf = surface.threshold(value=[rv_tag, rv_tag], scalars=tag_array_name, preference='cell')
    if rv_surf.n_points > 0:
        rv_ids = np.unique(rv_surf['vtkOriginalPointIds'])
        np.savetxt(os.path.join(output_dir, anatomy_subject_name + '_' + target_resolution + '_boundarynodefield' + '_rvendo' + '.csv'), rv_ids, fmt='%d')
    else:
        print("Nessun nodo trovato per RV!")

def extract_ids_from_tagged_vtp(vtu_path, vtp_path, tag_array_name, target_tag_value, output_csv):
    """
    1. Carica il VTP di superficie e il VTU volumetrico.
    2. Filtra il VTP prendendo solo la parte con il 'target_tag_value' (es. LV).
    3. Trova gli ID corrispondenti nella mesh volumetrica (VTU) basandosi sulla posizione.
    """
    print(f"--- Elaborazione Tag {target_tag_value} da {vtp_path} ---")
    
    # 1. Carica le mesh
    vol_mesh = pv.read(vtu_path)  # Mesh volumetrica (senza tag)
    surf_mesh_all = pv.read(vtp_path) # Mesh superficiale (con tag)
    
    # 2. Verifica che il nome del tag esista
    # Pyvista gestisce i dati in point_data o cell_data
    if tag_array_name not in surf_mesh_all.array_names:
        print(f"ERRORE: Campo '{tag_array_name}' non trovato nel VTP.")
        print(f"Campi disponibili: {surf_mesh_all.array_names}")
        return

    # 3. Estrai solo la superficie specifica (es. solo LV)
    # threshold filtra le celle/punti che hanno quel valore specifico
    specific_surf = surf_mesh_all.threshold(value=[target_tag_value, target_tag_value], 
                                            scalars=tag_array_name, 
                                            preference='cell') # O 'point' se i tag sono sui nodi
    
    if specific_surf.n_points == 0:
        print(f"ATTENZIONE: Nessun punto trovato con Tag={target_tag_value}. Verifica i valori in Paraview.")
        return

    print(f"  Punti superficie estratti: {specific_surf.n_points}")
    print(f"  Punti volume totale: {vol_mesh.n_points}")

    # 4. Mapping Spaziale (KDTree)
    # Costruiamo l'albero sui punti del volume (target degli ID)
    tree = cKDTree(vol_mesh.points)
    
    # Cerchiamo i punti della superficie estratta dentro il volume
    # tolerance=1e-4 assume che le mesh siano perfettamente allineate
    dist, indices = tree.query(specific_surf.points, k=1)
    
    # Filtro di sicurezza (se i punti distano troppo, ignorali)
    tolerance = 1e-4
    valid_mask = dist < tolerance
    final_indices = np.unique(indices[valid_mask])
    
    invalid_count = len(indices) - len(final_indices)
    if invalid_count > 0:
        print(f"  Nota: {invalid_count} punti scartati per tolleranza eccessiva.")

    # 5. Salva CSV
    np.savetxt(output_csv, final_indices, fmt='%d')
    print(f"  SALVATO: {output_csv} con {len(final_indices)} nodi unici.\n")



if __name__ == "__main__":
    geometric_data_dir = './cardiac-data/meta_data/geometric_data/'
    subject_name = 'kaggle503'
    target_resolution = 'coarse2'
    vtu_filename = 'kaggle503cobiveco.vtu'
    save_vtk_vtu_to_csv(subject_name, geometric_data_dir, target_resolution, vtu_filename)
    save_vtu_arrays_to_csv(subject_name, geometric_data_dir, target_resolution, vtu_filename)

    vtu_filename = 'kaggle503_classes.vtu'
    input_dir = geometric_data_dir + subject_name + '/'
    vtu_path = input_dir + vtu_filename
    output_dir = geometric_data_dir + subject_name + '/' + subject_name + '_' + target_resolution + '/'
    vtp_filename = 'kaggle503_surfaces.vtp'
    vtp_path = input_dir + vtp_filename
    #save_endo_nodes_to_csv(subject_name, geometric_data_dir, target_resolution, vtu_filename, lv_tag=3, rv_tag=2, tag_array_name='surClass')

    TAG_LV_ENDO = 3  # Valore per Endocardio LV
    TAG_RV_ENDO = 2  # Valore per Endocardio RV
    TAG_ARRAY_NAME = 'surClass'

    extract_ids_from_tagged_vtp(
        vol_path=vtu_path,
        vtp_path=vtp_path,
        tag_array_name=TAG_ARRAY_NAME,
        target_tag_value=TAG_LV_ENDO,
        output_csv=os.path.join(output_dir, subject_name + '_' + target_resolution + '_boundarynodefield' + '_lvendo' + '.csv')
    )

    # Genera file RV
    extract_ids_from_tagged_vtp(
        vol_path=vtu_path,
        vtp_path=vtp_path,
        tag_array_name=TAG_ARRAY_NAME,
        target_tag_value=TAG_RV_ENDO,
        output_csv=os.path.join(output_dir, subject_name + '_' + target_resolution + '_boundarynodefield' + '_rvendo' + '.csv')
    )