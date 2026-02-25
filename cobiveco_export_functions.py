import numpy as np
import pyvista as pv
import os
from vtkmodules.vtkIOLegacy import vtkUnstructuredGridReader
from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader
from vtkmodules.util import numpy_support as VN

def save_vtk_vtu_to_csv(anatomy_subject_name, geometric_data_dir, target_resolution, vtk_filename):
    input_dir = geometric_data_dir + anatomy_subject_name + '/'
    output_dir = geometric_data_dir + anatomy_subject_name + '/' + anatomy_subject_name + '_' + target_resolution + '/'
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
    if tag_array_name not in surface.cell_data:
        print(f"ERRORE: Campo '{tag_array_name}' non trovato. Campi disponibili: {surface.cell_data.keys()}")
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

if __name__ == "__main__":
    geometric_data_dir = './cardiac-data/meta_data/geometric_data/'
    subject_name = 'kaggle502'
    target_resolution = 'subdiv2'
    vtu_filename = 'kaggle502cobiveco.vtu'
    save_vtk_vtu_to_csv(subject_name, geometric_data_dir, target_resolution, vtu_filename)
    save_vtu_arrays_to_csv(subject_name, geometric_data_dir, target_resolution, vtu_filename)
    save_endo_nodes_to_csv(subject_name, geometric_data_dir, target_resolution, vtu_filename, lv_tag=4, rv_tag=2, tag_array_name='class')