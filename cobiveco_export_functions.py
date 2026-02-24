import numpy as np
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
                   + target_resolution + '_xyz.csv', unprocessed_node_xyz, delimiter=',')
        n_tetra = data.GetNumberOfCells()
        unprocessed_tetra = np.reshape(VN.vtk_to_numpy(data.GetCells().GetConnectivityArray()), [n_tetra, 4])
        print(unprocessed_tetra.shape)
        # Save tetra
        np.savetxt(output_dir + anatomy_subject_name + '_' + target_resolution + '_tetra.csv', unprocessed_tetra, delimiter=',')

def save_vtu_attributes_to_csv(anatomy_subject_name, geometric_data_dir, target_resolution, vtu_filename):
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
        np.savetxt(os.path.join(output_dir, anatomy_subject_name + '_' + target_resolution + '_' + array_name + '.csv'), np_array, delimiter=',')

    # Cell Data
""" cell_data = data.GetCellData()
    for i in range(cell_data.GetNumberOfArrays()):
        array_name = cell_data.GetArrayName(i)
        array = cell_data.GetArray(i)
        np_array = VN.vtk_to_numpy(array)
        np.savetxt(os.path.join(output_dir, array_name + '.csv'), np_array, delimiter=',')
     """

if __name__ == "__main__":
    geometric_data_dir = './cardiac-data/meta_data/geometric_data/'
    subject_name = 'kaggle502'
    target_resolution = 'subdiv2'
    vtu_filename = 'kaggle502cobiveco.vtu'
    save_vtk_vtu_to_csv(subject_name, geometric_data_dir, target_resolution, vtu_filename)
    save_vtu_attributes_to_csv(subject_name, geometric_data_dir, target_resolution, vtu_filename)