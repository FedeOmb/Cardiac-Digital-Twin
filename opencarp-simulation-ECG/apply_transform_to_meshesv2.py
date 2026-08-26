import os
from os import path
import pyvista as pv
import numpy as np
import open3d as o3d
import copy
import vtk


if __name__ == '__main__':
    case_name = "sb4201"
    input_folder = os.path.join("..", "cardiac-data", "meta_data","geometric_data","sb4201_not_reg")
    output_folder = os.path.join("..", "cardiac-data", "meta_data","geometric_data","sb4201")
    transform_file = os.path.join(input_folder, f"{case_name}_icp_transform.npy")
    mesh_files = [
        f"{case_name}_500mm_fibers.vtk",
        f"{case_name}_500mm_fibers.vtu",
        f"{case_name}_500mm.vtp",
        f"{case_name}_1500mm_fibers.vtk",
        f"{case_name}_1500mm_fibers.vtu",
        f"{case_name}_1500mm.vtp"    
    ]
    T = np.load(str(transform_file))
    
    for mesh_file in mesh_files:
        input_path = os.path.join(input_folder, mesh_file)
        output_path = os.path.join(output_folder, mesh_file.replace(".vtk", "_aligned.vtk").replace(".vtu", "_aligned.vtu").replace(".vtp", "_aligned.vtp"))
        print(f"Processing {input_path} -> {output_path}")
        mesh = pv.read(input_path)
        pts = mesh.points
        if pts is None or pts.size == 0:
            print(f"Mesh {input_path} has no points to transform. Skipping.")
            continue
        mesh_registred = mesh.copy().transform(T)
        mesh_registred.save(output_path)

