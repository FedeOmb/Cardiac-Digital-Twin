import pyvista as pv
import numpy as np
import sys
import os
import shutil
import subprocess

def vtk_poly_to_pts(input_vtk, output_pts, scale=1000.0, fmt="%.6f"):
    mesh = pv.read(input_vtk)
    if mesh.n_points == 0:
        raise RuntimeError(f"No points found in {input_vtk}")
    pts = mesh.points * scale
    with open(output_pts, "w") as f:
        f.write(f"{pts.shape[0]}\n")
        np.savetxt(f, pts, fmt=fmt + " " + fmt + " " + fmt)

if __name__ == "__main__":
    import os
    import sys
    import pyvista as pv
    from cobiveco_export_functions import scale_mesh_mantaining_point_data, tag_vol_region_from_tm, scale_vtksurfmesh

    # Esempio di scaling (se necessario)
    geometric_data_dir = os.path.join('..',"cardiac-data","meta_data","geometric_data")
    subject_name = 'sb4101'
   # input_filename = 'sb301_coarse1500_cmcoord.vtu'
    sim_folder = os.path.join(".", subject_name+'sim_input')
    sim_meshes_folder = os.path.join(sim_folder, subject_name+'meshes')
    sim_rn_folder = os.path.join(sim_folder, subject_name+'rootnodes')

    if not os.path.exists(sim_folder):
        os.makedirs(sim_folder)
    if not os.path.exists(sim_meshes_folder):
        os.makedirs(sim_meshes_folder)
    if not os.path.exists(sim_rn_folder):
        os.makedirs(sim_rn_folder)
        
    input_dir = os.path.join(geometric_data_dir, subject_name)
   # input_path = input_dir + input_filename
    scaled_out_filename = subject_name + '_500um_fibers.vtk'
    scaled_out_path = os.path.join(input_dir, scaled_out_filename)
    #scale_mesh_mantaining_point_data(input_path, scaled_out_path, scale=10000)
    tagged_filename = subject_name + '_500um_tagged'
    tagged_out_path = os.path.join(input_dir, tagged_filename+'.vtk')
    tag_vol_region_from_tm(scaled_out_path, tagged_out_path)

    shutil.move( tagged_out_path, os.path.join(sim_meshes_folder, tagged_filename+'.vtk'))

    cmd_convert =[
        'opencarp-docker',
        'meshtool', 'convert',
        '-imsh=' + os.path.join(sim_meshes_folder, tagged_filename+'.vtk'),
        '-omsh=' + os.path.join(sim_meshes_folder, tagged_filename),
        '-ofmt=carp_txt',
    ] 
    print("Running command:", " ".join(cmd_convert))
    subprocess.run(cmd_convert, check=True)
    lon_file_path = os.path.join(sim_meshes_folder, tagged_filename+'.lon')
    if os.path.exists(lon_file_path):
        os.remove(lon_file_path)
    shutil.copy( os.path.join(input_dir, subject_name + '_500mm_fibersOpencarp.lon'), lon_file_path)
    
    
    #scala coordinate mesh torso da mm a um
    torso_input_filename = 'sb3901_TORSOmm.vtk'
    torso_input_path = './sb3901_torso/' + torso_input_filename
    torso_scaled_out_filename = 'sb3901_torsoum.vtk'
    torso_scaled_out_path = './sb3901_torso/' + torso_scaled_out_filename

   # scale_vtksurfmesh(torso_input_path, torso_scaled_out_path, scale=1000)

    electrodes_input_filename = 'ECG_ELECTRODES.vtk'
    electrodes_input_path = './sb4101_meshes/' + electrodes_input_filename
    electrodes_output_filename = 'sb4101_electrodes_um.pts'
    electrodes_output_path = './sb4101_meshes/' + electrodes_output_filename
#    vtk_poly_to_pts(electrodes_input_path, electrodes_output_path, scale=1000)