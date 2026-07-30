import pyvista as pv
import numpy as np
import vtk
import sys
import os
import shutil
import subprocess
sys.path.insert(0, "../src")

def vtk_poly_to_pts(input_vtk, output_pts, scale=1000.0, fmt="%.6f"):
    mesh = pv.read(input_vtk)
    if mesh.n_points == 0:
        raise RuntimeError(f"No points found in {input_vtk}")
    pts = mesh.points * scale
    with open(output_pts, "w") as f:
        f.write(f"{pts.shape[0]}\n")
        np.savetxt(f, pts, fmt=fmt + " " + fmt + " " + fmt)

def tag_vol_region_fastendo_dense_sparse(input_vtu, output_vtk, input_vtp, endo_lv_tag, endo_rv_tag, tag_array_name, fastendo_thickness=500.0):
    # carica la mesh volumetrica con le coord cobiveco
    mesh = pv.read(input_vtu)
    #estrai supercifi endocardiche da vtp input
    surf_mesh_all = pv.read(input_vtp)
    endo_surf_ug = surf_mesh_all.threshold(value=[endo_lv_tag, endo_rv_tag], 
                                            scalars=tag_array_name, 
                                            preference='cell')
    endo_surface = endo_surf_ug.extract_surface()  # converte in vtkPolyData
    print(endo_surface.n_points, endo_surface.n_cells)
    print(endo_surface.bounds)
    print(mesh.bounds)
    mesh_with_dist = mesh.compute_implicit_distance(endo_surface, inplace=False)

    # Converti i dati dai nodi alle celle (calcola la media per ogni tetraedro)
    mesh_cells = mesh_with_dist.ptc() # Point-To-Cell data interpolation

    dist_cells  = mesh_cells.cell_data["implicit_distance"]
    tm_cells = mesh_cells.cell_data["tm"]
    ab_cells = mesh_cells.cell_data["ab"]
    rt_cells = mesh_cells.cell_data["rt"]
    tv_cells = mesh_cells.cell_data["tv"]
    # Crea un nuovo array vuoto per i tag volumetrici (es. interi a 32 bit)
    tags = np.zeros(mesh.n_cells, dtype=np.int32)
    print("dist min/max:", dist_cells.min(), dist_cells.max())
    print("abs dist min/max:", np.abs(dist_cells).min(), np.abs(dist_cells).max())
    print("percentili abs dist:", np.percentile(np.abs(dist_cells), [1, 5, 10, 25, 50]))
    print("n fastendo candidate:", np.sum(np.abs(dist_cells) < fastendo_thickness))
    # Tag 1: FastEndo — entro 0.5 mm dalla superficie endo (LV + RV)

    # Assegna gli altri tag alle celle in base al valore transmurale (da 0 a 1)
    fast_endo_mask = (dist_cells >= -fastendo_thickness) & (dist_cells <= 0) & (ab_cells < 0.9)  # fast endocardio completo
    tags[fast_endo_mask] = 4    
    tags[~fast_endo_mask] = 1 #endocardio senza divisione transmurale

    fast_endo_dense_mask_lv = (ab_cells < 0.4) & (tv_cells == 0)
    fast_endo_dense_mask_rv = (ab_cells < 0.2) & (tv_cells == 1)
    fast_endo_dense_mask_freewall = (rt_cells > 0.2) & (rt_cells < 0.5) & (tv_cells == 1)

    tags[fast_endo_dense_mask_lv & fast_endo_mask] = 5    #dense fastendo
    tags[fast_endo_dense_mask_rv & fast_endo_mask] = 5
    tags[fast_endo_dense_mask_freewall & fast_endo_mask] = 5
    # Salva i tag nella mesh come l'array attivo per le "scalars" (classi)
    mesh.cell_data["elemTag"] = tags
    mesh.set_active_scalars("elemTag")

    mesh.cell_data["fibers"] = mesh.cell_data["Fiber"]
    mesh.cell_data["sheet"] = mesh.cell_data["Sheet"]

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetInputData(mesh)
    writer.SetFileName(output_vtk)
    writer.SetFileVersion(42)
    writer.SetFileTypeToASCII()
    writer.Write()
    print(f"Mesh scalata salvata in: {output_vtk}")
    print("Mesh taggata salvata con successo!")

def tag_vol_region_fastendo_onlyendo(input_vtu, output_vtk, input_vtp, endo_lv_tag, endo_rv_tag, tag_array_name, fastendo_thickness=500.0):
    # carica la mesh volumetrica con le coord cobiveco
    mesh = pv.read(input_vtu)
    #estrai supercifi endocardiche da vtp input
    surf_mesh_all = pv.read(input_vtp)
    endo_surf_ug = surf_mesh_all.threshold(value=[endo_lv_tag, endo_rv_tag], 
                                            scalars=tag_array_name, 
                                            preference='cell')
    endo_surface = endo_surf_ug.extract_surface()  # converte in vtkPolyData
    print(endo_surface.n_points, endo_surface.n_cells)
    print(endo_surface.bounds)
    print(mesh.bounds)
    mesh_with_dist = mesh.compute_implicit_distance(endo_surface, inplace=False)

    # Converti i dati dai nodi alle celle (calcola la media per ogni tetraedro)
    mesh_cells = mesh_with_dist.ptc() # Point-To-Cell data interpolation

    dist_cells  = mesh_cells.cell_data["implicit_distance"]
    tm_cells = mesh_cells.cell_data["tm"]
    ab_cells = mesh_cells.cell_data["ab"]
    rt_cells = mesh_cells.cell_data["rt"]
    tv_cells = mesh_cells.cell_data["tv"]
    # Crea un nuovo array vuoto per i tag volumetrici (es. interi a 32 bit)
    tags = np.zeros(mesh.n_cells, dtype=np.int32)
    print("dist min/max:", dist_cells.min(), dist_cells.max())
    print("abs dist min/max:", np.abs(dist_cells).min(), np.abs(dist_cells).max())
    print("percentili abs dist:", np.percentile(np.abs(dist_cells), [1, 5, 10, 25, 50]))
    print("n fastendo candidate:", np.sum(np.abs(dist_cells) < fastendo_thickness))
    # Tag 1: FastEndo — entro 0.5 mm dalla superficie endo (LV + RV)

    # Assegna gli altri tag alle celle in base al valore transmurale (da 0 a 1)
    fast_endo_mask = (dist_cells >= -fastendo_thickness) & (dist_cells <= 0)  # fast endocardio completo
    tags[fast_endo_mask] = 4    
    tags[~fast_endo_mask] = 1 #endocardio senza divisione transmurale

    # fast_endo_dense_mask_lv = (ab_cells < 0.4) & (tv_cells == 0)
    # fast_endo_dense_mask_rv = (ab_cells < 0.2) & (tv_cells == 1)
    # fast_endo_dense_mask_freewall = (rt_cells > 0.2) & (rt_cells < 0.5) & (tv_cells == 1)

    # tags[fast_endo_dense_mask_lv & fast_endo_mask] = 5    #dense fastendo
    # tags[fast_endo_dense_mask_rv & fast_endo_mask] = 5
    # tags[fast_endo_dense_mask_freewall & fast_endo_mask] = 5
    # Salva i tag nella mesh come l'array attivo per le "scalars" (classi)
    mesh.cell_data["elemTag"] = tags
    mesh.set_active_scalars("elemTag")

    mesh.cell_data["fibers"] = mesh.cell_data["Fiber"]
    mesh.cell_data["sheet"] = mesh.cell_data["Sheet"]

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetInputData(mesh)
    writer.SetFileName(output_vtk)
    writer.SetFileVersion(42)
    writer.SetFileTypeToASCII()
    writer.Write()
    print(f"Mesh scalata salvata in: {output_vtk}")
    print("Mesh taggata salvata con successo!")

if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, "../opencarp-simulation-ECG")
    sys.path.insert(0,"../src")
    import pyvista as pv
    from cobiveco_export_functions import scale_mesh_mantaining_point_data, tag_vol_region_from_tm, tag_vol_region_from_tm_fastendo, scale_vtksurfmesh
    from path_config import set_path_mapping, get_path_mapping
    path_mapping = get_path_mapping()

    geometric_data_dir = os.path.join(path_mapping["data_path"],"geometric_data")
    subject_name = 'sb3901'
    sim_folder = os.path.join(".", subject_name)
    sim_meshes_folder = os.path.join(sim_folder, subject_name+'_meshes')
    sim_rn_folder = os.path.join(sim_folder, subject_name+'_rootnodes')

    if not os.path.exists(sim_folder):
        os.makedirs(sim_folder)
    if not os.path.exists(sim_meshes_folder):
        os.makedirs(sim_meshes_folder)
    if not os.path.exists(sim_rn_folder):
        os.makedirs(sim_rn_folder)
    TAG_LV_ENDO = 3  # Valore per Endocardio LV
    TAG_RV_ENDO = 4  # Valore per Endocardio RV
    TAG_ARRAY_NAME = 'class'        
    input_dir = os.path.join(geometric_data_dir, subject_name)
   # input_path = input_dir + input_filename
    input_vtk_filename = subject_name + '_500um_fibers.vtk'
    input_vtk_path = os.path.join(input_dir, input_vtk_filename)
    input_vtp_filename = subject_name + '_500um.vtp'
    input_vtp_path = os.path.join(input_dir, input_vtp_filename)
    #scale_mesh_mantaining_point_data(input_path, scaled_out_path, scale=10000)
    tagged_filename_noext = subject_name + '_500um_tagged_fastendo'
    tagged_out_vtk_path = os.path.join(input_dir, tagged_filename_noext+'.vtk')

    #scaled_out_vtp_filename = 'sb3901_500um.vtp'
    #scaled_out_vtp_path = input_dir + scaled_out_vtp_filename
    #scale_mesh_mantaining_point_data(input_vtp_path, scaled_out_vtp_path, scale=1000)
    tag_vol_region_fastendo_dense_sparse(input_vtk_path, tagged_out_vtk_path, input_vtp_path, TAG_LV_ENDO, TAG_RV_ENDO, TAG_ARRAY_NAME, fastendo_thickness=500.0)

    shutil.move( tagged_out_vtk_path, os.path.join(sim_meshes_folder, tagged_filename_noext+'.vtk'))

    cmd_convert =[
        'opencarp-docker',
        'meshtool', 'convert',
        '-imsh=' + os.path.join(sim_meshes_folder, tagged_filename_noext+'.vtk'),
        '-omsh=' + os.path.join(sim_meshes_folder, tagged_filename_noext),
        '-ofmt=carp_txt'
    ] 
    print("Running command:", " ".join(cmd_convert))
    subprocess.run(cmd_convert, check=True)
    lon_file_path = os.path.join(sim_meshes_folder, tagged_filename_noext+'.lon')
    if os.path.exists(lon_file_path):
        os.remove(lon_file_path)
    shutil.copy( os.path.join(input_dir, subject_name + '_500mm_fibersOpencarp.lon'), lon_file_path)
    
    electrodes_input_filename = subject_name + '_ECG_ELECTRODESmm.vtk'
    electrodes_input_path = os.path.join(input_dir, "torso_meshes",electrodes_input_filename)
    electrodes_output_filename = subject_name + '_electrodes_um.pts'
    electrodes_output_path = os.path.join(sim_meshes_folder, electrodes_output_filename)
    vtk_poly_to_pts(electrodes_input_path, electrodes_output_path, scale=1000)