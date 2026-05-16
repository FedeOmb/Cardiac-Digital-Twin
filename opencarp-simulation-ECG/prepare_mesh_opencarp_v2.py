import pyvista as pv
import numpy as np
import vtk
import sys

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
    from cobiveco_export_functions import scale_mesh_mantaining_point_data, scale_vtksurfmesh, tag_vol_region_from_tm, tag_vol_region_from_tm_fastendo
    TAG_LV_ENDO = 3  # Valore per Endocardio LV
    TAG_RV_ENDO = 4  # Valore per Endocardio RV
    TAG_ARRAY_NAME = 'class'
    # Esempio di scaling (se necessario)
    geometric_data_dir = "../cardiac-data/meta_data/geometric_data/"
    subject_name = 'sb3901'
    input_vtp_filename = 'sb3901_500mm.vtp'
    input_dir = geometric_data_dir + subject_name + '/'
    input_vtp_path = input_dir + input_vtp_filename
    scaled_out_vtp_filename = 'sb3901_500um.vtp'
    scaled_out_vtp_path = input_dir + scaled_out_vtp_filename
    scale_mesh_mantaining_point_data(input_vtp_path, scaled_out_vtp_path, scale=1000)
    tagged_filename = 'sb3901_500um_tagged_fastendo.vtk'
    tagged_out_path = input_dir + tagged_filename
    #tag_vol_region_from_tm(scaled_out_path, tagged_out_path)
    input_vtk_filename = 'sb3901_500um_fibers.vtk'
    input_vtk_path = input_dir + input_vtk_filename
    tag_vol_region_from_tm_fastendo(input_vtk_path, tagged_out_path, scaled_out_vtp_path, TAG_LV_ENDO, TAG_RV_ENDO, TAG_ARRAY_NAME, fastendo_thickness=400.0)

    #scala coordinate mesh torso da mm a um
    torso_input_filename = 'sb301_torsomm.vtk'
    torso_input_path = './sb301_torsomesh/' + torso_input_filename
    torso_scaled_out_filename = 'sb301_torsoum.vtk'
    torso_scaled_out_path = './sb301_torsomesh/' + torso_scaled_out_filename

    #scale_vtksurfmesh(torso_input_path, torso_scaled_out_path, scale=1000)

    electrodes_input_filename = 'sb4101_ECG_ELECTRODESmm.vtk'
    electrodes_input_path = input_dir + electrodes_input_filename
    electrodes_output_filename = 'sb4101_electrodes_opencarp.pts'
    electrodes_output_path = input_dir + electrodes_output_filename
    #vtk_poly_to_pts(electrodes_input_path, electrodes_output_path, scale=1000)