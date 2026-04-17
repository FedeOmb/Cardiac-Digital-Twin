import pyvista as pv
import numpy as np
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
    from cobiveco_export_functions import scale_mesh_mantaining_point_data, tag_vol_region_from_tm, scale_vtksurfmesh

    # Esempio di scaling (se necessario)
    geometric_data_dir = './'
    subject_name = 'sb301_meshes'
    input_filename = 'sb301_coarse1500_cmcoord.vtu'
    input_dir = geometric_data_dir + subject_name + '/'
    input_path = input_dir + input_filename
    scaled_out_filename = 'sb301_fine500um.vtk'
    scaled_out_path = input_dir + scaled_out_filename
    #scale_mesh_mantaining_point_data(input_path, scaled_out_path, scale=10000)
    tagged_filename = 'sb301_fine500um_tagged.vtk'
    tagged_out_path = input_dir + tagged_filename
    #tag_vol_region_from_tm(scaled_out_path, tagged_out_path)

    #scala coordinate mesh torso da mm a um
    torso_input_filename = 'sb301_torsomm.vtk'
    torso_input_path = './sb301_torsomesh/' + torso_input_filename
    torso_scaled_out_filename = 'sb301_torsoum.vtk'
    torso_scaled_out_path = './sb301_torsomesh/' + torso_scaled_out_filename

    #scale_vtksurfmesh(torso_input_path, torso_scaled_out_path, scale=1000)

    electrodes_input_filename = 'sb301_ECG_ELECTRODESmm.vtk'
    electrodes_input_path = './sb301_torsomesh/' + electrodes_input_filename
    electrodes_output_filename = 'sb301_electrodes_opencarp.pts'
    electrodes_output_path = './sb301_torsomesh/' + electrodes_output_filename
    vtk_poly_to_pts(electrodes_input_path, electrodes_output_path, scale=1000)