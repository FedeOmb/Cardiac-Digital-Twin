if __name__ == "__main__":
    import os
    import sys
    import pyvista as pv
    from cobiveco_export_functions import scale_mesh_mantaining_point_data, tag_vol_region_from_tm

    # Esempio di scaling (se necessario)
    geometric_data_dir = '/home/federico/opencarp-simulation/'
    subject_name = '503kaggle500_meshes'
    input_filename = '503kaggle500cm_fibers.vtu'
    input_dir = geometric_data_dir + subject_name + '/'
    input_path = input_dir + input_filename
    scaled_out_filename = '503kaggle500um_tagged_moved.vtk'
    scaled_out_path = input_dir + scaled_out_filename
    #scale_mesh_mantaining_point_data(input_path, scaled_out_path, scale=10000)
    tagged_filename = '503kaggle500um_tagged_moved3.vtk'
    tagged_out_path = input_dir + tagged_filename
    tag_vol_region_from_tm(scaled_out_path, tagged_out_path)


