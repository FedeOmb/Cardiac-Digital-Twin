import os
import pyvista as pv
import numpy as np
import open3d as o3d

PATH_CARDIAC_MESH = os.path.join(".", "503kaggle500_meshes", "503kaggle500um_tagged.vtk")
PATH_TORSO_MESH = os.path.join(".", "KCL_torso1", "KCL_torso1.vtk")

torso = pv.read(PATH_TORSO_MESH)

print(torso.cell_data.keys())   # [elemTag]
tags = torso.cell_data["elemTag"]   # array numpy shape (n_cells,)

pts = np.array(torso.points)        # shape (n_pts, 3)

biv = pv.read(PATH_CARDIAC_MESH)
biv_tags = biv.cell_data["elemTag"]
biv_pts  = np.array(biv.points)

# Estrae solo gli elementi cardiaci (34,35)
heart_mask = np.isin(tags, [34, 35])
heart_mesh = torso.extract_cells(np.where(heart_mask)[0])
# Estrae superficie del cuore (per registrazione ICP)
heart_surface = heart_mesh.extract_surface() 

# Torso senza cuore (per la sostituzione)
torso_no_cardiac = torso.extract_cells(np.where(~heart_mask)[0])

def pv_surface_to_o3d(pv_surface):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pv_surface.points))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30)
    )
    return pcd

# Surface del cuore nel torso (target) e del biv (source)
target_pcd = pv_surface_to_o3d(heart_surface)
source_pcd = pv_surface_to_o3d(biv.extract_surface())

# ICP point-to-plane
result = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd,
    max_correspondence_distance=5.0,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
)
T = result.transformation  # matrice 4x4
print(f"ICP fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.3f} mm")

# Applica trasformazione al biventricolo (in-place su pyvista mesh)
biv_registered = biv.transform(T)

# Unisci torso (senza cuore) + biv registrato
merged = torso_no_cardiac.merge(biv_registered)
merged.save("torso_heart_merged.vtk")

# Riconverti in CARP per openCARP
# meshtool preserva i cell_data come .elem + tag