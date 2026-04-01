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

biv_mesh = pv.read(PATH_CARDIAC_MESH)
biv_mesh_tags = biv_mesh.cell_data["elemTag"]
biv_mesh_pts  = np.array(biv_mesh.points)
biv_mesh_surface_pts = np.array(biv_mesh.extract_surface().points)

# Estrae solo gli elementi cardiaci (34,35)
heart_mask = np.isin(tags, [34, 35])
heart_mesh = torso.extract_cells(np.where(heart_mask)[0])
# Estrae superficie del cuore (per registrazione ICP)
heart_from_torso_surface = heart_mesh.extract_surface() 
heart_from_torso_surface_pts = np.array(heart_from_torso_surface.points)
heart_from_torso_surface.save("target_heart_surface_from_torso.vtk")

# Torso senza cuore (per la sostituzione)
torso_no_cardiac = torso.extract_cells(np.where(~heart_mask)[0])

def pv_surface_to_o3d(pv_surface):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pv_surface.points))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30)
    )
    return pcd

# PRE ALLINEAMENTO CON PCA
def coarse_alignment(source_pts, target_pts):
    source_center = source_pts.mean(axis=0)
    target_center = target_pts.mean(axis=0)

    def pca_axes(pts):
        cov = np.cov((pts - pts.mean(0)).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        return eigvecs[:, ::-1]
    
    R = pca_axes(target_pts) @ pca_axes(source_pts).T
    T_init = np.eye(4)
    T_init[:3, :3] = R
    T_init[:3, 3] = target_center - R @ source_center
    return T_init

# Surface del cuore nel torso (target) e del biv (source)
source_pcd = pv_surface_to_o3d(biv_mesh.extract_surface())
target_pcd = pv_surface_to_o3d(heart_from_torso_surface)

T_init_from_pca = coarse_alignment(biv_mesh_surface_pts, heart_from_torso_surface_pts)

# ICP point-to-plane
result = o3d.pipelines.registration.registration_icp(
    source_pcd, target_pcd,
    max_correspondence_distance=5.0,
    init=T_init_from_pca,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000, relative_fitness=1e-6, relative_rmse=1e-6)
)

T_final = result.transformation
print(f"ICP fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.3f} mm")

# Applica trasformazione al biventricolo
biv_registered = biv_mesh.transform(T_final)
biv_registered.save("503kaggle500_registered.vtk") 
# Unisci torso (senza cuore) + biv registrato
merged = torso_no_cardiac.merge(biv_registered)
merged.save("torso_heart_merged.vtk")

# Riconverti in CARP per openCARP
# meshtool preserva i cell_data come .elem + tag