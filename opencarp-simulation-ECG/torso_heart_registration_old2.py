import os
import pyvista as pv
import numpy as np
import open3d as o3d
import copy
import vtk

def clip_at_long_axis_percentile(mesh, percentile=92):
    """
    Trova l'asse lungo della mesh tramite PCA sui suoi punti,
    indipendentemente dall'orientamento nello spazio.
    Taglia al percentile specificato lungo quell'asse.
    """
    pts = np.array(mesh.points)
    centroid = pts.mean(axis=0)

    # PCA: autovettore con varianza massima = asse lungo (apice-base)
    cov = np.cov((pts - centroid).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    long_axis = eigenvectors[:, -1]   # eigenvalue massimo

    print(f"Asse lungo stimato (PCA): {long_axis.round(4)}")
    print(f"Varianza spiegata: {eigenvalues[-1]/eigenvalues.sum()*100:.1f}%")

    # Proietta tutti i punti sull'asse lungo
    proj = pts @ long_axis

    print(f"Proiezioni: min={proj.min():.1f}, max={proj.max():.1f}, "
          f"range={proj.max()-proj.min():.1f}")

    # Quota di taglio
    cut_val = np.percentile(proj, percentile)
    cut_origin = long_axis * cut_val

    print(f"Taglio al {percentile}° percentile: quota={cut_val:.1f}")
    print(f"Punti che verranno rimossi: "
          f"{(proj > cut_val).sum()} / {len(pts)} "
          f"({100*(proj > cut_val).mean():.1f}%)")

    clipped = mesh.clip(
        normal=long_axis.tolist(),
        origin=cut_origin.tolist(),
        invert=True
    )
    '''
    if (proj > cut_val).sum() == 0:
        print("⚠ Nessun punto rimosso — prova invert=True")
        clipped = mesh.clip(
            normal=long_axis.tolist(),
            origin=cut_origin.tolist(),
            invert=True
        )
    '''
    print(f"Punti: {len(pts)} → {len(clipped.points)}")
    return clipped, long_axis, cut_val

def clip_mesh_given_axis(mesh, long_axis, percentile):
    # Proietta tutti i punti sull'asse lungo
    pts_mesh = np.array(mesh.points)

    proj = pts_mesh @ long_axis

    print(f"Proiezioni: min={proj.min():.1f}, max={proj.max():.1f}, "
          f"range={proj.max()-proj.min():.1f}")

    # Quota di taglio
    cut_val = np.percentile(proj, percentile)
    cut_origin = long_axis * cut_val

    print(f"Taglio al {percentile}° percentile: quota={cut_val:.1f}")
    print(f"Punti che verranno rimossi: "
          f"{(proj > cut_val).sum()} / {len(pts)} "
          f"({100*(proj > cut_val).mean():.1f}%)")

    clipped = mesh.clip(
        normal=long_axis.tolist(),
        origin=cut_origin.tolist(),
        invert=True
    )
    print(f"Punti: {len(pts)} → {len(clipped.points)}")
    return clipped

def get_long_axis_from_ab(biv_mesh):
    """
    Calcola il vettore apice→base dalla coordinata ab del BIV.
    Più affidabile della PCA perché non dipende dalla geometria superficiale.
    """
    pts = np.array(biv_mesh.points)
    ab  = np.array(biv_mesh.point_data["ab"]).ravel()

    apex_center = pts[ab < 0.05].mean(axis=0)
    base_center = pts[ab > 0.95].mean(axis=0)

    long_axis = base_center - apex_center
    long_axis = long_axis / np.linalg.norm(long_axis)

    print(f"Asse apice→base dal BIV (ab): {long_axis.round(4)}")
    print(f"  Apice: {apex_center.round(0)}")
    print(f"  Base:  {base_center.round(0)}")
    return long_axis

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    
    o3d.visualization.draw_geometries([source_temp, target_temp], width=1280, height=720)
    
def pv_surface_to_o3d(pv_surface):
    #calcolo normali superficie con pyvista
    surf_with_normals = pv_surface.compute_normals(
        cell_normals=False, 
        point_normals=True, 
        auto_orient_normals=True
    )
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(surf_with_normals.points))
    normals = np.array(surf_with_normals.point_data["Normals"])
    pcd.normals = o3d.utility.Vector3dVector(normals)
    #pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    #pcd.orient_normals_consistent_tangent_plane(k=15)    
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

def global_registration_fpfh(source_pcd, target_pcd, voxel_size):

    def preprocess_point_cloud(pcd, voxel_size):
        pcd_down = pcd.voxel_down_sample(voxel_size)
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normal, max_nn=30)      
        )
        radius_feature = voxel_size * 5
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, 
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_feature, max_nn=100)      
        )
        return pcd_down, fpfh

    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)

    dist_threshold = voxel_size * 1.5
    print("Global registration using RANSAC...")
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=dist_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    #FAST RANSAC
    #dist_threshold = voxel_size * 0.5
    #print("Global registration using Fast RANSAC...")
    result_fast_ransac = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=dist_threshold))

    print(f"RANSAC fitness: {result_ransac.fitness:.4f}, RMSE: {result_ransac.inlier_rmse:.2f}")
    return result_ransac.transformation

def export_mesh_vtk42(mesh, filename):
    # Esporta mesh in formato VTK 4.2 (legacy) con pyvista
    # Per compatibilità con meshtool

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(filename)
    writer.SetInputData(mesh)
    writer.SetFileVersion(42)
    writer.SetFileTypeToASCII()
    writer.Write()
    print("mesh esportata in VTK 4.2: ", filename)

if __name__ == "__main__":
    PATH_CARDIAC_MESH = os.path.join(".", "503kaggle500_meshes", "503kaggle500um_tagged.vtk")
    PATH_TORSO_MESH = os.path.join(".", "KCL_torso1", "KCL_torso1.vtk")
    voxel_size = 3000  # in um 

    torso = pv.read(PATH_TORSO_MESH)

    print(torso.cell_data.keys())   # [elemTag]
    tags = torso.cell_data["elemTag"]   # array numpy shape (n_cells,)

    pts = np.array(torso.points)        # shape (n_pts, 3)

    biv_mesh = pv.read(PATH_CARDIAC_MESH)
    biv_mesh_tags = biv_mesh.cell_data["elemTag"]
    biv_mesh_pts  = np.array(biv_mesh.points)
    biv_surface = biv_mesh.extract_surface()
    biv_mesh_surface_pts = np.array(biv_surface.points)

    # Estrae solo gli elementi cardiaci (34=RV, 35=LV)
    heart_mask = np.isin(tags, [34, 35])
    heart_mesh = torso.extract_cells(np.where(heart_mask)[0])
    # Estrae superficie del cuore (per registrazione ICP)
    heart_from_torso_surface = heart_mesh.extract_surface() 
    heart_from_torso_surface.save("heart_from_torso_surface.vtk")

    long_axis_from_biv = get_long_axis_from_ab(biv_surface)

    heart_from_torso_clipped = clip_mesh_given_axis(heart_mesh, long_axis_from_biv, percentile=70)

    # Estrae superficie del cuore (per registrazione ICP)
    heart_from_torso_clipped_surf = heart_from_torso_clipped.extract_surface() 
    heart_from_torso_clipped_surf.save("heart_from_torso_surface_clipped.vtk.vtk")

    # Torso senza cuore (per la sostituzione)
    torso_no_cardiac = torso.extract_cells(np.where(~heart_mask)[0])
    #torso_no_cardiac.save("torso_no_biv_mesh.vtk", binary=False)
    export_mesh_vtk42(torso_no_cardiac, "torso_no_biv_mesh.vtk")

    source_pcd = pv_surface_to_o3d(biv_surface)
    target_pcd = pv_surface_to_o3d(heart_from_torso_clipped_surf)

    T_global_ransac = global_registration_fpfh(source_pcd, target_pcd, voxel_size)
    draw_registration_result(source_pcd, target_pcd, T_global_ransac)
    #T_init_from_pca = coarse_alignment(biv_mesh_surface_pts, heart_from_torso_surface_pts)
    biv_after_ransac = biv_mesh.copy().transform(T_global_ransac)
    #biv_after_ransac.save("503kaggle500_after_ransac.vtk") 
    export_mesh_vtk42(biv_after_ransac, "503kaggle500_after_ransac.vtk")

    normals = np.asarray(source_pcd.normals)
    print("Num Normali:", normals.shape[0])
    print("normali zero:", np.sum(np.linalg.norm(normals, axis=1) < 1e-6))
    print("norma media:", np.linalg.norm(normals, axis=1).mean())

    # ICP point-to-plane
    
    print("Local registration using ICP...")
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance=voxel_size * 0.8,
        init=T_global_ransac,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )

    T_final = result.transformation
    print(f"ICP fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.3f}")
    draw_registration_result(source_pcd, target_pcd, T_final)

    # Applica trasformazione al biventricolo
    biv_registered = biv_mesh.copy().transform(T_final)
    #biv_registered.save("503kaggle500_reg_icp.vtk", binary=False) 
    export_mesh_vtk42(biv_registered, "503kaggle500_reg_icp.vtk")
    # Unisci torso (senza cuore) + biv registrato
    #merged = torso_no_cardiac.merge(biv_registered)
    #merged.save("torso_heart_merged.vtk")
