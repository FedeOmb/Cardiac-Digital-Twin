import os
import pyvista as pv
import numpy as np
import open3d as o3d
import copy

 # array (3,3): [apice, centroide_RV, centroide_LV]
def landmark_based_init(biv_landmarks, torso_landmarks):
    """
    Calcola T_init da 3 coppie di landmark anatomici.
    Usa SVD per trovare la rotazione ottimale (metodo Kabsch).
    """
    # Centroidi
    src_c = biv_landmarks.mean(axis=0)
    tgt_c = torso_landmarks.mean(axis=0)

    # Matrici centrate
    A = biv_landmarks   - src_c
    B = torso_landmarks - tgt_c

    # SVD → rotazione ottimale (metodo Kabsch)
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Correggi riflessione (det deve essere +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = tgt_c - R @ src_c

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = t
    return T

def get_ventricle_centroids(mesh):
    """Estrae centroidi RV e LV dai tag della mesh."""
    tags = mesh.cell_data["elemTag"]
    
    rv_cells = np.where(np.isin(tags, [34]))[0]  # tag RV
    lv_cells = np.where(np.isin(tags, [35]))[0]  # tag LV

    def cells_centroid(mesh, cell_ids):
        pts_idx = set()
        for cid in cell_ids:
            pts_idx.update(mesh.get_cell(cid).point_ids)
        return mesh.points[list(pts_idx)].mean(axis=0)

    rv_center = cells_centroid(mesh, rv_cells)
    lv_center = cells_centroid(mesh, lv_cells)
    return rv_center, lv_center

def clip_to_basal_plane(mesh, base_normal=None, base_origin=None):
    """
    Ritaglia la mesh con il piano basale del BIV.
    Se base_normal e base_origin non sono specificati,
    li stima automaticamente dal piano di taglio della mesh sorgente.
    """
    if base_normal is None:
        # Stima automatica: il piano basale è perpendicolare all'asse
        # apice-base. L'asse apice-base coincide col primo PC della mesh.
        pts = np.array(mesh.points)
        cov = np.cov((pts - pts.mean(0)).T)
        _, vecs = np.linalg.eigh(cov)
        base_normal = vecs[:, -1]  # asse di maggiore varianza = asse lungo

        # L'origine è il punto con z-score massimo lungo quell'asse
        # (la base è la parte opposta all'apice)
        proj = pts @ base_normal
        base_origin = pts[proj.argmax()]  # punto più "alto" = base

    clipped = mesh.clip(
        normal=base_normal.tolist(),
        origin=base_origin.tolist(),
        invert=False  # mantieni la parte sotto il piano basale
    )
    return clipped, base_normal, base_origin

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    
    o3d.visualization.draw_geometries([source_temp, target_temp], width=1280, height=720)
    
def pv_surface_to_o3d(pv_surface):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pv_surface.points))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    # Orienta le normali in modo coerente (tutte verso l'esterno)
    pcd.orient_normals_consistent_tangent_plane(k=15)    
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
        ransac_n=4,
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
    heart_from_torso_surface_pts = np.array(heart_from_torso_surface.points)
    heart_from_torso_surface.save("target_heart_surface_from_torso.vtk")

    # Torso senza cuore (per la sostituzione)
    torso_no_cardiac = torso.extract_cells(np.where(~heart_mask)[0])

    # base_normal e base_origin vengono stimati dal BIV e applicati al cuore del torso
    _, base_normal, base_origin = clip_to_basal_plane(biv_surface)

    # Applica lo stesso piano di taglio al torso
    heart_surface_clipped, _, _ = clip_to_basal_plane(
        heart_from_torso_surface,
        base_normal=base_normal,
        base_origin=base_origin
    )
    # Landmark-based T_init
    rv_biv, lv_biv = get_ventricle_centroids(biv_mesh)
    rv_torso, lv_torso = get_ventricle_centroids(heart_mesh)

    apex_biv = np.array(biv_surface.points)[np.array(biv_surface.points) @ base_normal == (np.array(biv_surface.points) @ base_normal).min()][0]
    apex_torso = np.array(heart_surface_clipped.points)[np.array(heart_surface_clipped.points) @ base_normal == (np.array(heart_surface_clipped.points) @ base_normal).min()][0]

    biv_landmarks   = np.array([apex_biv,   rv_biv,   lv_biv])
    torso_landmarks = np.array([apex_torso, rv_torso, lv_torso])
    T_init = landmark_based_init(biv_landmarks, torso_landmarks)

    # Surface del cuore nel torso (target) e del biv (source)
    source_pcd = pv_surface_to_o3d(biv_mesh.extract_surface())
    target_pcd = pv_surface_to_o3d(heart_from_torso_surface)
    
    eval_init = o3d.pipelines.registration.evaluate_registration(
        source_pcd, target_pcd,
        max_correspondence_distance=voxel_size * 1.5,
        transformation=T_init
    )

    print(f"Fitness landmark init: {eval_init.fitness:.4f}")

    draw_registration_result(source_pcd, target_pcd, T_init)

    '''
    T_global_ransac = global_registration_fpfh(source_pcd, target_pcd, voxel_size)
    draw_registration_result(source_pcd, target_pcd, T_global_ransac)
    #T_init_from_pca = coarse_alignment(biv_mesh_surface_pts, heart_from_torso_surface_pts)
    biv_after_ransac = biv_mesh.transform(T_global_ransac)
    #biv_after_ransac.save("503kaggle500_ransac.vtk") 

    normals = np.asarray(source_pcd.normals)
    print("Num Normali:", normals.shape[0])
    print("normali zero:", np.sum(np.linalg.norm(normals, axis=1) < 1e-6))
    print("norma media:", np.linalg.norm(normals, axis=1).mean())

    # ICP point-to-plane
    print("Local registration using ICP...")
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance=voxel_size * 0.4,
        init=T_global_ransac,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=3000, relative_fitness=1e-6, relative_rmse=1e-6)
    )

    T_final = result.transformation
    print(f"ICP fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.3f}")
    draw_registration_result(source_pcd, target_pcd, T_final)

    # Applica trasformazione al biventricolo
    biv_registered = biv_mesh.transform(T_final)
    #biv_registered.save("503kaggle500_registered.vtk") 
    # Unisci torso (senza cuore) + biv registrato
    merged = torso_no_cardiac.merge(biv_registered)
    #merged.save("torso_heart_merged.vtk")
'''