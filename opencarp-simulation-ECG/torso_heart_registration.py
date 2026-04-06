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

def get_basal_plane_from_ab(biv_mesh, ab_threshold=0.99):
    """
    Stima il piano basale dalla coordinata apicobasale 'ab'.
    Usa i nodi con ab > ab_threshold per fit del piano con PCA.
    """
    pts = np.array(biv_mesh.points)
    ab  = np.array(biv_mesh.point_data["ab"]).ravel()

    print(f"Attributo 'ab': min={ab.min():.4f}, max={ab.max():.4f}")
    print(f"Nodi con ab > {ab_threshold}: {(ab > ab_threshold).sum()}")

    basal_pts = pts[ab > ab_threshold]

    if len(basal_pts) < 10:
        raise ValueError(
            f"Troppo pochi punti basali ({len(basal_pts)}) con soglia {ab_threshold}. "
            f"Riduci ab_threshold (es. 0.90)."
        )

    # Fit del piano con PCA sui punti basali
    centroid = basal_pts.mean(axis=0)
    cov = np.cov((basal_pts - centroid).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # La normale al piano = autovettore con varianza MINIMA (piano di best fit)
    base_normal = eigenvectors[:, 0]   # minore eigenvalue = normale al piano

    # Convenzione: la normale deve puntare verso l'apice (ab=0)
    apex_pt = pts[ab.argmin()]
    if np.dot(base_normal, apex_pt - centroid) > 0:
        base_normal = -base_normal   # inverti se punta nella direzione sbagliata

    print(f"Centroide piano basale: {centroid.round(1)}")
    print(f"Normale piano basale:   {base_normal.round(4)}")
    print(f"Planarità punti basali (eigenvalue minore): {eigenvalues[0]:.2f} "
          f"(più basso = più piano)")

    return base_normal, centroid

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

def clip_torso_to_biv_base(torso_heart_surface, base_normal, base_height, margin_factor=1.05):
    """
    Taglia la mesh del torso allo stesso piano basale del BIV.
    margin_factor > 1.0 include una piccola margine oltre la base del BIV
    per non perdere la geometria di raccordo.
    """
    pts = np.array(torso_heart_surface.points)
    torso_proj = pts @ np.array(base_normal)

    print(f"\nMesh torso lungo asse basale:")
    print(f"  min={torso_proj.min():.1f}, max={torso_proj.max():.1f}")
    print(f"  Piano di taglio a: {base_height * margin_factor:.1f}")

    # L'origine del piano viene spostata alla quota basale del BIV (+ margine)
    clip_origin = np.array(base_normal) * base_height * margin_factor

    clipped = torso_heart_surface.clip(
        normal=(-np.array(base_normal)).tolist(),  # inverte: tieni la parte SOTTO
        origin=clip_origin.tolist(),
        invert=False
    )

    n_before = len(torso_heart_surface.points)
    n_after  = len(clipped.points)
    print(f"  Punti prima: {n_before}, dopo taglio: {n_after} "
          f"({100*n_after/n_before:.1f}% mantenuto)")

    if n_after < 100:
        raise ValueError(
            "Il taglio ha rimosso troppi punti. "
        )
    return clipped

def clip_mesh_to_basal_planev2(mesh, base_normal, base_origin, invert=False):
    """
    Taglia la mesh del torso con il piano basale estratto dal BIV.
    La normale punta verso l'apice → clip taglia tutto sopra la base.
    """
    # Verifica quanti punti del torso cadono su ciascun lato del piano
    pts   = np.array(mesh.points)
    dists = (pts - base_origin) @ base_normal   # positivo = lato apice
    print(f"\nDistribuizione punti torso rispetto al piano:")
    print(f"  Lato apice  (d > 0): {(dists > 0).sum()} punti")
    print(f"  Lato base   (d < 0): {(dists < 0).sum()} punti")

    clipped = mesh.clip(
        normal=base_normal.tolist(),
        origin=base_origin.tolist(),
        invert=invert
    )

    if len(clipped.points) < 50:
        print("Troppo pochi punti dopo il taglio — prova con invert=True")
        clipped = mesh.clip(
            normal=base_normal.tolist(),
            origin=base_origin.tolist(),
            invert=not invert
        )

    print(f"Punti torso: {len(mesh.points)} → {len(clipped.points)} "
          f"({100*len(clipped.points)/len(mesh.points):.1f}% mantenuto)")
    return clipped

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
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(pv_surface.points))
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30)
    )
    # Orienta le normali in modo coerente (tutte verso l'esterno)
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


if __name__ == "__main__":
    PATH_CARDIAC_MESH = os.path.join(".", "503kaggle500_meshes", "503kaggle500um_tagged.vtk")
    PATH_TORSO_MESH = os.path.join(".", "KCL_torso1", "KCL_torso1.vtk")
    voxel_size = 2000  # in um 

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

    # base_normal e base_origin vengono stimati dal BIV e applicati al cuore del torso
    #base_normal, base_origin = get_basal_plane_from_ab(biv_surface)
    #print("Base normal stimata da mesh biv:", base_normal)
    #print("Base origin stimata da mesh biv:", base_origin)

    # Applica lo stesso piano di taglio al torso

    #heart_from_torso_surf_clipped = clip_torso_to_biv_base(heart_from_torso_surface, base_normal, base_height)
    #heart_from_torso_surf_clipped= clip_mesh_to_basal_planev2(heart_from_torso_surface, base_normal, base_origin)
    #long_axis_from_biv = get_long_axis_from_ab(biv_surface)
    #heart_from_torso_surf_clipped = clip_mesh_given_axis(heart_from_torso_surface, long_axis_from_biv, percentile=70)
    #heart_from_torso_surf_clipped.save("heart_from_torso_clipped.vtk")

    '''
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
    source_pcd = pv_surface_to_o3d(biv_surface)
    target_pcd = pv_surface_to_o3d(heart_from_torso_clipped_surf)

    T_global_ransac = global_registration_fpfh(source_pcd, target_pcd, voxel_size)
    draw_registration_result(source_pcd, target_pcd, T_global_ransac)
    #T_init_from_pca = coarse_alignment(biv_mesh_surface_pts, heart_from_torso_surface_pts)
    biv_after_ransac = biv_mesh.transform(T_global_ransac)
    biv_after_ransac.save("503kaggle500_after_ransac.vtk") 

    normals = np.asarray(source_pcd.normals)
    print("Num Normali:", normals.shape[0])
    print("normali zero:", np.sum(np.linalg.norm(normals, axis=1) < 1e-6))
    print("norma media:", np.linalg.norm(normals, axis=1).mean())

    # ICP point-to-plane
    
    print("Local registration using ICP...")
    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd,
        max_correspondence_distance=voxel_size * 0.5,
        init=T_global_ransac,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )

    T_final = result.transformation
    print(f"ICP fitness: {result.fitness:.4f}, RMSE: {result.inlier_rmse:.3f}")
    draw_registration_result(source_pcd, target_pcd, T_final)

    # Applica trasformazione al biventricolo
    biv_registered = biv_mesh.transform(T_final)
    biv_registered.save("503kaggle500_reg_icp.vtk") 
    # Unisci torso (senza cuore) + biv registrato
    #merged = torso_no_cardiac.merge(biv_registered)
    #merged.save("torso_heart_merged.vtk")
