import os
import numpy as np
import pandas as pd
from carputils import settings
from carputils import tools

#TORORD_LIB_ENDO = './torord-model-opencarp/ToRORd_fkatp_endo.so'
#TORORD_LIB_MID = './torord-model-opencarp/ToRORd_fkatp_endo_mid.so'
#TORORD_LIB_EPI  = './torord-model-opencarp/ToRORd_fkatp_epi.so'

def parser():
    parser = tools.standard_parser()
    parser.add_argument('--meshname',
                        default='./sb301_meshes/sb301_fine500um_opencarp',
                        help='Percorso base della mesh per openCARP')
    parser.add_argument('--vtx-file',
                        default='./sb301_rootnodes/sb301_fine500um_active_root_nodes.vtx',
                        help='File contenente gli indici dei nodi root (candidate_root_nodes.vtx)')
    parser.add_argument('--times-file',
                        default='./sb301_rootnodes/sb301_candidate_root_nodes_times.csv',
                        help='File CSV contenente i tempi di attivazione (candidate_root_nodes_times.csv)')
    parser.add_argument('--tend',
                        type=float, default=200.0,
                        help='Durata della simulazione in ms')
    parser.add_argument('--outdir',
                        default='test_monodomain_torord_sb301_v7',
                        help='Directory di output (simID)')
    return parser

def jobID(args):
    return args.outdir

def get_nodes_by_radius(pts_file, root_node, radius=1500):
    pts = np.loadtxt(pts_file, skiprows=1)  # Salta la prima riga con il numero di punti
    root_coords = pts[root_node]
    distances = np.linalg.norm(pts - root_coords, axis=1)
    return np.where(distances <= radius)[0]

@tools.carpexample(parser, jobID)
def run(args, job):
    # ==========================================
    # 1. Lettura dei nodi e dei tempi di attivazione
    # ==========================================
    print(f"Leggendo i nodi da {args.vtx_file} e i tempi da {args.times_file}...")
    nodes = np.loadtxt(args.vtx_file, dtype=int)  
    
    # Assicuriamoci che nodes sia iterabile anche in caso contenga un solo nodo
    if nodes.ndim == 0:
        nodes = [nodes.item()]

    # Legge il file CSV con pandas e prende la prima colonna (tempi)
    # Gestisce automaticamente la presenza di un header come "activation_time_ms"
    times_df = pd.read_csv(args.times_file)
    times = times_df.iloc[:, 0].values
    
    assert len(nodes) == len(times), "Il numero di nodi deve corrispondere al numero dei tempi di attivazione"

    # ==========================================
    # 2. Generazione dei file .vtx per gli stimoli
    # ==========================================
    vtx_dir = os.path.join(args.outdir, 'sb301_rootnodes_vtx')
    os.makedirs(vtx_dir, exist_ok=True)
    PTS_PATH = './sb301_meshes/sb301_fine500um_opencarp.pts'
    stim_cmds = ['-num_stim', len(nodes)]
    for i, (node, t) in enumerate(zip(nodes, times)):
        nearby_nodes = get_nodes_by_radius(PTS_PATH,node, radius=2000)
        vtx_filename = os.path.join(vtx_dir, f'RN{i+1}.vtx')
        with open(vtx_filename, 'w') as f:
            f.write(f"{len(nearby_nodes)}\n")  # Numero di nodi da stimolare
            f.write("intra\n")  # Header standard di openCARP per i vertici intracellulari
            for n in nearby_nodes:
                f.write(f"{n}\n")

        # Parametri dello stimolo
        stim_cmds += [
            f'-stim[{i}].name', f'RN{i+1}',
            f'-stim[{i}].elec.vtx_file', vtx_filename,
            f'-stim[{i}].ptcl.duration', 4.0, # 4ms da paper camps
            f'-stim[{i}].ptcl.npls', 1,
            f'-stim[{i}].ptcl.start', float(t),
            f'-stim[{i}].pulse.strength', 10.0,  # 5uA/cm2 convertito da paper camps 53 pA/pF x 100 pF/cm2
            f'-stim[{i}].crct.type', 0,
        ]

    # ==========================================
    # 3. Composizione del comando openCARP
    # ==========================================
    cmd = tools.carp_cmd()
    
    # Parametri Generali e Setup della Mesh
    cmd += [
        '-simID', args.outdir,
        '-meshname', args.meshname,
        '-tend', args.tend,
        '-spacedt', 1.0,
        '-timedt', 1.0,
        '-dt', 25.0,
        '-bidomain', 0,
        '-parab_solve', 1,
        '-mass_lumping', 1,
        '-phie_rec_ptf', 'sb301_electrodes_opencarp',
        '-phie_recovery_file', 'sb301_phie_recovery_test7',
    ]

    #caricamento modello torord esterno compilato
    #cmd += [
    #    '-num_external_imp', 3,
    #    '-external_imp[0]',  os.path.abspath(TORORD_LIB_ENDO),
    #    '-external_imp[1]',  os.path.abspath(TORORD_LIB_MID),
    #    '-external_imp[2]',  os.path.abspath(TORORD_LIB_EPI),
    #]

    # Proprietà di Conduzione (GRegions)
    cmd += [
        '-num_gregions', 2,
        '-gregion[0].name', 'Miocardio',
        '-gregion[0].num_IDs', 2,
        '-gregion[0].ID[0]', 1,
        '-gregion[0].ID[1]', 2,
        '-gregion[0].g_il', 0.2521, #unità misura S/m calibrati con tunecv per CV 0.64 m/s
        '-gregion[0].g_it', 0.1445, # calibrata  per CV 0.44 m/s
        '-gregion[0].g_in', 0.1591, # calibrata  per CV 0.47 m/s

        '-gregion[1].name', 'FastEndo',
        '-gregion[1].num_IDs', 1,
        '-gregion[1].ID[0]', 3,
        '-gregion[1].g_il', 0.2521 * 5.0,
        '-gregion[1].g_it', 0.1445 * 5.0, 
        '-gregion[1].g_in', 0.1591 * 5.0,
        #'-gregion[1].g_mult', 5.0,
    ]

    torord_params_common = (
        'nao=140.0,'
        'cao=1.8,'
        'ko=5.4,'
        'GNa=75.0,'
        'GNaL=0.0279,'
        'GKr=0.046,'
        'GKs=0.006,'
        'GK1=0.1908,'
        'Gncx=0.0008,'
        'GKb=0.003,'
        'GpCa=0.0005,'
        'PCa=0.0001007,'
        'Pnak=30.0'
    )

    # Eterogeneità Cellulare (ImpRegions per il modello tenTusscherPanfilov)
    cmd += [
        '-num_imp_regions', 3,
        
        '-imp_region[0].name', 'Endocardio',
        '-imp_region[0].num_IDs', 1,
        '-imp_region[0].ID[0]', 3,      #tag endocardio 3 
        '-imp_region[0].im', 'Tomek',
        '-imp_region[0].im_param', 'flags=ENDO',
        
        '-imp_region[1].name', 'Mid_Miocardio',
        '-imp_region[1].num_IDs', 1,
        '-imp_region[1].ID[0]', 2,      #tag mid miocardio 2
        '-imp_region[1].im', 'Tomek',
        '-imp_region[1].im_param', 'flags=MCELL',

        '-imp_region[2].name', 'Epicardio',
        '-imp_region[2].num_IDs', 1,
        '-imp_region[2].ID[0]', 1,      #tag epicardio 1
        '-imp_region[2].im', 'Tomek',
        '-imp_region[2].im_param', 'flags=EPI',
    ]

    # Phys Regions (Dominio Intracellulare Globale)
    cmd += [
        '-num_phys_regions', 1,
        '-phys_region[0].ptype', 0,
        '-phys_region[0].name', 'Intracellular domain',
        '-phys_region[0].num_IDs', 3,
        '-phys_region[0].ID[0]', 1,
        '-phys_region[0].ID[1]', 2,
        '-phys_region[0].ID[2]', 3,
    ]

    # Aggiunta configurazione Stimoli Dinamici all'esecuzione
    cmd += stim_cmds

    # 4. Esecuzione simulazione openCARP
    job.carp(cmd)

if __name__ == '__main__':
    run()