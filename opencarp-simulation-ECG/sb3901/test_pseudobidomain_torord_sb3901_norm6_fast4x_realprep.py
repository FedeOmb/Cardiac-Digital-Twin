import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from carputils import settings
from carputils import tools

TOMEK_LIB = '../tomek-model/Tomek_editv3.so'


def parser():
    parser = tools.standard_parser()
    parser.add_argument('--meshname',
                        default='./sb3901_torso/sb3901_finalmesh_torsobiv_opencarpv2',
                        help='Percorso base della mesh per openCARP')
    parser.add_argument('--vtx-file',
                        default='./sb3901_rootnodes/sb3901_root_nodes_mapping_torsobiv_v2.vtx',
                        help='File contenente gli indici dei nodi root (candidate_root_nodes.vtx)')
    parser.add_argument('--vtx-dir',
                        default='./sb3901_rootnodes/sb3901_rootnodes_mapped_torsobiv_v2',
                        help='Directory contenente i file .vtx per gli stimoli')
    parser.add_argument('--times-file',
                        default='./sb3901_rootnodes/sb3901_candidate_root_nodes_times.csv',
                        help='File CSV contenente i tempi di attivazione (candidate_root_nodes_times.csv)')
    parser.add_argument('--tend',
                        type=float, default=500.0,
                        help='Durata della simulazione in ms')
    parser.add_argument('--outdir',
                        default='test_pseudobidomain_torord_sb3901_norm6_fast4x_realprep',
                        help='Directory di output (simID)')
    return parser

def jobID(args):
    return args.outdir

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
    # 2. Configurazione degli stimoli dai file .vtx esistenti
    # ==========================================
    vtx_dir = args.vtx_dir
    stim_cmds = ['-num_stim', len(nodes)]
    for i, (node, t) in enumerate(zip(nodes, times)):
        vtx_filename = os.path.join(vtx_dir, f'RN{i+1}.vtx')

        # Parametri dello stimolo
        stim_cmds += [
            f'-stim[{i}].name', f'RN{i+1}',
            f'-stim[{i}].elec.vtx_file', vtx_filename,
            f'-stim[{i}].ptcl.duration', 4.0, # 4ms da paper camps
            f'-stim[{i}].ptcl.npls', 1,
            f'-stim[{i}].ptcl.start', float(t),
            f'-stim[{i}].pulse.strength', 50,  # 53uA/cm2 convertito da paper camps 53 pA/pF x 100 pF/cm2
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
        '-dt', 10,
        '-bidomain', 2, # pseudobidomain
        '-gridout_i', 3,
        '-gridout_e', 3,
        '-parab_solve', 1,
        '-mass_lumping', 1,
        '-floating_ground', 1,
        '-output_level', 5,
       #aumento tolleranza solver 
        '-cg_tol_ellip',   1e-6,        
        '-cg_maxit_ellip', 2000,                
        '-phie_rec_ptf', './sb3901_meshes/sb3901_electrodesum',
        '-phie_recovery_file', 'sb3901_phie_recovery_from_pseudobid',
    ]

    #caricamento modello tomek modificato con parametri
    cmd += [
        '-num_external_imp', 1,
        '-external_imp[0]',  os.path.abspath(TOMEK_LIB),
    ]

    # Proprietà di Conduzione (GRegions)
    cmd += [
        '-num_gregions', 3,
        '-gregion[0].name', 'Miocardio',
        '-gregion[0].num_IDs', 3,
        '-gregion[0].ID[0]', 1,
        '-gregion[0].ID[1]', 2,
        '-gregion[0].ID[2]', 3,    
        '-gregion[0].g_il', 0.2613, #unità misura S/m calibrati pseudobidommain con tunecv per CV 0.50 m/s
        '-gregion[0].g_it', 0.0794, # calibrata  per CV 0.21 m/s
        '-gregion[0].g_in', 0.0392, # calibrata  per CV 0.13 m/s
        '-gregion[0].g_el', 0.9511,
        '-gregion[0].g_et', 0.3583,
        '-gregion[0].g_en', 0.3537,

        '-gregion[1].name', 'FastEndo',
        '-gregion[1].num_IDs', 1,
        '-gregion[1].ID[0]', 4,
        '-gregion[1].g_il', 0.2613 * 3.0,
        '-gregion[1].g_it', 0.0794 * 3.0, 
        '-gregion[1].g_in', 0.0392 * 3.0,
        '-gregion[1].g_el', 0.9511 * 3.0,
        '-gregion[1].g_et', 0.3583 * 3.0,
        '-gregion[1].g_en', 0.3537 * 3.0,

        '-gregion[2].name', 'Torso',
        '-gregion[2].num_IDs', 1,
        '-gregion[2].ID[0]', 10,
        '-gregion[2].g_bath', 0.216, # S/m
    ]


    # Eterogeneità Cellulare (ImpRegions per il modello tenTusscherPanfilov)
    cmd += [
        '-num_imp_regions', 3,
        
        '-imp_region[0].name', 'Endocardio',
        '-imp_region[0].num_IDs', 2,
        '-imp_region[0].ID[0]', 3,      #tag endocardio 3 
        '-imp_region[0].ID[1]', 4,      #tag fast endo 4
        '-imp_region[0].im', 'Tomek_editv3',
        '-imp_region[0].im_param', 'flags=ENDO',
        
        '-imp_region[1].name', 'Mid_Miocardio',
        '-imp_region[1].num_IDs', 1,
        '-imp_region[1].ID[0]', 2,      #tag mid miocardio 2
        '-imp_region[1].im', 'Tomek_editv3',
        '-imp_region[1].im_param', 'flags=MCELL',

        '-imp_region[2].name', 'Epicardio',
        '-imp_region[2].num_IDs', 1,
        '-imp_region[2].ID[0]', 1,      #tag epicardio 1
        '-imp_region[2].im', 'Tomek_editv3',
        '-imp_region[2].im_param', 'flags=EPI',
    ]

    # Phys Regions (Dominio Intracellulare Globale)
    cmd += [
        '-num_phys_regions', 2,

        '-phys_region[0].name', 'Extracellular domain',
        '-phys_region[0].ptype',   1,
        '-phys_region[0].num_IDs', 5,
        '-phys_region[0].ID[0]', 1,
        '-phys_region[0].ID[1]', 2,
        '-phys_region[0].ID[2]', 3,        
        '-phys_region[0].ID[3]', 4,         
        '-phys_region[0].ID[4]', 10,   

        '-phys_region[1].name', 'Intracellular domain',
        '-phys_region[1].ptype', 0,        
        '-phys_region[1].num_IDs', 4,
        '-phys_region[1].ID[0]', 1,
        '-phys_region[1].ID[1]', 2,
        '-phys_region[1].ID[2]', 3,
        '-phys_region[1].ID[3]', 4,     
    ]

    # Aggiunta configurazione Stimoli Dinamici all'esecuzione
    cmd += stim_cmds

    # 4. Esecuzione simulazione openCARP
    job.carp(cmd)

if __name__ == '__main__':
    run()