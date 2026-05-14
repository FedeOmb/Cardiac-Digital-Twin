import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from carputils import settings
from carputils import tools

TOMEK_LIB = './tomek-model/Tomek_edit.so'


def parser():
    parser = tools.standard_parser()
    parser.add_argument('--meshname',
                        default='./sb301_torsomesh/sb301_finalmesh_torsobiv_opencarp',
                        help='Percorso base della mesh per openCARP')
    parser.add_argument('--vtx-file',
                        default='./sb301_rootnodes/sb301_root_nodes_mapping_torsobiv.vtx',
                        help='File contenente gli indici dei nodi root (candidate_root_nodes.vtx)')
    parser.add_argument('--vtx-dir',
                        default='./sb301_rootnodes/sb301_rootnodes_mapped_torsobiv',
                        help='Directory contenente i file .vtx per gli stimoli')
    parser.add_argument('--times-file',
                        default='./sb301_rootnodes/sb301_candidate_root_nodes_times.csv',
                        help='File CSV contenente i tempi di attivazione (candidate_root_nodes_times.csv)')
    parser.add_argument('--tend',
                        type=float, default=400.0,
                        help='Durata della simulazione in ms')
    parser.add_argument('--outdir',
                        default='test_pseudobidomain_torord_sb301_hf',
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
    vtx_dir = os.path.join('sb301_rootnodes', 'sb301_rootnodes_mapped_torsobiv')
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
        '-dt', 10.0,
        '-bidomain', 2, # pseudobidomain
        '-parab_solve', 1,
        '-mass_lumping', 1,
        '-floating_ground', 1,
       # '-phie_rec_ptf', './sb301_torsomesh/sb301_electrodes_opencarp',
       # '-phie_recovery_file', 'sb301_phie_recovery_test',
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
        '-gregion[0].num_IDs', 2,
        '-gregion[0].ID[0]', 1,
        '-gregion[0].ID[1]', 2,
        '-gregion[0].g_il', 0.6422, #unità misura S/m calibrati pseudobidommain con tunecv per CV 0.65 m/s e g_el 0.546 S/m
        '-gregion[0].g_it', 0.3819, # calibrata  per CV 0.36 m/s e g_et 0.203 S/m
        '-gregion[0].g_in', 3.045, # calibrata  per CV 0.48 m/s e g_en 0.203 S/m
        '-gregion[0].g_el', 0.546,
        '-gregion[0].g_et', 0.203,
        '-gregion[0].g_en', 0.203,

        '-gregion[1].name', 'FastEndo',
        '-gregion[1].num_IDs', 1,
        '-gregion[1].ID[0]', 3,
        '-gregion[1].g_il', 0.2615 * 3.0,
        '-gregion[1].g_it', 0.1093 * 3.0, 
        '-gregion[1].g_in', 0.1661 * 3.0,
        '-gregion[1].g_el', 0.546 * 3.0,
        '-gregion[1].g_et', 0.203 * 3.0,
        '-gregion[1].g_en', 0.203 * 3.0,

        '-gregion[2].name', 'Torso',
        '-gregion[2].num_IDs', 1,
        '-gregion[2].ID[0]', 10,
        '-gregion[2].g_bath', 0.216, # S/m
    ]

    torord_params_hfbase_endo = (               # Heart Failure non-ischemica
        'flags=ENDO',
        'GNaL_b*1.30',    # 130%
        'thL*1.80',         # 180%
        'Gto_b*0.40',       # 40%
        'GK1_b*0.68',      # 68%
        'PNaK_b*0.70',     # 70%
        'Gncx_b*1.65',    # 165%
        'Jup_b*0.45',     # SERCA 45%
        'Cajsr_half*0.80', # RyR sens. 80%
        'Jrel_b*1.30',    # SR leak 130%
        'CaMKo*1.50'      # CaMKII 150%
    )
    torord_params_hfbase_mid = (               # Heart Failure non-ischemica
        'flags=MCELL',
        'GNaL_b*1.30',    # 130%
        'thL*1.80',         # 180%
        'Gto_b*0.40',       # 40%
        'GK1_b*0.68',      # 68%
        'PNaK_b*0.70',     # 70%
        'Gncx_b*1.65',    # 165%
        'Jup_b*0.45',     # SERCA 45%
        'Cajsr_half*0.80', # RyR sens. 80%
        'Jrel_b*1.30',    # SR leak 130%
        'CaMKo*1.50'      # CaMKII 150%
    )
    torord_params_hfbase_epi = (               # Heart Failure non-ischemica
        'flags=EPI',
        'GNaL_b*1.30',    # 130%
        'thL*1.80',         # 180%
        'Gto_b*0.40',       # 40%
        'GK1_b*0.68',      # 68%
        'PNaK_b*0.70',     # 70%
        'Gncx_b*1.65',    # 165%
        'Jup_b*0.45',     # SERCA 45%
        'Cajsr_half*0.80', # RyR sens. 80%
        'Jrel_b*1.30',    # SR leak 130%
        'CaMKo*1.50'      # CaMKII 150%
    )

    # Eterogeneità Cellulare (ImpRegions per il modello tenTusscherPanfilov)
    cmd += [
        '-num_imp_regions', 3,
        
        '-imp_region[0].name', 'Endocardio',
        '-imp_region[0].num_IDs', 1,
        '-imp_region[0].ID[0]', 3,      #tag endocardio 3 
        '-imp_region[0].im', 'Tomek_edit',
        '-imp_region[0].im_param', ','.join(torord_params_hfbase_endo),
        
        '-imp_region[1].name', 'Mid_Miocardio',
        '-imp_region[1].num_IDs', 1,
        '-imp_region[1].ID[0]', 2,      #tag mid miocardio 2
        '-imp_region[1].im', 'Tomek_edit',
        '-imp_region[1].im_param', ','.join(torord_params_hfbase_mid),

        '-imp_region[2].name', 'Epicardio',
        '-imp_region[2].num_IDs', 1,
        '-imp_region[2].ID[0]', 1,      #tag epicardio 1
        '-imp_region[2].im', 'Tomek_edit',
        '-imp_region[2].im_param', ','.join(torord_params_hfbase_epi),
    ]

    # Phys Regions (Dominio Intracellulare Globale)
    cmd += [
        '-num_phys_regions', 2,
        '-phys_region[0].ptype', 0,
        '-phys_region[0].name', 'Intracellular domain',
        '-phys_region[0].num_IDs', 3,
        '-phys_region[0].ID[0]', 1,
        '-phys_region[0].ID[1]', 2,
        '-phys_region[0].ID[2]', 3,

        '-phys_region[1].name',    'Extracellular domain',
        '-phys_region[1].ptype',   1,
        '-phys_region[1].num_IDs', 4,
        '-phys_region[1].ID[0]', 1,
        '-phys_region[1].ID[1]', 2,
        '-phys_region[1].ID[2]', 3,        
        '-phys_region[1].ID[3]', 10,        
    ]

    # Aggiunta configurazione Stimoli Dinamici all'esecuzione
    cmd += stim_cmds

    # 4. Esecuzione simulazione openCARP
    job.carp(cmd)

if __name__ == '__main__':
    run()