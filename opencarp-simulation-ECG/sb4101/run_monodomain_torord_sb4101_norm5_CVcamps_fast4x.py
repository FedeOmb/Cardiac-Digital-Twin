import os
import numpy as np
import pandas as pd
from carputils import settings
from carputils import tools

TOMEK_LIB = '../tomek-model/Tomek_editv3.so'


def parser():
    parser = tools.standard_parser()
    parser.add_argument('--meshname',
                        default='./sb3901_meshes/sb3901_500um_taggedv2_fastendo',
                        help='Percorso base della mesh per openCARP')
    parser.add_argument('--vtx-file',
                        default='./sb3901_rootnodes/sb3901_fine500um_candidate_root_nodes.vtx',
                        help='File contenente gli indici dei nodi root (candidate_root_nodes.vtx)')
    parser.add_argument('--times-file',
                        default='./sb3901_rootnodes/sb3901_candidate_root_nodes_times.csv',
                        help='File CSV contenente i tempi di attivazione (candidate_root_nodes_times.csv)')
    parser.add_argument('--tend',
                        type=float, default=600.0,
                        help='Durata della simulazione in ms')
    parser.add_argument('--inputdir',
                        default='.',
                        help='Directory di input')                        
    parser.add_argument('--outdir',
                        default='./test_monodomain_torord_sb3901_norm2_fast_latcalc',
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
    vtx_dir = os.path.join(args.outdir, 'sb3901_rootnodes_vtx')
    os.makedirs(vtx_dir, exist_ok=True)
    PTS_PATH = args.meshname+'.pts'
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
            f'-stim[{i}].pulse.strength', 50,  # 50uA/cm2 convertito da paper camps 53 pA/pF x 100 pF/cm2
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
        '-bidomain', 0,
        '-parab_solve', 1,
        '-mass_lumping', 1,
       # '-pstrat', 1,
        '-output_level', 5,
        '-phie_rec_ptf', os.path.join(args.inputdir, 'sb3901_meshes', 'sb3901_electrodesum'),
        '-phie_recovery_file', 'sb3901_phie_recovery_norm2_prep_fast',
    ]

    # cmd += [
    #  '-prepacing_beats',   250,
    #  '-prepacing_bcl',     1000.0,
    #  '-prepacing_stimdur', 4.0,    # coerente con il tuo protocollo tissutale
    #  '-prepacing_stimstr', 60.0,   # unità µA/µF, default  per single-cell
    #  '-prepacing_lats', os.path.join(args.inputdir, 'sb3901_meshes', 'sb3901_latmap_zero.dat'),
    # ]
# Aggiungi queste righe per chiedere a openCARP di calcolare i LAT
    cmd += [
        '-num_LATs', 1,
        '-lats[0].ID', 'ACT',
        '-lats[0].measurand', 0,       # Misura il voltaggio (Vm)
        '-lats[0].all', 0,             # Registra solo la prima attivazione
        '-lats[0].threshold', -10.0,   # Soglia di depolarizzazione in mV
        '-lats[0].mode', 0,           
    ]
    #caricamento modello tomek modificato con parametri
    cmd += [
        '-num_external_imp', 1,
        '-external_imp[0]',  os.path.abspath(TOMEK_LIB),
    ]

    # Proprietà di Conduzione (GRegions)
    cmd += [
        '-num_gregions', 2,
        '-gregion[0].name', 'Miocardio',
        '-gregion[0].num_IDs', 3,
        '-gregion[0].ID[0]', 1,
        '-gregion[0].ID[1]', 2,
        '-gregion[0].ID[2]', 3,
        '-gregion[0].g_il', 0.2615, #unità misura S/m calibrati con tunecv per CV 0.65 m/s
        '-gregion[0].g_it', 0.1093, # calibrata  per CV 0.36 m/s
        '-gregion[0].g_in', 0.1661, # calibrata  per CV 0.48 m/s

        '-gregion[1].name', 'FastEndo',
        '-gregion[1].num_IDs', 1,
        '-gregion[1].ID[0]', 4,
        '-gregion[1].g_il', 0.2615 * 4.0,
        '-gregion[1].g_it', 0.1093 * 4.0, 
        '-gregion[1].g_in', 0.1661 * 4.0,
        #'-gregion[1].g_mult', 5.0,
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
        '-imp_region[1].im_param', 'flags=EPI',

        '-imp_region[2].name', 'Epicardio',
        '-imp_region[2].num_IDs', 1,
        '-imp_region[2].ID[0]', 1,      #tag epicardio 1
        '-imp_region[2].im', 'Tomek_editv3',
        '-imp_region[2].im_param', 'flags=EPI',
        #'-num_adjustments', 1,   
        #'-adjustment[0].variable', 'Tomek_edit.GKs_b',
        #'-adjustment[0].file', './sb3901_meshes/gks_tomek_gradient.adj',
        #'-adjustment[0].dump', 1,
    ]
    # Phys Regions (Dominio Intracellulare Globale)
    cmd += [
        '-num_phys_regions', 1,
        '-phys_region[0].ptype', 0,
        '-phys_region[0].name', 'Intracellular domain',
        '-phys_region[0].num_IDs', 4,
        '-phys_region[0].ID[0]', 1,
        '-phys_region[0].ID[1]', 2,
        '-phys_region[0].ID[2]', 3,
        '-phys_region[0].ID[3]', 4,
    ]

    # Aggiunta configurazione Stimoli Dinamici all'esecuzione
    cmd += stim_cmds

    # 4. Esecuzione simulazione openCARP
    job.carp(cmd)

if __name__ == '__main__':
    run()