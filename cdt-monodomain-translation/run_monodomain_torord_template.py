import os
import numpy as np
import pandas as pd
from carputils import settings
from carputils import tools

TOMEK_LIB = './tomek-model/Tomek_editv3.so'

def parser():
    parser = tools.standard_parser()
    parser.add_argument('--casename', required=True)
    parser.add_argument('--meshname_path',
                        default='sb3901_500um_taggedv2_fastendo',
                        help='Percorso base della mesh per openCARP senza estensione')
    parser.add_argument('--rootnodes_path',
                        default='sb3901_fine500um_candidate_root_nodes.vtx',
                        help='filename contenente gli indici dei nodi root (candidate_root_nodes.vtx)')
    parser.add_argument('--times_path',
                        default='sb3901_candidate_root_nodes_times.csv',
                        help='filename CSV contenente i tempi di attivazione (candidate_root_nodes_times.csv)')
    parser.add_argument('--lat_path')
    parser.add_argument('--prepacing_cycle_length')
    parser.add_argument('--prepacing_beats')
    parser.add_argument('--electrodes_path', required=True)
    parser.add_argument('--conductivity_myo_gil', required=True)
    parser.add_argument('--conductivity_myo_git', required=True)    
    parser.add_argument('--conductivity_myo_gin', required=True)    
    parser.add_argument('--fast_scale', required=True)
    parser.add_argument('--ionic_adj_path')
    parser.add_argument('--tend',
                        type=float, default=600.0,
                        help='Durata della simulazione in ms')
    parser.add_argument('--inputdir',
                        default='./sb4301',
                        help='Directory di input')                        
    parser.add_argument('--outdir',
                        default='./sb4301/test_monodomain_torord',
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

    MESH_FOLDER = os.path.join(args.inputdir, f"{args.casename}+_meshes")
    ROOT_NODES_FOLDER = os.path.join(args.inputdir, f"{args.casename}+_rootnodes")
    # ==========================================
    # 1. Lettura dei nodi e dei tempi di attivazione
    # ==========================================
    print(f"Leggendo i nodi da {args.rootnodes_path} e i tempi da {args.times_path}...")
    nodes = np.loadtxt(args.rootnodes_path, dtype=int)  
    
    # Assicuriamoci che nodes sia iterabile anche in caso contenga un solo nodo
    if nodes.ndim == 0:
        nodes = [nodes.item()]

    # Legge il file CSV con pandas e prende la prima colonna (tempi)
    # Gestisce automaticamente la presenza di un header come "activation_time_ms"
    #times_df = pd.read_csv(args.times_path, skiprows=None)
    #times = times_df.iloc[:, 0].values
    times = np.loadtxt(args.times_path, dtype=float)      
    assert len(nodes) == len(times), "Il numero di nodi deve corrispondere al numero dei tempi di attivazione"

    # ==========================================
    # 2. Generazione dei file .vtx per gli stimoli
    # ==========================================
    vtx_dir = os.path.join(args.outdir, f'{args.casename}_rootnodes_vtx')
    os.makedirs(vtx_dir, exist_ok=True)
    PTS_PATH = args.meshname_path+'.pts'
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
            f'-stim[{i}].pulse.strength', 100,  # 50uA/cm2 convertito da paper camps 53 pA/pF x 100 pF/cm2
            f'-stim[{i}].crct.type', 0,
        ]

    # ==========================================
    # 3. Composizione del comando openCARP
    # ==========================================
    cmd = tools.carp_cmd()
    
    # Parametri Generali e Setup della Mesh
    cmd += [
        '-simID', args.outdir,
        '-meshname', args.meshname_path,
        '-tend', args.tend,
        '-spacedt', 1.0,
        '-timedt', 1.0,
        '-dt', 10.0,
        '-bidomain', 0,
        '-parab_solve', 1,
        '-mass_lumping', 1,
       # '-pstrat', 1,
        '-output_level', 5,
        '-phie_rec_ptf', args.electrodes_path,
        '-phie_recovery_file', f'{args.casename}_phie_recovery',
    ]

    cmd += [
      '-prepacing_beats',   args.prepacing_beats,
      '-prepacing_bcl',     args.prepacing_cycle_length,
      '-prepacing_stimdur', 4.0,    # coerente con il tuo protocollo tissutale
      '-prepacing_stimstr', 60.0,   # unità µA/µF, default  per single-cell
      '-prepacing_lats', args.lat_path,
     ]
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
        '-gregion[0].g_il', args.conductivity_myo_gil, #unità misura S/m calibrati con tunecv per CV 0.65 m/s
        '-gregion[0].g_it', args.conductivity_myo_git, # calibrata  per CV 0.36 m/s
        '-gregion[0].g_in', args.conductivity_myo_gin, # calibrata  per CV 0.48 m/s

        '-gregion[1].name', 'FastEndo',
        '-gregion[1].num_IDs', 1,
        '-gregion[1].ID[0]', 4,
        '-gregion[1].g_il', f"{float(args.conductivity_myo_gil) * float(args.fast_scale)}",
        '-gregion[1].g_it', f"{float(args.conductivity_myo_git) * float(args.fast_scale)}", 
        '-gregion[1].g_in', f"{float(args.conductivity_myo_gin) * float(args.fast_scale)}",
        #'-gregion[1].g_mult', 5.0,
    ]

    # Eterogeneità Cellulare (ImpRegions per il modello tenTusscherPanfilov)
    cmd += [
        '-num_imp_regions', 1,
        
        '-imp_region[0].name', 'Endocardio',
        '-imp_region[0].num_IDs', 2,
        '-imp_region[0].ID[0]', 1,      #tag endocardio 
        '-imp_region[0].ID[1]', 4,      #tag fast endo 4
        '-imp_region[0].im', 'Tomek_editv3',
        '-imp_region[0].im_param', 'flags=ENDO,GKs_b*5,GKr_b*0.5,tjCa=60',
    ]
    # Phys Regions (Dominio Intracellulare Globale)
    cmd += [
        '-num_phys_regions', 1,
        '-phys_region[0].ptype', 0,
        '-phys_region[0].name', 'Intracellular domain',
        '-phys_region[0].num_IDs', 2,
        '-phys_region[0].ID[0]', 1,
        '-phys_region[0].ID[1]', 4,
    ]

    # Aggiunta configurazione Stimoli Dinamici all'esecuzione
    cmd += stim_cmds

    # 4. Esecuzione simulazione openCARP
    job.carp(cmd)

if __name__ == '__main__':
    run()