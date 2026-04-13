import os
import numpy as np
import pandas as pd
from carputils import settings
from carputils import tools

def parser():
    parser = tools.standard_parser()
    parser.add_argument('--meshname',
                        default='./503kaggle500_opencarpmesh/503kaggle500_opencarp',
                        help='Percorso base della mesh per openCARP')
    parser.add_argument('--vtx-file',
                        default='candidate_root_nodes.vtx',
                        help='File contenente gli indici dei nodi root (candidate_root_nodes.vtx)')
    parser.add_argument('--times-file',
                        default='candidate_root_nodes_times.csv',
                        help='File CSV contenente i tempi di attivazione (candidate_root_nodes_times.csv)')
    parser.add_argument('--tend',
                        type=float, default=100.0,
                        help='Durata della simulazione in ms')
    parser.add_argument('--outdir',
                        default='test_monodomain_tentusscher1',
                        help='Directory di output (simID)')
    return parser

def jobID(args):
    return args.outdir

@tools.carpexecution
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
    vtx_dir = os.path.join(args.outdir, 'activation_vtx')
    os.makedirs(vtx_dir, exist_ok=True)
    
    stim_cmds = ['-num_stim', len(nodes)]
    for i, (node, t) in enumerate(zip(nodes, times)):
        vtx_filename = os.path.join(vtx_dir, f'RN{i+1}.vtx')
        with open(vtx_filename, 'w') as f:
            f.write("1\nintra\n")  # Header standard di openCARP per i vertici intracellulari
            f.write(f"{node}\n")
        
        # Parametri dello stimolo
        stim_cmds += [
            f'-stim[{i}].name', f'RN{i+1}',
            f'-stim[{i}].elec.vtx_file', vtx_filename,
            f'-stim[{i}].ptcl.duration', 2.0,
            f'-stim[{i}].ptcl.npls', 1,
            f'-stim[{i}].ptcl.start', float(t),
            f'-stim[{i}].pulse.strength', 100.0,
            f'-stim[{i}].crct.type', 0
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
        '-dt', 25,
        '-bidomain', 0,
        '-parab_solve', 1,
        '-mass_lumping', 1,
        '-phie_rec_ptf', 'electrode_location_test',
        '-phie_recovery_file', 'phie_recovery_test'
    ]

    # Proprietà di Conduzione (GRegions)
    cmd += [
        '-num_gregions', 1,
        '-gregion[0].name', 'Miocardio',
        '-gregion[0].num_IDs', 3,
        '-gregion[0].ID[0]', 1,
        '-gregion[0].ID[1]', 2,
        '-gregion[0].ID[2]', 3,
        '-gregion[0].g_il', 0.174,
        '-gregion[0].g_it', 0.019,
        '-gregion[0].g_in', 0.019,
    ]

    # Eterogeneità Cellulare (ImpRegions per il modello tenTusscherPanfilov)
    cmd += [
        '-num_imp_regions', 3,
        
        '-imp_region[0].name', 'Endocardio',
        '-imp_region[0].num_IDs', 1,
        '-imp_region[0].ID[0]', 1,
        '-imp_region[0].im', 'tenTusscherPanfilov',
        '-imp_region[0].im_param', 'flags=ENDO',
        
        '-imp_region[1].name', 'Mid_Miocardio',
        '-imp_region[1].num_IDs', 1,
        '-imp_region[1].ID[0]', 2,
        '-imp_region[1].im', 'tenTusscherPanfilov',
        '-imp_region[1].im_param', 'flags=MCELL',
        
        '-imp_region[2].name', 'Epicardio',
        '-imp_region[2].num_IDs', 1,
        '-imp_region[2].ID[0]', 3,
        '-imp_region[2].im', 'tenTusscherPanfilov',
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
    args = parser().parse_args()
    tools.run(args, run)