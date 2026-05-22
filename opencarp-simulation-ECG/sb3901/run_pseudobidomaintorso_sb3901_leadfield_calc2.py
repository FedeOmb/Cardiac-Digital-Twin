import os
import sys
sys.path.insert(0, "../")
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from carputils import settings
from carputils import tools
#from leadfield_carputils import leadfield_setup
from carputils.leadfield import leadfield_setup

TOMEK_LIB = '../tomek-model/Tomek_editv3.so'


def parser():
    parser = tools.standard_parser()
    parser.add_argument('--meshname',
                        default='./sb3901_torso/sb3901_finalmesh_torsobiv_opencarpv2',
                        help='Percorso base della mesh per openCARP')
    parser.add_argument('--vtx-dir',
                        default='./sb3901_torso/sb3901_electrodes_vtx',
                        help='Directory contenente i file .vtx degli elettrodi')
    parser.add_argument('--tend',
                        type=float, default=2.0,
                        help='Durata della simulazione in ms')
    parser.add_argument('--outdir',
                        default='test_pseudobidomain_sb3901_leadfield_calc',
                        help='Directory di output(simID)')
    parser.add_argument('--leadfield-outdir',
                        default='./sb3901_torso/leadfields',
                        help='Directory di output dei lead fields')
        
    return parser

def jobID(args):
    return args.outdir

@tools.carpexample(parser, jobID)
def run(args, job):

    # 1. Configura il leadfield passivo (Assicurati che electrodes_dir contenga RA.vtx, LA.vtx, ecc.)
    stim_cmds, elec_cmds, run_opts = leadfield_setup(
        electrodes_dir=args.vtx_dir,  # I tuoi elettrodi mappati sul torso
        lf_config=12,                         # 12 derivazioni standard
    )
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
        '-bidomain', 1, # pseudobidomain
        '-gridout_i', 3,         
        '-gridout_e', 3,    
        '-parab_solve', 1,
        '-mass_lumping', 1,
        '-floating_ground', 1,
        '-output_level', 5,
       #aumento tolleranza solver 
        '-cg_tol_ellip',   1e-6,        
        '-cg_maxit_ellip', 2000,                
    ]

    # Proprietà di Conduzione (GRegions)
    cmd += [
        '-num_gregions', 3,
        '-gregion[0].name', 'Miocardio',
        '-gregion[0].num_IDs', 3,
        '-gregion[0].ID[0]', 1,
        '-gregion[0].ID[1]', 2,
        '-gregion[0].ID[2]', 3,    
        '-gregion[0].g_il', 0.15, #conduttività standard dal paper Ana Minchole
        '-gregion[0].g_it', 0.045, 
        '-gregion[0].g_in', 0.0225, 
        '-gregion[0].g_el', 0.546,
        '-gregion[0].g_et', 0.203,
        '-gregion[0].g_en', 0.203,

        '-gregion[1].name', 'FastEndo',
        '-gregion[1].num_IDs', 1,
        '-gregion[1].ID[0]', 4,
        '-gregion[1].g_il', 0.15 * 3.0,
        '-gregion[1].g_it', 0.045 * 3.0, 
        '-gregion[1].g_in', 0.0225 * 3.0,
        '-gregion[1].g_el', 0.546 * 3.0,
        '-gregion[1].g_et', 0.203 * 3.0,
        '-gregion[1].g_en', 0.203 * 3.0,

        '-gregion[2].name', 'Torso',
        '-gregion[2].num_IDs', 1,
        '-gregion[2].ID[0]', 10,
        '-gregion[2].g_bath', 0.216, # S/m
    ]

    cmd += [
            '-num_external_imp', 1,
            '-external_imp[0]',  os.path.abspath(TOMEK_LIB),
    ]

    cmd += [
        '-num_imp_regions', 3,
        
        '-imp_region[0].name', 'Endocardio',
        '-imp_region[0].num_IDs', 2,
        '-imp_region[0].ID[0]', 3,      
        '-imp_region[0].ID[1]', 4,      # Fast Endo
        '-imp_region[0].im', 'Tomek_editv3',
        '-imp_region[0].im_param', 'flags=ENDO',
        
        '-imp_region[1].name', 'Mid_Miocardio',
        '-imp_region[1].num_IDs', 1,
        '-imp_region[1].ID[0]', 2,      
        '-imp_region[1].im', 'Tomek_editv3',
        '-imp_region[1].im_param', 'flags=MCELL',

        '-imp_region[2].name', 'Epicardio',
        '-imp_region[2].num_IDs', 1,
        '-imp_region[2].ID[0]', 1,      
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

    # 3. Appendi gli stimoli reciproci e le configurazioni degli elettrodi di carputils
    cmd += stim_cmds + elec_cmds + run_opts

    # 4. Esecuzione simulazione openCARP
    job.carp(cmd)

if __name__ == '__main__':
    run()