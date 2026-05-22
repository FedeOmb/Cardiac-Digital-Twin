import os
from carputils import tools
from carputils.leadfield import leadfield_setup

def parser():
    parser = tools.standard_parser()
    parser.add_argument('--meshname',
                        default='./sb3901_meshes/sb3901_500um_taggedv2_fastendo',
                        help='Percorso base della mesh per openCARP')
    parser.add_argument('--vtx-dir',
                        default='./sb3901_torso/sb3901_electrodes_vtx',
                        help='Directory contenente i file .vtx degli elettrodi')
    parser.add_argument('--lf-dir', 
                        default='./test_pseudobidomain_sb3901_leadfield_calc/POSTPROC_DIR', 
                        help='Percorso del file dei lead fields generati')   
    parser.add_argument('--vm-file', 
                    default='./test_monodomain_torord_sb3901_norm3/vm.igb', 
                    help='Percorso del file vm.igb da proiettare')
    return parser

@tools.carpexample(parser)
def run(args, job):
    
    # Configura la proiezione (experiment 4)
    stim_cmds, elec_cmds, run_opts = leadfield_setup(
        electrodes_dir=args.vtx_dir,        # I tuoi elettrodi mappati sul torso
        lf_dir=args.lf_dir,                 # La cartella dove hai salvato i Lead Fields stazionari
        lf_config=12,                       # 12 derivazioni standard
        lf_vmfile=args.vm_file              # FILE VM simulazione monodominio per abilitare il postprocessing
    )
    
    cmd = tools.carp_cmd()
    cmd += [
        '-meshname', args.meshname, # Serve la mesh del torso originale
    ]
    
    # Appendi i parametri generati da carputils per la proiezione
    cmd += stim_cmds + elec_cmds + run_opts    
    job.carp(cmd)

if __name__ == '__main__':
    run()