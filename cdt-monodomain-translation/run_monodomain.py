#!/usr/bin/env python3
"""
run_monodomain.py  —  openCARP Monodomain Simulation (Tomek / CDT pipeline)

Parametri passati a riga di comando (tutti con default override):
  --case          Identificativo paziente (es. sb4301)
  --meshname      Path base mesh (senza estensione)
  --vtx-file      Root nodes .vtx dall'inferenza CDT
  --times-file    Tempi attivazione .csv dall'inferenza CDT
  --electrodes    Path base file elettrodi (senza estensione)
  --init-lats     File LAT iniziali per pre-pacing (.dat)
  --gks-adj       File gradiente GKs (.adj)
  --tomek-lib     Libreria Tomek (.so)
  --outdir        Directory di output (simID openCARP)
  --tend          Durata simulazione [ms]  (default: 500)
  --dt            Passo temporale [us]     (default: 10)
  --stim-radius   Raggio stimolo [um]      (default: 2000)
  --stim-duration Durata stimolo [ms]      (default: 4.0)
  --stim-strength Intensità stimolo [pA/pF](default: 53.0)
  --prepacing-beats  Beat pre-pacing       (default: 800)
  --prepacing-bcl    BCL pre-pacing [ms]   (default: 810.0)
"""

import os
import numpy as np
import pandas as pd
from carputils import settings, tools


# ─── Parser ──────────────────────────────────────────────────────────────────

def parser():
    p = tools.standard_parser()

    # Identificativo
    p.add_argument('--case',          required=True,
                   help='Identificativo paziente (es. sb4301)')

    # File di input
    p.add_argument('--meshname',      required=True,
                   help='Path base mesh (senza estensione)')
    p.add_argument('--vtx-file',      required=True,
                   help='Root nodes .vtx (output inferenza CDT)')
    p.add_argument('--times-file',    required=True,
                   help='Tempi attivazione .csv (output inferenza CDT)')
    p.add_argument('--electrodes',    required=True,
                   help='Path base elettrodi (senza estensione)')
    p.add_argument('--init-lats',     required=True,
                   help='LAT iniziali pre-pacing (.dat)')
    p.add_argument('--gks-adj',       required=True,
                   help='Gradiente GKs (.adj)')
    p.add_argument('--tomek-lib',     required=True,
                   help='Libreria Tomek (.so)')

    # Output
    p.add_argument('--outdir',        required=True,
                   help='Directory di output (simID)')

    # Parametri simulazione
    p.add_argument('--tend',          type=float, default=500.0,
                   help='Durata simulazione [ms]')
    p.add_argument('--dt',            type=float, default=10.0,
                   help='Passo temporale [us]')
    p.add_argument('--spacedt',       type=float, default=1.0,
                   help='Passo output spaziale [ms]')
    p.add_argument('--timedt',        type=float, default=1.0,
                   help='Passo output temporale [ms]')
    p.add_argument('--stim-radius',   type=int,   default=2000,
                   help='Raggio ricerca nodi stimolo [um]')
    p.add_argument('--stim-duration', type=float, default=4.0,
                   help='Durata stimolo [ms]')
    p.add_argument('--stim-strength', type=float, default=53.0,
                   help='Intensita stimolo [pA/pF]')
    p.add_argument('--prepacing-beats', type=int,   default=800)
    p.add_argument('--prepacing-bcl',   type=float, default=810.0,
                   help='BCL pre-pacing [ms]')
    return p


def jobID(args):
    return args.outdir


# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_nodes_by_radius(pts_file, root_node, radius):
    pts = np.loadtxt(pts_file, skiprows=1)
    dists = np.linalg.norm(pts - pts[root_node], axis=1)
    return np.where(dists <= radius)[0]


# ─── Main run ────────────────────────────────────────────────────────────────

@tools.carpexample(parser, jobID)
def run(args, job):

    # 1. Lettura root nodes e tempi di attivazione ----------------------------
    print(f"[1/4] Lettura root nodes: {args.vtx_file}")
    nodes_raw = np.loadtxt(args.vtx_file, dtype=int)
    nodes = [nodes_raw.item()] if nodes_raw.ndim == 0 else list(nodes_raw)

    print(f"[1/4] Lettura tempi: {args.times_file}")
    times = pd.read_csv(args.times_file).iloc[:, 0].values

    assert len(nodes) == len(times), (
        f"Mismatch nodi/tempi: {len(nodes)} vs {len(times)}")
    print(f"[1/4] Root nodes: {len(nodes)} | "
          f"Tempi: {times.min():.1f}-{times.max():.1f} ms")

    # 2. Generazione file .vtx per ogni stimolo --------------------------------
    print(f"[2/4] Generazione vtx stimoli (raggio={args.stim_radius} um)...")
    vtx_dir = os.path.join(args.outdir, f'{args.case}_rootnodes_vtx')
    os.makedirs(vtx_dir, exist_ok=True)
    pts_file = args.meshname + '.pts'

    stim_cmds = ['-num_stim', len(nodes)]
    for i, (node, t) in enumerate(zip(nodes, times)):
        nearby = get_nodes_by_radius(pts_file, node, args.stim_radius)
        vtx_path = os.path.join(vtx_dir, f'RN{i+1}.vtx')
        with open(vtx_path, 'w') as fh:
            fh.write(f'{len(nearby)}\nintra\n')
            fh.writelines(f'{n}\n' for n in nearby)
        stim_cmds += [
            f'-stim[{i}].name',          f'RN{i+1}',
            f'-stim[{i}].elec.vtx_file',  vtx_path,
            f'-stim[{i}].ptcl.duration',  args.stim_duration,
            f'-stim[{i}].ptcl.npls',       1,
            f'-stim[{i}].ptcl.start',      float(t),
            f'-stim[{i}].pulse.strength',  args.stim_strength,
            f'-stim[{i}].crct.type',        0,
        ]

    # 3. Composizione comando openCARP ----------------------------------------
    print("[3/4] Composizione comando openCARP...")
    cmd = tools.carp_cmd()

    # Setup generale
    cmd += [
        '-simID',           args.outdir,
        '-meshname',        args.meshname,
        '-tend',            args.tend,
        '-spacedt',         args.spacedt,
        '-timedt',          args.timedt,
        '-dt',              args.dt,
        '-bidomain',        0,
        '-parab_solve',     1,
        '-mass_lumping',    1,
        '-output_level',    5,
        '-phie_rec_ptf',    args.electrodes,
        '-phie_recovery_file', f'{args.case}_phie_recovery',
    ]

    # Pre-pacing
    cmd += [
        '-prepacing_beats',   args.prepacing_beats,
        '-prepacing_bcl',     args.prepacing_bcl,
        '-prepacing_stimdur', args.stim_duration,
        '-prepacing_stimstr', 60.0,
        '-prepacing_lats',    args.init_lats,
    ]

    # LAT computation
    cmd += [
        '-num_LATs',          1,
        '-lats[0].ID',        'ACT',
        '-lats[0].measurand', 0,
        '-lats[0].all',       0,
        '-lats[0].threshold', -10.0,
        '-lats[0].mode',      0,
    ]

    # Modello ionico Tomek
    cmd += [
        '-num_external_imp',  1,
        '-external_imp[0]',   os.path.abspath(args.tomek_lib),
    ]

    # Regioni di conduzione (GRegions)
    cmd += [
        '-num_gregions',       2,
        '-gregion[0].name',    'Miocardio',
        '-gregion[0].num_IDs', 3,
        '-gregion[0].ID[0]',   1,
        '-gregion[0].ID[1]',   2,
        '-gregion[0].ID[2]',   3,
        '-gregion[0].g_il',    0.3831,
        '-gregion[0].g_it',    0.1692,
        '-gregion[0].g_in',    0.2524,
        '-gregion[1].name',    'FastEndo',
        '-gregion[1].num_IDs', 1,
        '-gregion[1].ID[0]',   4,
        '-gregion[1].g_il',    0.3831 * 4.0,
        '-gregion[1].g_it',    0.1692 * 4.0,
        '-gregion[1].g_in',    0.2524 * 4.0,
    ]

    # Eterogeneita cellulare (ImpRegions — Tomek ENDO/MCELL/EPI)
    cmd += [
        '-num_imp_regions',         3,
        '-imp_region[0].name',      'Endocardio',
        '-imp_region[0].num_IDs',   2,
        '-imp_region[0].ID[0]',     3,
        '-imp_region[0].ID[1]',     4,
        '-imp_region[0].im',        'Tomek_editv3',
        '-imp_region[0].im_param',  'flags=ENDO',
        '-imp_region[1].name',      'Mid_Miocardio',
        '-imp_region[1].num_IDs',   1,
        '-imp_region[1].ID[0]',     2,
        '-imp_region[1].im',        'Tomek_editv3',
        '-imp_region[1].im_param',  'flags=MCELL',
        '-imp_region[2].name',      'Epicardio',
        '-imp_region[2].num_IDs',   1,
        '-imp_region[2].ID[0]',     1,
        '-imp_region[2].im',        'Tomek_editv3',
        '-imp_region[2].im_param',  'flags=EPI',
        '-num_adjustments',         1,
        '-adjustment[0].variable',  'Tomek_editv3.GKs_b',
        '-adjustment[0].file',      args.gks_adj,
        '-adjustment[0].dump',      1,
    ]

    # Dominio intracellulare globale (PhysRegions)
    cmd += [
        '-num_phys_regions',       1,
        '-phys_region[0].ptype',   0,
        '-phys_region[0].name',    'Intracellular domain',
        '-phys_region[0].num_IDs', 4,
        '-phys_region[0].ID[0]',   1,
        '-phys_region[0].ID[1]',   2,
        '-phys_region[0].ID[2]',   3,
        '-phys_region[0].ID[3]',   4,
    ]

    # Stimoli dinamici
    cmd += stim_cmds

    # 4. Avvio simulazione
    print("[4/4] Avvio openCARP...")
    job.carp(cmd)


if __name__ == '__main__':
    run()
