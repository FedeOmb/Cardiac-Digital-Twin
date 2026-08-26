#
# This file is part of openCARP
# (see https://www.openCARP.org).
#
# The openCARP project licenses this file to you under the
# Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
"""
Build stimulation and electrode parameter lists for openCARP lead-field ECG
computations.

Supported configurations (``lf_config`` argument of :func:`leadfield_setup`):

* ``1``  — bipolar (Positive, Negative)
* ``3``  — Einthoven I, II, III (RA, LA, LL)
* ``5``  — limb leads + aVR/aVL/aVF + V1
* ``12`` — full clinical 12-lead (I, II, III, aVR, aVL, aVF, V1-V6)

To add a new configuration: add its electrode list to ``_ELECTRODES``,
implement ``build_<n>lead_stims`` and ``build_<n>lead_electrodes``,
then register both in ``_BUILDERS``.

Each configuration emits a sequence of stimuli with ``crct.type = 12``
(``LF_I``); the simulator runs one reciprocity solve per stim and the
post-processor combines them by ``name`` to yield the clinical leads.
"""

import os, glob, math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Defaults shared by every stim entry; individual stims only need to
# declare index, name, and strength.
_STIM_DEFAULTS = {
    "crct_type": 12,  # LF_I (openCARP.prm circuit type 12)
}

# Expected electrode names for each lf_config value.
_ELECTRODES = {
    1:  ["Positive", "Negative"],
    3:  ["RA", "LA", "LL"],
    5:  ["RA", "LA", "LL", "V1"],
    12: ["RA", "LA", "LL", "V1", "V2", "V3", "V4", "V5", "V6"],
}


def create_electrode_param_block(stim_indices, vtx_file):
    """
    Create electrode parameter block for given stimulation indices.

    This function generates a list of command-line parameters to assign a vertex file (vtx_file)
    for each stimulation electrode based on its index. For every index in the list, it produces
    a parameter entry in the format:
        -stim[<index>].elec.vtx_file  <vtx_file>

    Parameters:
        stim_indices (list of int): A list of stimulation indices for which the electrode parameter block should be created.
        vtx_file (str): The vertex file to be assigned for each stimulation electrode.

    Returns:
        list: A list of parameters (as strings) for electrode vertex file assignments.
    """
    params = []
    for stim in stim_indices:
        params.extend([
            f'-stim[{stim}].elec.vtx_file', vtx_file,
        ])
    return params


def create_stim_params(stim_configs):
    """
    Create stimulation parameters from configuration dictionaries.

    Each dict in ``stim_configs`` must contain ``index``, ``name``, and
    ``strength``.  ``crct_type`` is optional and falls back to ``_STIM_DEFAULTS``
    when absent.

    Parameters:
        stim_configs (list of dict): Stimulation configuration entries.

    Returns:
        list: Flat list of command-line parameter strings.
    """
    params = []
    for cfg in stim_configs:
        c = {**_STIM_DEFAULTS, **cfg}
        i = c["index"]
        params.extend([
            f"-stim[{i}].name",               c["name"],
            f"-stim[{i}].crct.type",      c["crct_type"],
            f"-stim[{i}].pulse.strength", c["strength"]
        ])
    return params


# ==============================================================================
# BIPOLAR (Single Lead)
# ==============================================================================
def build_bipolar_stims(stimulus_strength):
    stims = [
        {"index": 0, "name": "I", "strength":  stimulus_strength},  # positive
        {"index": 1, "name": "I", "strength": -stimulus_strength},  # negative
    ]
    return ["-num_stim", len(stims)] + create_stim_params(stims)


def build_bipolar_electrodes(lead_positions):
    return (create_electrode_param_block([0], lead_positions["Positive"]) +
            create_electrode_param_block([1], lead_positions["Negative"]))


# ==============================================================================
# 3-LEAD CONFIG (Solves I, II, III)
# ==============================================================================
def build_3lead_stims(stimulus_strength):
    stims = []
    idx = 0

    # Lead I (LA - RA)
    stims.extend([
        {"index": idx,   "name": "I", "strength":  stimulus_strength},
        {"index": idx+1, "name": "I", "strength": -stimulus_strength},
    ]); idx += 2

    # Lead II (LL - RA)
    stims.extend([
        {"index": idx,   "name": "II", "strength":  stimulus_strength},
        {"index": idx+1, "name": "II", "strength": -stimulus_strength},
    ]); idx += 2

    # Lead III (LL - LA)
    stims.extend([
        {"index": idx,   "name": "III", "strength":  stimulus_strength},
        {"index": idx+1, "name": "III", "strength": -stimulus_strength},
    ]); idx += 2

    return ["-num_stim", len(stims)] + create_stim_params(stims)


def build_3lead_electrodes(lead_positions):
    indices = {"RA": [], "LA": [], "LL": []}
    idx = 0

    def assign(elec_name):
        nonlocal idx
        indices[elec_name].append(idx)
        idx += 1

    assign("LA"); assign("RA")  # Lead I
    assign("LL"); assign("RA")  # Lead II
    assign("LL"); assign("LA")  # Lead III

    return (create_electrode_param_block(indices["RA"], lead_positions["RA"]) +
            create_electrode_param_block(indices["LA"], lead_positions["LA"]) +
            create_electrode_param_block(indices["LL"], lead_positions["LL"]))


# ==============================================================================
# 5-LEAD CONFIG (Solves I, II, III, aVR, aVL, aVF, V1)
# ==============================================================================
def build_5lead_stims(stimulus_strength):
    stims = []
    idx = 0

    wct_sink = -stimulus_strength / 3.0
    aug_sink = -stimulus_strength / 2.0

    # Limb leads (I, II, III)
    stims.extend([
        {"index": idx,   "name": "I", "strength":  stimulus_strength},
        {"index": idx+1, "name": "I", "strength": -stimulus_strength},
    ]); idx += 2

    stims.extend([
        {"index": idx,   "name": "II", "strength":  stimulus_strength},
        {"index": idx+1, "name": "II", "strength": -stimulus_strength},
    ]); idx += 2

    stims.extend([
        {"index": idx,   "name": "III", "strength":  stimulus_strength},
        {"index": idx+1, "name": "III", "strength": -stimulus_strength},
    ]); idx += 2

    # Augmented leads (aVR, aVL, aVF)
    stims.extend([
        {"index": idx,   "name": "aVR", "strength": stimulus_strength},  # RA
        {"index": idx+1, "name": "aVR", "strength": aug_sink},           # LA
        {"index": idx+2, "name": "aVR", "strength": aug_sink},           # LL
    ]); idx += 3

    stims.extend([
        {"index": idx,   "name": "aVL", "strength": stimulus_strength},  # LA
        {"index": idx+1, "name": "aVL", "strength": aug_sink},           # RA
        {"index": idx+2, "name": "aVL", "strength": aug_sink},           # LL
    ]); idx += 3

    stims.extend([
        {"index": idx,   "name": "aVF", "strength": stimulus_strength},  # LL
        {"index": idx+1, "name": "aVF", "strength": aug_sink},           # RA
        {"index": idx+2, "name": "aVF", "strength": aug_sink},           # LA
    ]); idx += 3

    # Precordial lead V1 (Wilson Central Terminal)
    stims.extend([
        {"index": idx,   "name": "V1", "strength": stimulus_strength},   # V1
        {"index": idx+1, "name": "V1", "strength": wct_sink},            # RA
        {"index": idx+2, "name": "V1", "strength": wct_sink},            # LA
        {"index": idx+3, "name": "V1", "strength": wct_sink},            # LL
    ]); idx += 4

    return ["-num_stim", len(stims)] + create_stim_params(stims)


def build_5lead_electrodes(lead_positions):
    indices = {"RA": [], "LA": [], "LL": [], "V1": []}
    idx = 0

    def assign(elec_name):
        nonlocal idx
        indices[elec_name].append(idx)
        idx += 1

    assign("LA"); assign("RA")  # Lead I
    assign("LL"); assign("RA")  # Lead II
    assign("LL"); assign("LA")  # Lead III

    assign("RA"); assign("LA"); assign("LL")  # aVR
    assign("LA"); assign("RA"); assign("LL")  # aVL
    assign("LL"); assign("RA"); assign("LA")  # aVF

    # V1 (V1, RA, LA, LL)
    assign("V1"); assign("RA"); assign("LA"); assign("LL")

    return (create_electrode_param_block(indices["RA"], lead_positions["RA"]) +
            create_electrode_param_block(indices["LA"], lead_positions["LA"]) +
            create_electrode_param_block(indices["LL"], lead_positions["LL"]) +
            create_electrode_param_block(indices["V1"], lead_positions["V1"]))


# ==============================================================================
# 12-LEAD CONFIG (Solves I, II, III, aVR, aVL, aVF, V1-V6)
# ==============================================================================
def build_12lead_stims(stimulus_strength):
    stims = []
    idx = 0

    wct_sink = -stimulus_strength / 3.0
    aug_sink = -stimulus_strength / 2.0

    # Limb leads (I, II, III)
    stims.extend([
        {"index": idx,   "name": "I", "strength":  stimulus_strength},
        {"index": idx+1, "name": "I", "strength": -stimulus_strength},
    ]); idx += 2

    stims.extend([
        {"index": idx,   "name": "II", "strength":  stimulus_strength},
        {"index": idx+1, "name": "II", "strength": -stimulus_strength},
    ]); idx += 2

    stims.extend([
        {"index": idx,   "name": "III", "strength":  stimulus_strength},
        {"index": idx+1, "name": "III", "strength": -stimulus_strength},
    ]); idx += 2

    # Augmented limb leads (aVR: RA - 0.5*(LA+LL))
    stims.extend([
        {"index": idx,   "name": "aVR", "strength": stimulus_strength},
        {"index": idx+1, "name": "aVR", "strength": aug_sink},
        {"index": idx+2, "name": "aVR", "strength": aug_sink},
    ]); idx += 3

    # aVL: LA - 0.5*(RA+LL)
    stims.extend([
        {"index": idx,   "name": "aVL", "strength": stimulus_strength},
        {"index": idx+1, "name": "aVL", "strength": aug_sink},
        {"index": idx+2, "name": "aVL", "strength": aug_sink},
    ]); idx += 3

    # aVF: LL - 0.5*(RA+LA)
    stims.extend([
        {"index": idx,   "name": "aVF", "strength": stimulus_strength},
        {"index": idx+1, "name": "aVF", "strength": aug_sink},
        {"index": idx+2, "name": "aVF", "strength": aug_sink},
    ]); idx += 3

    # Precordial leads V1-V6 (Wilson Central Terminal)
    for v_name in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        stims.extend([
            {"index": idx,   "name": v_name, "strength": stimulus_strength},
            {"index": idx+1, "name": v_name, "strength": wct_sink},
            {"index": idx+2, "name": v_name, "strength": wct_sink},
            {"index": idx+3, "name": v_name, "strength": wct_sink},
        ]); idx += 4

    return ["-num_stim", len(stims)] + create_stim_params(stims)


def build_12lead_electrodes(lead_positions):
    indices = {
        "RA": [], "LA": [], "LL": [],
        "V1": [], "V2": [], "V3": [], "V4": [], "V5": [], "V6": [],
    }
    idx = 0

    def assign(elec_name, count=1):
        nonlocal idx
        for _ in range(count):
            indices[elec_name].append(idx)
            idx += 1

    assign("LA"); assign("RA")          # Lead I
    assign("LL"); assign("RA")          # Lead II
    assign("LL"); assign("LA")          # Lead III
    assign("RA"); assign("LA"); assign("LL")  # aVR
    assign("LA"); assign("RA"); assign("LL")  # aVL
    assign("LL"); assign("RA"); assign("LA")  # aVF
    for v in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        assign(v); assign("RA"); assign("LA"); assign("LL")

    params = []
    for key in ["RA", "LA", "LL", "V1", "V2", "V3", "V4", "V5", "V6"]:
        if indices[key]:
            params += create_electrode_param_block(indices[key], lead_positions[key])
    return params


_BUILDERS = {
    1:  (build_bipolar_stims, build_bipolar_electrodes),
    3:  (build_3lead_stims,   build_3lead_electrodes),
    5:  (build_5lead_stims,   build_5lead_electrodes),
    12: (build_12lead_stims,  build_12lead_electrodes),
}


def plot_ecg(input_dir, output_dir, noise_threshold=0.001):
    """
    Plot ECG traces from lead-field postprocessing output.

    Parameters
    ----------
    input_dir : str
        Directory containing ``ECG_*.dat`` files (two columns: time, voltage).
    output_dir : str
        Directory where PNG images are saved (created if absent).
    noise_threshold : float
        Leads with peak-to-peak amplitude below this fraction of the strongest
        lead are zeroed out before plotting.
    """
    plt.switch_backend('Agg')
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "ECG_*.dat")))
    if not files:
        raise FileNotFoundError(f"No ECG_*.dat files found in {input_dir}")

    all_data, global_max_amp = [], 0.0
    for path in files:
        try:
            data = np.loadtxt(path)
            all_data.append(data)
            v = data[:, 1] - data[0, 1]
            global_max_amp = max(global_max_amp, v.max() - v.min())
        except (ValueError, IndexError):
            print(f"Warning: could not load {path}, skipping.")
            all_data.append(None)

    cols = 3
    rows = math.ceil(len(files) / cols)
    fig_summary, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows), sharex=True)
    axes = axes.flatten()

    for i, (path, data) in enumerate(zip(files, all_data)):
        lead_name = os.path.basename(path).replace("ECG_", "").replace(".dat", "")
        if data is None:
            continue
        fig_ind = None
        try:
            time = data[:, 0]
            voltage = data[:, 1] - data[0, 1]

            if (voltage.max() - voltage.min()) < global_max_amp * noise_threshold:
                voltage = np.zeros_like(voltage)

            ax = axes[i]
            ax.plot(time, voltage, color='blue', linewidth=1.5)
            ax.set_title(f"Lead {lead_name}", fontsize=10, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.axhline(0, color='black', linewidth=0.5)
            if i >= len(files) - cols:
                ax.set_xlabel("Time (ms)")
            if i % cols == 0:
                ax.set_ylabel("Voltage (mV)")

            fig_ind, ax_ind = plt.subplots(figsize=(6, 4))
            ax_ind.plot(time, voltage, color='blue', linewidth=2)
            ax_ind.set_title(f"ECG Trace - Lead {lead_name}", fontsize=14, fontweight='bold')
            ax_ind.set_xlabel("Time (ms)", fontsize=12)
            ax_ind.set_ylabel("Voltage (mV)", fontsize=12)
            ax_ind.grid(True, linestyle='--', alpha=0.7)
            ax_ind.axhline(0, color='black', linewidth=0.8)
            fig_ind.tight_layout()
            fig_ind.savefig(os.path.join(output_dir, f"Lead_{lead_name}.png"), dpi=150)
        except Exception as e:
            print(f"Warning: could not plot lead {lead_name}: {e}")
        finally:
            if fig_ind is not None:
                plt.close(fig_ind)

    for j in range(i + 1, len(axes)):
        fig_summary.delaxes(axes[j])

    fig_summary.tight_layout()
    fig_summary.savefig(os.path.join(output_dir, "Summary_All_Leads.png"), dpi=300)
    plt.close(fig_summary)


def leadfield_setup(lf_config, electrodes_dir, stimulus_strength=1.0,
                    post_processing_opts=64, ecg_timedt=None,
                    lf_dir=None, lf_vmfile=None):
    """
    Build stimulus, electrode, and run-option CLI parameter lists for a lead-field
    computation.

    Parameters
    ----------
    lf_config : int
        Lead configuration selector: 1 (bipolar), 3 (Einthoven I/II/III),
        5 (limb + aVR/aVL/aVF + V1), or 12 (full clinical 12-lead).
    electrodes_dir : str
        Directory containing one ``.vtx`` file per electrode
        (e.g. ``RA.vtx``, ``LA.vtx``).
    stimulus_strength : float
        Total injected current in µA at each positive electrode.  The default
        of 1 µA is the natural unit for lead-field reciprocity: ECG traces
        are later scaled by the transmembrane current distribution.
    post_processing_opts : int
        openCARP ``-post_processing_opts`` bit-flag.  Defaults to ``64``, the
        Lead Field bit (``openCARP.prm`` post-processing menu entry 64 =
        ``"Leadfield"``).  Override to OR-in additional post-processing bits
        (e.g. ``64 | 2``); the Lead Field bit should always remain set.
    ecg_timedt : float, optional
        openCARP ``-ecg_timedt`` — temporal resolution of the output ECG traces
        in ms.  May be coarser than the simulation timestep.  Omit to use the
        openCARP default (1 ms).
    lf_dir : str, optional
        Path to a directory containing pre-computed ``LF_*.igb`` sensitivity
        maps.  When provided, openCARP skips recomputing the Lead Fields.
    lf_vmfile : str, optional
        Path to an existing transmembrane-voltage file (``vm.igb``).  When
        provided, ``-experiment 4`` (post-process only) is emitted automatically
        so the cardiac solver is skipped and the existing Vm is reprojected.

    Returns
    -------
    stim_cmds : list
        Flat parameter list starting with ``['-num_stim', N, ...]``.
    elec_cmds : list
        Flat parameter list of ``-stim[i].elec.vtx_file`` assignments.
    run_opts : list
        Flat openCARP parameter list containing at minimum ``-experiment`` and
        ``-post_processing_opts``.  ``-experiment`` is ``4`` when ``lf_vmfile``
        is given (post-process only), otherwise ``0`` (full pipeline).
        ``-ecg_timedt``, ``-lf_dir``, and ``-lf_vmfile`` are appended when
        their arguments are supplied.

    Raises
    ------
    ValueError
        If ``lf_config`` is not one of 1, 3, 5, 12.
    FileNotFoundError
        If any required ``.vtx`` file is missing from ``electrodes_dir``.
    """
    if lf_config not in _BUILDERS:
        raise ValueError(
            f"Unknown lf_config={lf_config}; supported: {sorted(_BUILDERS)}."
        )
    lead_positions = {
        name: os.path.join(electrodes_dir, f"{name}.vtx")
        for name in _ELECTRODES[lf_config]
    }
    missing = [name for name, path in lead_positions.items()
               if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing electrode vtx files for {missing} under {electrodes_dir}"
        )
    build_stims, build_electrodes = _BUILDERS[lf_config]
    experiment = 4 if lf_vmfile else 0
    run_opts = ['-experiment', experiment, '-post_processing_opts', post_processing_opts]
    if ecg_timedt is not None: run_opts += ['-ecg_timedt', ecg_timedt]
    if lf_dir:                 run_opts += ['-lf_dir',     lf_dir]
    if lf_vmfile:              run_opts += ['-lf_vmfile',  lf_vmfile]
    return build_stims(stimulus_strength), build_electrodes(lead_positions), run_opts
