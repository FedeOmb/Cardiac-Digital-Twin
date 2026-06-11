# This is a stand alone script that plots ECGs from body surface potentials
# simulated from monodomain simulations.
# The normalisation strategy is the same as in ecg_functions.py.
import numpy as np
from scipy import signal
import matplotlib
import os
import csv
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess
import wfdb
#plt.switch_backend('Agg')# backend non-interattivo per ambiente headless
# Helper functions from ecg_functions.py:

LEAD_NAMES_8 = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
LEAD_NAMES_12 = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def filter_butterworth_ecg(b, a, ecg):
    return signal.filtfilt(b, a, ecg)

def __filter_ecg(original_ecg, low_a_filtfilt, low_b_filtfilt, high_a_filtfilt, high_b_filtfilt):
    # First we filter out the low frequencies using a high-pass filter and the lower thresholds
    processed_ecg = filter_butterworth_ecg(b=low_b_filtfilt, a=low_a_filtfilt, ecg=original_ecg)
    # Secondly we filter out the high frequencies using a low-pass filter and the higher thresholds
    return filter_butterworth_ecg(b=high_b_filtfilt, a=high_a_filtfilt, ecg=processed_ecg)

def zero_align_ecg(original_ecg):
    return original_ecg - (original_ecg[:, 0:1] + original_ecg[:, -2:-1]) / 2  # align at zero


def __preprocess_ecg_without_normalise(original_ecg, filtering, zero_align, frequency, high_freq_cut, low_freq_cut):
    processed_ecg = original_ecg
    if filtering:
        high_w = high_freq_cut / (frequency / 2)  # Normalize the frequency
        high_b_filtfilt, high_a_filtfilt = signal.butter(4, high_w,'low')  # Butterworth filter of fourth order.
        low_w = low_freq_cut / (frequency / 2)  # Normalize the frequency
        low_b_filtfilt, low_a_filtfilt = signal.butter(4, low_w, 'high')
        processed_ecg = __filter_ecg(processed_ecg, low_a_filtfilt=low_a_filtfilt, low_b_filtfilt=low_b_filtfilt,
                                     high_a_filtfilt=high_a_filtfilt, high_b_filtfilt=high_b_filtfilt)
    if zero_align:
        processed_ecg = zero_align_ecg(processed_ecg)
    return processed_ecg

def preprocess_ecg_without_normalise(original_ecg, filtering, zero_align, frequency, high_freq_cut, low_freq_cut):
    processed_ecg = original_ecg.copy()
    if filtering:
        high_w = high_freq_cut / (frequency / 2.0)
        low_w = low_freq_cut / (frequency / 2.0)
        high_b_filtfilt, high_a_filtfilt = signal.butter(4, high_w, 'low')
        low_b_filtfilt, low_a_filtfilt = signal.butter(4, low_w, 'high')
        processed_ecg = __filter_ecg(
            processed_ecg,
            low_a_filtfilt=low_a_filtfilt,
            low_b_filtfilt=low_b_filtfilt,
            high_a_filtfilt=high_a_filtfilt,
            high_b_filtfilt=high_b_filtfilt,
        )
    if zero_align:
        processed_ecg = zero_align_ecg(processed_ecg)
    return processed_ecg

def __set_reference_lead_is_positive(max_qrs_end, reference_ecg):
    approx_qrs_end = min(reference_ecg.shape[1], max_qrs_end)  # Approximate end of QRS.
    reference_lead_max = np.absolute(np.amax(reference_ecg[:, :approx_qrs_end], axis=1))
    reference_lead_min = np.absolute(np.amin(reference_ecg[:, :approx_qrs_end], axis=1))
    reference_lead_is_positive = reference_lead_max >= reference_lead_min
    reference_amplitudes = np.zeros(shape=nb_leads, dtype=np.float64)
    reference_amplitudes[reference_lead_is_positive] = reference_lead_max[reference_lead_is_positive]
    reference_amplitudes[np.logical_not(reference_lead_is_positive)] = reference_lead_min[
        np.logical_not(reference_lead_is_positive)]
    # if verbose:
    #     print('reference_lead_is_positive')
    #     print(reference_lead_is_positive)
    #     print('reference_amplitudes')
    #     print(reference_amplitudes)
    return reference_lead_is_positive  # Have some R progression by normalising by the
    # largest absolute amplitude lead

def __normalise_ecg_based_on_rwave_8_leads(original_ecg, qrs_onset, reference_ecg, max_len_qrs, reference_lead_is_positive):
    if nb_leads != 8 or original_ecg.shape[0] != 8:
        raise(Exception, 'This function is hardcoded for the specific ECG configuration: I, II, V1, ..., V6')
    # print('Normalising ECG ', original_ecg.shape)
    approx_qrs_end = min(reference_ecg.shape[1], max_len_qrs+qrs_onset)  # Approximate end of QRS.
    # approx_qrs_width = min(original_ecg.shape[1], max_len_qrs)  # This approximation is more robust.
    # print('approx_qrs_end ', approx_qrs_end)
    # print(np.amax(original_ecg[:, :approx_qrs_end]))
    reference_amplitudes = np.empty(shape=nb_leads, dtype=np.float64)
    reference_amplitudes[reference_lead_is_positive] = np.absolute(
        np.amax(original_ecg[:, :approx_qrs_end], axis=1)[
            reference_lead_is_positive])
    reference_amplitudes[np.logical_not(reference_lead_is_positive)] = \
        np.absolute(np.amin(original_ecg[:, :approx_qrs_end], axis=1))[np.logical_not(
            reference_lead_is_positive)]
    normalised_ecg = np.zeros(original_ecg.shape)
    normalised_ecg[:2, :] = original_ecg[:2, :] / np.mean(
        reference_amplitudes[:2])  # Normalise limb leads separatedly
    normalised_ecg[2:nb_leads, :] = original_ecg[2:nb_leads, :] / np.mean(
        reference_amplitudes[2:nb_leads])
    return normalised_ecg

def preprocess_ecg(original_ecg, reference_ecg, filtering, zero_align, frequency, high_freq_cut, low_freq_cut,
                   max_len_qrs, qrs_onset, reference_lead_is_positive):
    processed_ecg = original_ecg
    if filtering:
        high_w = high_freq_cut / (frequency / 2)  # Normalize the frequency
        high_b_filtfilt, high_a_filtfilt = signal.butter(4, high_w,'low')  # Butterworth filter of fourth order.
        low_w = low_freq_cut / (frequency / 2)  # Normalize the frequency
        low_b_filtfilt, low_a_filtfilt = signal.butter(4, low_w, 'high')
        processed_ecg = __filter_ecg(processed_ecg, low_a_filtfilt=low_a_filtfilt, low_b_filtfilt=low_b_filtfilt,
                                     high_a_filtfilt=high_a_filtfilt, high_b_filtfilt=high_b_filtfilt)
    if zero_align:
        processed_ecg = zero_align_ecg(processed_ecg)
    if normalise:
        processed_ecg = __normalise_ecg_based_on_rwave_8_leads(original_ecg=processed_ecg, qrs_onset=qrs_onset,
                                                               reference_ecg=reference_ecg, max_len_qrs=max_len_qrs,
                                                               reference_lead_is_positive=reference_lead_is_positive)
    return processed_ecg

def import_simulated_ecg_8leads_raw(filename, monoalg_activation_offset):
    data = np.genfromtxt(filename, skip_footer=1)
    t = data[:, 0] - monoalg_activation_offset
    LA = data[:, 1]
    RA = data[:, 2]
    LL = data[:, 3]
    RL = data[:, 4]
    V1 = data[:, 5]
    V2 = data[:, 6]
    V3 = data[:, 7]
    V4 = data[:, 8]
    V5 = data[:, 9]
    V6 = data[:, 10]

    # Ealuate Wilson's central terminal
    VW = 1.0 / 3.0 * (RA + LA + LL)

    # Evaluate simulated ECG lead traces
    V1 = V1 - VW
    V2 = V2 - VW
    V3 = V3 - VW
    V4 = V4 - VW
    V5 = V5 - VW
    V6 = V6 - VW
    I = LA - RA
    II = LL - RA
    III = LL - LA
    aVL = LA - (RA + LL) / 2.0
    aVF = LL - (LA + RA) / 2.0
    aVR = RA - (LA + LL) / 2.0
    ecgs = np.vstack((I, II, V1, V2, V3, V4, V5, V6))
    return t, ecgs

def import_simulated_ecg_12leads_raw(filename, monoalg_activation_offset):
    data = np.genfromtxt(filename, skip_footer=1)
    t = data[:, 0] - monoalg_activation_offset
    LA = data[:, 1]
    RA = data[:, 2]
    LL = data[:, 3]
    RL = data[:, 4]
    V1 = data[:, 5]
    V2 = data[:, 6]
    V3 = data[:, 7]
    V4 = data[:, 8]
    V5 = data[:, 9]
    V6 = data[:, 10]

    # Ealuate Wilson's central terminal
    VW = 1.0 / 3.0 * (RA + LA + LL)

    # Evaluate simulated ECG lead traces
    V1 = V1 - VW
    V2 = V2 - VW
    V3 = V3 - VW
    V4 = V4 - VW
    V5 = V5 - VW
    V6 = V6 - VW
    I = LA - RA
    II = LL - RA
    III = LL - LA
    aVL = LA - (RA + LL) / 2.0
    aVF = LL - (LA + RA) / 2.0
    aVR = RA - (LA + LL) / 2.0
    ecgs = np.vstack((I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6))
    return t, ecgs

def _normalise_sig_name(name):
    return name.strip().upper().replace('AVR', 'AVR').replace('AVL', 'AVL').replace('AVF', 'AVF')

def import_ptb_ecg_8leads(record_path): # input: path al record PTB senza estensione

    record = wfdb.rdrecord(record_path)
    if record.p_signal is None:
        raise ValueError("WFDB non ha restituito p_signal. Verifica che il record PTB sia leggibile.")

    sig_names = [_normalise_sig_name(s) for s in record.sig_name]
    sig_map = {name: i for i, name in enumerate(sig_names)}

    missing = [lead for lead in LEAD_NAMES_8 if lead not in sig_map]
    if missing:
        raise ValueError(
            f"Nel record PTB mancano queste derivazioni richieste: {missing}. Disponibili: {sig_names}"
        )

    data = record.p_signal
    ecg = np.vstack([data[:, sig_map[lead]] for lead in LEAD_NAMES_8])
    fs = float(record.fs)
    t_ms = np.arange(ecg.shape[1]) * 1000.0 / fs
    return t_ms, ecg, fs, sig_names

def import_ptb_ecg_12leads(record_path): # input: path al record PTB senza estensione
    record = wfdb.rdrecord(record_path)
    if record.p_signal is None:
        raise ValueError("WFDB non ha restituito p_signal. Verifica che il record PTB sia leggibile.")

    sig_names = [_normalise_sig_name(s) for s in record.sig_name]
    sig_map = {name: i for i, name in enumerate(sig_names)}

    missing = [lead for lead in LEAD_NAMES_12 if lead not in sig_map]
    if missing:
        raise ValueError(
            f"Nel record PTB mancano queste derivazioni richieste: {missing}. Disponibili: {sig_names}"
        )

    data = record.p_signal
    ecg = np.vstack([data[:, sig_map[lead]] for lead in LEAD_NAMES_12])
    fs = float(record.fs)
    t_ms = np.arange(ecg.shape[1]) * 1000.0 / fs
    return t_ms, ecg, fs, sig_names

def resample_ecg(ecg, original_fs, target_fs):
    if abs(original_fs - target_fs) < 1e-9:
        return ecg.copy()
    n_target = int(round(ecg.shape[1] * target_fs / original_fs))
    return signal.resample(ecg, n_target, axis=1)


def find_qrs_anchor(ecg, fs, search_window_ms=250):
    n = min(ecg.shape[1], int(round(search_window_ms * fs / 1000.0)))
    segment = ecg[:, :n]
    ref = np.mean(np.abs(segment), axis=0)
    return int(np.argmax(ref))

def align_qrs_anchor(reference_ecg, reference_t_ms, reference_qrs_anchor, simulated_ecg, simulated_t_ms, simulated_qrs_anchor, fs):
    shift = int(reference_qrs_anchor - simulated_qrs_anchor)
    if shift > 0:
        reference_ecg = reference_ecg[:, shift:]
        reference_t_ms = np.arange(reference_ecg.shape[1]) * 1000.0 / fs
    elif shift < 0:
        reference_ecg = np.pad(reference_ecg, ((0, 0), (-shift, 0)), mode='constant')
        reference_t_ms = np.arange(reference_ecg.shape[1]) * 1000.0 / fs

    n = min(reference_ecg.shape[1], simulated_ecg.shape[1])
    return reference_ecg[:, :n], reference_t_ms[:n], simulated_ecg[:, :n], simulated_t_ms[:n]

def find_r_peaks_from_multilead(ecg, fs, min_distance_ms=500):
    aggregate = np.mean(np.abs(ecg), axis=0)
    distance = max(1, int(round(min_distance_ms * fs / 1000.0)))
    prominence = max(np.std(aggregate) * 0.5, 1e-6)
    peaks, _ = signal.find_peaks(aggregate, distance=distance, prominence=prominence)
    return peaks


def select_central_ptb_beat(reference_ecg, fs, target_len_ms=600.0, qrs_pre_ms=120.0):
    peaks = find_r_peaks_from_multilead(reference_ecg, fs)
    if len(peaks) == 0:
        raise ValueError('Nessun picco R trovato nel record PTB.')

    center_time_idx = reference_ecg.shape[1] // 2
    central_peak = peaks[np.argmin(np.abs(peaks - center_time_idx))]

    half_len = int(round(target_len_ms * fs / 1000.0 / 2.0))
    start = central_peak - int(round(qrs_pre_ms * fs / 1000.0))
    end = start + 2 * half_len

    if start < 0:
        end -= start
        start = 0
    if end > reference_ecg.shape[1]:
        shift = end - reference_ecg.shape[1]
        start = max(0, start - shift)
        end = reference_ecg.shape[1]

    beat = reference_ecg[:, start:end]
    t_ms = np.arange(beat.shape[1]) * 1000.0 / fs
    qrs_anchor = central_peak - start
    return beat, t_ms, qrs_anchor, central_peak, peaks

def align_by_qrs_peak(reference_ecg, simulated_ecg, fs, search_window_ms=250):
    ref_peak = find_qrs_anchor(reference_ecg, fs, search_window_ms=search_window_ms)
    sim_peak = find_qrs_anchor(simulated_ecg, fs, search_window_ms=search_window_ms)

    if ref_peak == sim_peak:
        return reference_ecg, simulated_ecg, ref_peak

    if sim_peak < ref_peak:
        pad = ref_peak - sim_peak
        simulated_ecg = np.pad(simulated_ecg, ((0, 0), (pad, 0)), mode='constant')[:, :max(reference_ecg.shape[1], simulated_ecg.shape[1] + pad)]
    else:
        pad = sim_peak - ref_peak
        reference_ecg = np.pad(reference_ecg, ((0, 0), (pad, 0)), mode='constant')[:, :max(reference_ecg.shape[1] + pad, simulated_ecg.shape[1])]

    n = min(reference_ecg.shape[1], simulated_ecg.shape[1])
    return reference_ecg[:, :n], simulated_ecg[:, :n], max(ref_peak, sim_peak)


def get_signed_qrs_amplitudes(ecg, qrs_onset_idx, qrs_window_ms, fs):
    qrs_len = max(1, int(round(qrs_window_ms * fs / 1000.0)))
    end = min(ecg.shape[1], qrs_onset_idx + qrs_len)
    seg = ecg[:, qrs_onset_idx:end]
    maxv = np.max(seg, axis=1)
    minv = np.min(seg, axis=1)
    use_pos = np.abs(maxv) >= np.abs(minv)
    amps = np.where(use_pos, maxv, minv)
    return amps

def scale_simulated_to_reference(reference_ecg, simulated_ecg, fs, qrs_onset_idx, qrs_window_ms=150, mode='per_lead', eps=1e-8):
    ref_amp = get_signed_qrs_amplitudes(reference_ecg, qrs_onset_idx, qrs_window_ms, fs)
    sim_amp = get_signed_qrs_amplitudes(simulated_ecg, qrs_onset_idx, qrs_window_ms, fs)

    if mode == 'global':
        s = np.mean(np.abs(ref_amp)) / max(np.mean(np.abs(sim_amp)), eps)
        factors = np.full(reference_ecg.shape[0], s)
    elif mode == 'two_groups':
        if reference_ecg.shape[0] == 12:
            limb_s = np.mean(np.abs(ref_amp[:2])) / max(np.mean(np.abs(sim_amp[:2])), eps)
            prec_s = np.mean(np.abs(ref_amp[6:])) / max(np.mean(np.abs(sim_amp[6:])), eps)
            factors = np.array([limb_s]*6 + [prec_s]*6, dtype=float)
        else:
            limb_s = np.mean(np.abs(ref_amp[:2])) / max(np.mean(np.abs(sim_amp[:2])), eps)
            prec_s = np.mean(np.abs(ref_amp[2:])) / max(np.mean(np.abs(sim_amp[2:])), eps)
            factors = np.array([limb_s, limb_s, prec_s, prec_s, prec_s, prec_s, prec_s, prec_s], dtype=float)
    elif mode == 'per_lead':
        factors = np.abs(ref_amp) / np.maximum(np.abs(sim_amp), eps)
    else:
        raise ValueError("mode deve essere 'global', 'two_groups' oppure 'per_lead'")

    scaled = simulated_ecg * factors[:, None]
    return scaled, factors, ref_amp, sim_amp

def crop_to_same_length(reference_ecg, simulated_ecg, reference_t_ms, simulated_t_ms):
    n = min(reference_ecg.shape[1], simulated_ecg.shape[1])
    return reference_ecg[:, :n], simulated_ecg[:, :n], reference_t_ms[:n], simulated_t_ms[:n]
def read_csv_file(filename, skiprows=0, usecols=None):
    return np.loadtxt(filename, delimiter=',', dtype=float, skiprows=skiprows, usecols=usecols)

def read_ecg_from_csv(filename, nb_leads):
    folded_data = read_csv_file(filename=filename)
    return unfold_ecg_matrix(data=folded_data, nb_leads=nb_leads)

def unfold_ecg_matrix(data, nb_leads):
    # Check dimensions of data to decide how to reshape
    if len(data.shape) == 1:
        ecg = np.reshape(data, (1, int(nb_leads), -1), order='C')
    elif len(data.shape) == 2:
        ecg = np.reshape(data, (data.shape[0], int(nb_leads), -1), order='C')
    else:
        raise "How did you save an array with more than 2 dim in a CSV? This was not supported yet in 2023!"
    return ecg

def visualise_ecgs(reference_ecg, simulated_ecgs, simulated_t_ms, lead_names, casename, ref_t_ms=None, ylim=None, out_png=None):
    nb_leads = len(lead_names)
    if nb_leads == 12:
        fig, axes = plt.subplots(2, 6, figsize=(20, 15), sharex=False)
    else:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=False)
    axes = axes.ravel()

    if ref_t_ms is None:
        ref_t_ms = np.arange(reference_ecg.shape[1])

    for i in range(nb_leads):
        axes[i].plot(ref_t_ms, reference_ecg[i, :], label='Clinical PTB', color='limegreen', linewidth=2.5)
        axes[i].plot(simulated_t_ms, simulated_ecgs[i, :], label='Simulation scaled', color='black', linewidth=1.2)
        axes[i].set_title(lead_names[i])
        axes[i].set_xlabel('Time [ms]')
        axes[i].set_ylabel('mV')
        axes[i].grid(True, alpha=0.3)
        if ylim is not None:
            axes[i].set_ylim(ylim)

    axes[-1].legend(loc='best')
    fig.suptitle(casename)
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.show()

def print_scaling_report(factors, ref_amp, sim_amp, mode, lead_names):
    print('\n=== Scaling report ===')
    print('Mode:', mode)
    for i, lead in enumerate(lead_names):
        print(f"{lead:>2s} | ref_amp={ref_amp[i]: .4f} mV | sim_amp={sim_amp[i]: .4f} mV | scale={factors[i]: .4f}")

def save_ecg_csv(filepath, t_ms, ecg, lead_names):
    header = ['time_ms'] + lead_names
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(ecg.shape[1]):
            writer.writerow([float(t_ms[i])] + [float(ecg[j, i]) for j in range(ecg.shape[0])])

def save_ecg_csv_per_row(filepath, t_ms, ecg, lead_names):
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        for i, lead_name in enumerate(lead_names):
            writer.writerow(ecg[i, :].tolist())

###################################################################################################################
## Main script
filtering = True
normalise = True
zero_align = True
frequency = 1000
high_freq_cut = 150
low_freq_cut = 0.5
max_len_ecg = 500
max_len_qrs = 150
qrs_window_ms = 150.0
nb_leads = 12
qrs_onset = 0
scaling_mode = 'two_groups' # 'global', 'two_groups', 'per_lead'
sim_folder = os.path.join(".", "sb3901","test_pseudobidomain_torord_sb3901_norm6_fast4x_realprep")
casename = 'sb3901'
sim_ecg_igb_filename = casename + '_phie_recovery_norm6.igb'
sim_ecg_igb_path = os.path.join(sim_folder, sim_ecg_igb_filename)
sim_ecg_filename = casename + '_phierec_ascii_norm6_prep_fast.txt'
sim_ecg_path = os.path.join(sim_folder, sim_ecg_filename)
ptb_casename = "patient131"
ptb_filename ="s0273lre"
ptb_record_path=os.path.join('..', '..','ptb-diagnostic-ecg-database-1.0.0', ptb_casename, ptb_filename)  # Path SENZA estensione, es. './ptb_ecg_records/s0010_re'
ptb_target_len_ms = 600.0
ptb_qrs_pre_ms = 120.0
# Preprocess the clinical data
#print('Preprocessing clinical ECGs')
#clinical_data_filename_path = '/mnt/scratch/jenny/'+casename+'_clinical_full_ecg.csv'
#reference_ecg = np.genfromtxt(clinical_data_filename_path, delimiter=',')
#reference_ecg = __preprocess_ecg_without_normalise(original_ecg=reference_ecg, filtering=filtering, zero_align=zero_align,
#                                                   frequency=frequency, low_freq_cut=low_freq_cut, high_freq_cut=high_freq_cut)
#max_qrs_end = qrs_onset + max_len_qrs
#reference_lead_is_positive = __set_reference_lead_is_positive(max_qrs_end=max_qrs_end, reference_ecg=reference_ecg)
#reference_ecg = preprocess_ecg(original_ecg=reference_ecg, reference_ecg=reference_ecg, filtering=filtering, zero_align=zero_align,
#                                frequency=frequency, low_freq_cut=low_freq_cut, high_freq_cut=high_freq_cut, max_len_qrs=max_len_qrs,
#                               qrs_onset=qrs_onset, reference_lead_is_positive=reference_lead_is_positive)

# Read in simulated ECGs
print('Importing and preprocessing simulated ECG')
cmd_convert_igb_txt =[
    'opencarp-docker',
    'igbextract', '-l', "0-9", "-o","asciiTm", "-O",
    sim_ecg_path,
    sim_ecg_igb_path
] 

print("Running command:", " ".join(cmd_convert_igb_txt))
subprocess.run(cmd_convert_igb_txt, check=True)

simulated_t_ms, simulated_ecgs_12leads = import_simulated_ecg_12leads_raw(filename=sim_ecg_path, monoalg_activation_offset=0)
#simulation_frequency = simulated_ecgs_12leads.shape[1]
sim_freq = 1000.0 / np.mean(np.diff(simulated_t_ms)) if len(simulated_t_ms) > 1 else 1000.0

print('simulated_ecgs_12leads.shape ', simulated_ecgs_12leads.shape)
print('simulation_frequency ', sim_freq)
#max_len_qrs = 200 * 4

print('Importing PTB ECG with wfdb...')
reference_t_ms_full, reference_ecg_12_full, reference_fs, reference_sig_names = import_ptb_ecg_12leads(ptb_record_path)
print(f'PTB ECG shape: {reference_ecg_12_full.shape}, fs = {reference_fs:.2f} Hz')
print('Available PTB signals:', reference_sig_names)

#processed_simulated_ecgs = __preprocess_ecg_without_normalise(original_ecg=simulated_ecgs_12leads, filtering=filtering, zero_align=zero_align,
#                                frequency=simulation_frequency, low_freq_cut=low_freq_cut, high_freq_cut=high_freq_cut)
#processed_simulated_ecgs = preprocess_ecg(original_ecg=simulated_ecgs_12leads, reference_ecg=reference_ecg, filtering=filtering, zero_align=zero_align,
#                                frequency=simulation_frequency, low_freq_cut=low_freq_cut, high_freq_cut=high_freq_cut, max_len_qrs=max_len_qrs,
#                               qrs_onset=qrs_onset, reference_lead_is_positive=reference_lead_is_positive)
# Normalizza rispetto al picco globale del QRS (finestra 0-150 ms)
#qrs_window = int(150)  # campioni (con freq=1000 Hz → 150 ms)
#qrs_segment = simulated_ecgs_12leads[:, :qrs_window]
#norm_factor = np.max(np.abs(qrs_segment))
#simulated_ecgs_norm = simulated_ecgs_12leads / norm_factor
#print(f"Fattore di normalizzazione: {norm_factor:.4f} mV")

# Normalizza ciascuna derivazione al proprio picco (per vedere morfologia)
#simulated_ecgs_norm_perlead = simulated_ecgs_12leads.copy()
#for i in range(12):
#    peak = np.max(np.abs(simulated_ecgs_norm_perlead[i, :qrs_window]))
#    if peak > 1e-6:  # evita divisione per zero
#        simulated_ecgs_norm_perlead[i, :] /= peak

# Plot together
print('Visualising ECGs together')
#visualise_ecgs(nb_leads=nb_leads, reference_ecg=simulated_ecgs_12leads, simulated_ecgs=simulated_ecgs_norm, simulated_t=simulated_t,
#               lead_names=['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], casename=casename)
#
# reference_ecg_12_full = resample_ecg(reference_ecg_12_full, original_fs=reference_fs, target_fs=sim_freq)
# reference_t_ms = np.arange(reference_ecg_12_full.shape[1]) * 1000.0 / sim_freq
# reference_fs = sim_freq

reference_ecg_12_full = preprocess_ecg_without_normalise(
    original_ecg=reference_ecg_12_full,
    filtering=filtering,
    zero_align=zero_align,
    frequency=reference_fs,
    high_freq_cut=high_freq_cut,
    low_freq_cut=low_freq_cut,
)

simulated_ecgs_12 = preprocess_ecg_without_normalise(
    original_ecg=simulated_ecgs_12leads,
    filtering=filtering,
    zero_align=zero_align,
    frequency=sim_freq,
    high_freq_cut=high_freq_cut,
    low_freq_cut=low_freq_cut,
)
reference_beat, reference_t_ms, reference_qrs_anchor, central_peak, all_peaks = select_central_ptb_beat(
    reference_ecg_12_full,
    fs=reference_fs,
    target_len_ms=ptb_target_len_ms,
    qrs_pre_ms=ptb_qrs_pre_ms,
)
print(f'Numero di picchi R trovati nel PTB: {len(all_peaks)}')
print(f'Picco centrale selezionato al campione: {central_peak}')
print(f'Beat PTB estratto shape: {reference_beat.shape}')
if abs(reference_fs - sim_freq) > 1e-9:
    reference_beat = resample_ecg(reference_beat, original_fs=reference_fs, target_fs=sim_freq)
    reference_t_ms = np.arange(reference_beat.shape[1]) * 1000.0 / sim_freq
    reference_qrs_anchor = int(round(reference_qrs_anchor * sim_freq / reference_fs))
    reference_fs = sim_freq

n = min(reference_beat.shape[1], simulated_ecgs_12.shape[1])
reference_beat = reference_beat[:, :n]
reference_t_ms = reference_t_ms[:n]
simulated_ecgs_12 = simulated_ecgs_12[:, :n]
simulated_t_ms = simulated_t_ms[:n]

sim_qrs_anchor = np.argmax(np.mean(np.abs(simulated_ecgs_12), axis=0))
reference_beat, reference_t_ms, simulated_ecgs_12, simulated_t_ms = align_qrs_anchor(
    reference_beat,
    reference_t_ms,
    reference_qrs_anchor,
    simulated_ecgs_12,
    simulated_t_ms,
    sim_qrs_anchor,
    fs=sim_freq,
)
scaled_simulated_ecgs_12, scale_factors_12, ref_amp_12, sim_amp_12 = scale_simulated_to_reference(
    reference_ecg=reference_beat,
    simulated_ecg=simulated_ecgs_12,
    fs=sim_freq,
    qrs_onset_idx=min(int(sim_qrs_anchor), reference_beat.shape[1] - 1),
    qrs_window_ms=qrs_window_ms,
    mode=scaling_mode,
)

print_scaling_report(scale_factors_12, ref_amp_12, sim_amp_12, scaling_mode, LEAD_NAMES_12)

# Save and plot 12 leads
clinical_12_csv = os.path.join('.', ptb_casename + '_ptb_clinical_beat_12leads.csv')
scaled_12_csv = os.path.join(sim_folder, casename + '_scaled_simulated_ecg_12leads.csv')
out_12_png = os.path.join(sim_folder, f'{casename}_norm1_scaled_to_{ptb_casename}_{scaling_mode}_12leads.png')

save_ecg_csv(clinical_12_csv, reference_t_ms, reference_beat, LEAD_NAMES_12)
save_ecg_csv_per_row(scaled_12_csv, simulated_t_ms, scaled_simulated_ecgs_12, LEAD_NAMES_12)
save_ecg_csv_per_row(clinical_12_csv, reference_t_ms, reference_beat, LEAD_NAMES_12)

print(f'Saved clinical beat 12 leads CSV: {clinical_12_csv}')
print(f'Saved scaled simulated ECG 12 leads CSV: {scaled_12_csv}')
visualise_ecgs(
    reference_ecg=reference_beat,
    simulated_ecgs=scaled_simulated_ecgs_12,
    simulated_t_ms=simulated_t_ms,
    ref_t_ms=reference_t_ms,
    lead_names=LEAD_NAMES_12,
    casename=f'{casename} | PTB vs scaled simulation ({scaling_mode}) 12 leads',
    ylim=None,
    out_png=out_12_png,
)

# Extract and save 8 leads
indices_8_leads = [LEAD_NAMES_12.index(l) for l in LEAD_NAMES_8]
reference_beat_8 = reference_beat[indices_8_leads, :]
scaled_simulated_ecgs_8 = scaled_simulated_ecgs_12[indices_8_leads, :]

clinical_csv = os.path.join('.', ptb_casename + '_ptb_clinical_beat.csv')
scaled_csv = os.path.join(sim_folder, casename + '_scaled_simulated_ecg.csv')
out_png = os.path.join(sim_folder, f'{casename}_norm1_scaled_to_{ptb_casename}_{scaling_mode}.png')

save_ecg_csv(clinical_csv, reference_t_ms, reference_beat_8, LEAD_NAMES_8)
save_ecg_csv_per_row(scaled_csv, simulated_t_ms, scaled_simulated_ecgs_8, LEAD_NAMES_8)
save_ecg_csv_per_row(clinical_csv, reference_t_ms, reference_beat_8, LEAD_NAMES_8)

print(f'Saved clinical beat CSV: {clinical_csv}')
print(f'Saved scaled simulated ECG CSV: {scaled_csv}')
visualise_ecgs(
    reference_ecg=reference_beat_8,
    simulated_ecgs=scaled_simulated_ecgs_8,
    simulated_t_ms=simulated_t_ms,
    ref_t_ms=reference_t_ms,
    lead_names=LEAD_NAMES_8,
    casename=f'{casename} | PTB vs scaled simulation ({scaling_mode})',
    ylim=None,
    out_png=out_png,
)
