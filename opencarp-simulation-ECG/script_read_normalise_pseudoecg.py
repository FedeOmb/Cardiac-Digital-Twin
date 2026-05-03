# This is a stand alone script that plots ECGs from body surface potentials
# simulated from monodomain simulations.
# The normalisation strategy is the same as in ecg_functions.py.
import numpy as np
from scipy import signal
import matplotlib
import os
#matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import wfdb
#plt.switch_backend('Agg')# backend non-interattivo per ambiente headless
# Helper functions from ecg_functions.py:

LEAD_NAMES_8 = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

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
        limb_s = np.mean(np.abs(ref_amp[:2])) / max(np.mean(np.abs(sim_amp[:2])), eps)
        prec_s = np.mean(np.abs(ref_amp[2:])) / max(np.mean(np.abs(sim_amp[2:])), eps)
        factors = np.array([limb_s, limb_s, prec_s, prec_s, prec_s, prec_s, prec_s, prec_s], dtype=float)
    elif mode == 'per_lead':
        factors = np.abs(ref_amp) / np.maximum(np.abs(sim_amp), eps)
    else:
        raise ValueError("mode deve essere uno tra: 'global', 'two_groups', 'per_lead'")

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

def visualise_ecgs_old(nb_leads, reference_ecg, simulated_ecgs, simulated_t_ms,  lead_names, casename):
    nb_cols = (nb_leads * 2) ** 0.5
    if nb_cols - int(nb_cols) == 0. and nb_cols / 2 - int(nb_cols / 2) == 0.:
        nb_rows = nb_cols / 2
    else:
        # Try to make 2 rows and the necessary columns
        nb_cols = nb_leads / 2
        if nb_cols - int(nb_cols) == 0. and nb_cols / 2 - int(nb_cols / 2) == 0.:
            nb_rows = nb_cols / 2
        else:
            nb_cols = nb_leads
            nb_rows = 1
    fig, axes = plt.subplots(int(nb_rows), int(nb_cols), figsize=(20, 10))
    axes = np.reshape(axes, nb_leads)
    for lead_i in range(nb_leads):
        time_steps = np.arange(reference_ecg.shape[1])
        axes[lead_i].plot(time_steps, reference_ecg[lead_i, :], label='Clinical', color='lime', linewidth=3.)
        axes[lead_i].plot(simulated_t, simulated_ecgs[lead_i, :], color='k', label='Simulation', linewidth=1.)
        axes[lead_i].set_title(lead_names[lead_i])
        axes[lead_i].set_ylim([-1.5, 1.5])
        for tick in axes[lead_i].xaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
        for tick in axes[lead_i].yaxis.get_major_ticks():
            tick.label1.set_fontsize(14)
    axes[nb_leads-1].legend(loc='center left', bbox_to_anchor=(0.1, 0.2), fontsize=14)
    fig.suptitle(casename)
    plt.show()
    #plt.savefig(casename+'_ecg_comparison.png')

def visualise_ecgs(reference_ecg, simulated_ecgs, simulated_t_ms, lead_names, casename, ref_t_ms=None, ylim=None, out_png=None):
    nb_leads = len(lead_names)
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

def print_scaling_report(factors, ref_amp, sim_amp, mode):
    print('\n=== Scaling report ===')
    print('Mode:', mode)
    for i, lead in enumerate(LEAD_NAMES_8):
        print(f"{lead:>2s} | ref_amp={ref_amp[i]: .4f} mV | sim_amp={sim_amp[i]: .4f} mV | scale={factors[i]: .4f}")


###################################################################################################################
## Main script
filtering = True
normalise = False
zero_align = False
frequency = 1000
high_freq_cut = 150
low_freq_cut = 0.5
max_len_ecg = 200
max_len_qrs = 150
qrs_window_ms = 150.0
nb_leads = 8
qrs_onset = 0
scaling_mode = 'per_lead' # 'global', 'two_groups', 'per_lead'
simulation_folder = './test_monodomain_torord_sb301_hf2/'
casename = 'sb301'
sim_ecg_filename=simulation_folder + casename + '_phierec_ascii_hf2.txt'
ptb_record_path=os.path.join('..', 'PTB-ecg', 'HF', 'patient115','s0023_re')  # Path SENZA estensione, es. './ptb_ecg_records/s0010_re'

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
simulated_t_ms, simulated_ecgs_8leads = import_simulated_ecg_8leads_raw(filename=sim_ecg_filename, monoalg_activation_offset=0)
#simulation_frequency = simulated_ecgs_8leads.shape[1]
sim_freq = 1000.0 / np.mean(np.diff(simulated_t_ms)) if len(simulated_t_ms) > 1 else 1000.0

print('simulated_ecgs_8leads.shape ', simulated_ecgs_8leads.shape)
print('simulation_frequency ', sim_freq)
max_len_qrs = 200 * 4

print('Importing PTB ECG with wfdb...')
reference_t_ms, reference_ecg_8, reference_fs, reference_sig_names = import_ptb_ecg_8leads(ptb_record_path)
print(f'PTB ECG shape: {reference_ecg_8.shape}, fs = {reference_fs:.2f} Hz')
print('Available PTB signals:', reference_sig_names)

#processed_simulated_ecgs = __preprocess_ecg_without_normalise(original_ecg=simulated_ecgs_8leads, filtering=filtering, zero_align=zero_align,
#                                frequency=simulation_frequency, low_freq_cut=low_freq_cut, high_freq_cut=high_freq_cut)
#processed_simulated_ecgs = preprocess_ecg(original_ecg=simulated_ecgs_8leads, reference_ecg=reference_ecg, filtering=filtering, zero_align=zero_align,
#                                frequency=simulation_frequency, low_freq_cut=low_freq_cut, high_freq_cut=high_freq_cut, max_len_qrs=max_len_qrs,
#                               qrs_onset=qrs_onset, reference_lead_is_positive=reference_lead_is_positive)
# Normalizza rispetto al picco globale del QRS (finestra 0-150 ms)
#qrs_window = int(150)  # campioni (con freq=1000 Hz → 150 ms)
#qrs_segment = simulated_ecgs_8leads[:, :qrs_window]
#norm_factor = np.max(np.abs(qrs_segment))
#simulated_ecgs_norm = simulated_ecgs_8leads / norm_factor
#print(f"Fattore di normalizzazione: {norm_factor:.4f} mV")

# Normalizza ciascuna derivazione al proprio picco (per vedere morfologia)
#simulated_ecgs_norm_perlead = simulated_ecgs_8leads.copy()
#for i in range(8):
#    peak = np.max(np.abs(simulated_ecgs_norm_perlead[i, :qrs_window]))
#    if peak > 1e-6:  # evita divisione per zero
#        simulated_ecgs_norm_perlead[i, :] /= peak

# Plot together
print('Visualising ECGs together')
#visualise_ecgs(nb_leads=nb_leads, reference_ecg=simulated_ecgs_8leads, simulated_ecgs=simulated_ecgs_norm, simulated_t=simulated_t,
#               lead_names=['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'], casename=casename)

reference_ecg_8 = resample_ecg(reference_ecg_8, original_fs=reference_fs, target_fs=sim_freq)
reference_t_ms = np.arange(reference_ecg_8.shape[1]) * 1000.0 / sim_freq
reference_fs = sim_freq

reference_ecg_8 = __preprocess_ecg_without_normalise(
    original_ecg=reference_ecg_8,
    filtering=filtering,
    zero_align=zero_align,
    frequency=reference_fs,
    high_freq_cut=high_freq_cut,
    low_freq_cut=low_freq_cut,
)

simulated_ecgs_8 = __preprocess_ecg_without_normalise(
    original_ecg=simulated_ecgs_8leads,
    filtering=filtering,
    zero_align=zero_align,
    frequency=sim_freq,
    high_freq_cut=high_freq_cut,
    low_freq_cut=low_freq_cut,
)

reference_ecg_8, simulated_ecgs_8, reference_t_ms, simulated_t = crop_to_same_length(
    reference_ecg_8, simulated_ecgs_8, reference_t_ms, simulated_t_ms
)

reference_ecg_8, simulated_ecgs_8, qrs_anchor = align_by_qrs_peak(
    reference_ecg_8,
    simulated_ecgs_8,
    fs=sim_freq,
    search_window_ms=600,
)

n = min(reference_ecg_8.shape[1], simulated_ecgs_8.shape[1])
reference_ecg_8 = reference_ecg_8[:, :n]
simulated_ecgs_8 = simulated_ecgs_8[:, :n]
reference_t_ms = reference_t_ms[:n]
simulated_t_ms = simulated_t_ms[:n]

scaled_simulated_ecgs_8, scale_factors, ref_amp, sim_amp = scale_simulated_to_reference(
    reference_ecg=reference_ecg_8,
    simulated_ecg=simulated_ecgs_8,
    fs=sim_freq,
    qrs_onset_idx=qrs_anchor,
    qrs_window_ms=qrs_window_ms,
    mode=scaling_mode,
)

print_scaling_report(scale_factors, ref_amp, sim_amp, scaling_mode)

out_png = os.path.join('.', casename + '_ptb_vs_scaled_simulation.png')
visualise_ecgs(
    reference_ecg=reference_ecg_8,
    simulated_ecgs=scaled_simulated_ecgs_8,
    simulated_t_ms=simulated_t_ms,
    ref_t_ms=reference_t_ms,
    lead_names=LEAD_NAMES_8,
    casename=f'{casename} | PTB vs scaled simulation ({scaling_mode})',
    ylim=None,
    out_png=out_png,
)

