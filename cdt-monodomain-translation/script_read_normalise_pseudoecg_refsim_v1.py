# Script per plottare e scalare ECG simulati da simulazioni monodominio (phie recovery, openCARP)
# verso un ECG di riferimento da un'ALTRA simulazione monodominio già preprocessato in formato CSV.
# La normalizzazione è eseguita solo sugli 8 lead principali (I, II, V1-V6).
import numpy as np
from scipy import signal
import matplotlib
import os
import csv
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import subprocess

LEAD_NAMES_8 = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# ---------------------------------------------------------------------------
# Filtro e preprocessing
# ---------------------------------------------------------------------------
def filter_butterworth_ecg(b, a, ecg):
    return signal.filtfilt(b, a, ecg)

def __filter_ecg(original_ecg, low_a_filtfilt, low_b_filtfilt, high_a_filtfilt, high_b_filtfilt):
    processed_ecg = filter_butterworth_ecg(b=low_b_filtfilt, a=low_a_filtfilt, ecg=original_ecg)
    return filter_butterworth_ecg(b=high_b_filtfilt, a=high_a_filtfilt, ecg=processed_ecg)

def zero_align_ecg(original_ecg):
    return original_ecg - (original_ecg[:, 0:1] + original_ecg[:, -2:-1]) / 2

def preprocess_ecg_without_normalise(original_ecg, filtering, zero_align, frequency, high_freq_cut, low_freq_cut):
    processed_ecg = original_ecg.copy()
    if filtering:
        high_w = high_freq_cut / (frequency / 2.0)
        low_w  = low_freq_cut  / (frequency / 2.0)
        high_b_filtfilt, high_a_filtfilt = signal.butter(4, high_w, 'low')
        low_b_filtfilt,  low_a_filtfilt  = signal.butter(4, low_w,  'high')
        processed_ecg = __filter_ecg(
            processed_ecg,
            low_a_filtfilt=low_a_filtfilt, low_b_filtfilt=low_b_filtfilt,
            high_a_filtfilt=high_a_filtfilt, high_b_filtfilt=high_b_filtfilt,
        )
    if zero_align:
        processed_ecg = zero_align_ecg(processed_ecg)
    return processed_ecg

# ---------------------------------------------------------------------------
# Import ECG simulato "target" da file .txt openCARP (phie recovery)
# ---------------------------------------------------------------------------
def import_simulated_ecg_8leads_raw(filename, monoalg_activation_offset=0):
    """Legge il file ASCII generato da igbextract e restituisce t e 8 lead."""
    data = np.genfromtxt(filename, skip_footer=1)
    t  = data[:, 0] - monoalg_activation_offset
    RA = data[:, 2];  LA = data[:, 1];  LL = data[:, 3]
    VW = 1.0 / 3.0 * (RA + LA + LL)
    V1 = data[:, 5] - VW;  V2 = data[:, 6] - VW;  V3 = data[:, 7] - VW
    V4 = data[:, 8] - VW;  V5 = data[:, 9] - VW;  V6 = data[:, 10] - VW
    I  = LA - RA
    II = LL - RA
    ecgs = np.vstack((I, II, V1, V2, V3, V4, V5, V6))
    return t, ecgs

# ---------------------------------------------------------------------------
# Import ECG di RIFERIMENTO da CSV di simulazione già preprocessata
# ---------------------------------------------------------------------------
def import_reference_ecg_from_csv(csv_path, fs_reference, lead_names=None, reference_start_ms=0.0):
    """
    Importa un ECG di riferimento (simulato, gia' preprocessato) da file CSV.

    Formati supportati:
      A) CSV con intestazione: nomi colonne.
         - Se la prima colonna e' 'time_ms' (o simile), viene usata come asse temporale.
         - Le colonne rimanenti corrispondono a LEAD_NAMES_8 (in qualsiasi ordine).
      B) CSV senza intestazione:
         - 9 colonne -> col 0 = time_ms, col 1-8 = lead nell'ordine di LEAD_NAMES_8.
         - 8 colonne -> tutte le colonne sono lead nell'ordine di LEAD_NAMES_8.
         - 8 righe x N colonne -> formato row-per-lead.

    Restituisce:
        t_ms        : array 1D [n_samples] dei tempi in ms
        ecg         : array 2D [8, n_samples]
        fs_reference: frequenza di campionamento (passata come parametro)
    """
    if lead_names is None:
        lead_names = LEAD_NAMES_8

    with open(csv_path, 'r') as f:
        first_line = f.readline().strip()

    has_header = not all(c in '0123456789.,-+eE \t' for c in first_line)

    if has_header:
        with open(csv_path, 'r') as f:
            reader  = csv.reader(f)
            headers = [h.strip().upper() for h in next(reader)]
        data = np.loadtxt(csv_path, delimiter=',', skiprows=1)

        time_col = None
        for candidate in ['TIME_MS', 'TIME', 'T', 'T_MS']:
            if candidate in headers:
                time_col = headers.index(candidate)
                break

        lead_indices = []
        for lead in lead_names:
            if lead.upper() in headers:
                lead_indices.append(headers.index(lead.upper()))
            else:
                raise ValueError(
                    f"Lead '{lead}' non trovata nel CSV. Colonne disponibili: {headers}"
                )

        ecg = data[:, lead_indices].T  # [8, n_samples]
        if time_col is not None:
            t_ms = data[:, time_col]
        else:
            t_ms = np.arange(ecg.shape[1]) * 1000.0 / fs_reference

    else:
        data = np.loadtxt(csv_path, delimiter=',')

        if data.shape[0] == len(lead_names) and data.shape[1] > len(lead_names):
            # row-per-lead
            ecg  = data
            t_ms = np.arange(ecg.shape[1]) * 1000.0 / fs_reference
        elif data.shape[1] == len(lead_names):
            ecg  = data.T
            t_ms = np.arange(ecg.shape[1]) * 1000.0 / fs_reference
        elif data.shape[1] == len(lead_names) + 1:
            t_ms = data[:, 0]
            ecg  = data[:, 1:].T
        else:
            raise ValueError(
                f"Il CSV ha shape {data.shape} ma ci si aspettano {len(lead_names)} lead. "
                "Verificare formato del file."
            )

    if ecg.shape[0] != len(lead_names):
        raise ValueError(
            f"Shape finale ECG = {ecg.shape}: attesi {len(lead_names)} lead come prima dimensione."
        )
   # --- Trim a partire da reference_start_ms (se richiesto) ---
    if reference_start_ms and reference_start_ms > 0.0:
        idx0 = np.searchsorted(t_ms, reference_start_ms)
        if idx0 >= ecg.shape[1]:
            raise ValueError(f"reference_start_ms ({reference_start_ms} ms) oltre la durata dell'ECG di riferimento ({t_ms[-1] if t_ms.size else '0'} ms).")
        ecg = ecg[:, idx0:]
        # riallinea il tempo in modo che il primo campione valga 0 ms
        t_ms = t_ms[idx0:] - t_ms[idx0]
    return t_ms, ecg, fs_reference

# ---------------------------------------------------------------------------
# Selezione beat centrale dal segnale di riferimento
# ---------------------------------------------------------------------------
def find_r_peaks_from_multilead(ecg, fs, min_distance_ms=500):
    aggregate  = np.mean(np.abs(ecg), axis=0)
    distance   = max(1, int(round(min_distance_ms * fs / 1000.0)))
    prominence = max(np.std(aggregate) * 0.5, 1e-6)
    peaks, _   = signal.find_peaks(aggregate, distance=distance, prominence=prominence)
    return peaks

def select_central_beat(reference_ecg, fs, target_len_ms=600.0, qrs_pre_ms=120.0):
    """
    Seleziona il battito centrale dal segnale di riferimento.
    Se non si trovano picchi R (segnale gia' un singolo battito), usa l'intero segnale.
    """
    peaks = find_r_peaks_from_multilead(reference_ecg, fs)

    if len(peaks) == 0:
        t_ms       = np.arange(reference_ecg.shape[1]) * 1000.0 / fs
        qrs_anchor = int(np.argmax(np.mean(np.abs(reference_ecg), axis=0)))
        print("Nessun picco R rilevato nel riferimento: uso l'intero segnale come beat.")
        return reference_ecg, t_ms, qrs_anchor, qrs_anchor, np.array([qrs_anchor])

    center_time_idx = reference_ecg.shape[1] // 2
    central_peak    = peaks[np.argmin(np.abs(peaks - center_time_idx))]

    half_len = int(round(target_len_ms * fs / 1000.0 / 2.0))
    start    = central_peak - int(round(qrs_pre_ms * fs / 1000.0))
    end      = start + 2 * half_len

    if start < 0:
        end  -= start
        start = 0
    if end > reference_ecg.shape[1]:
        shift = end - reference_ecg.shape[1]
        start = max(0, start - shift)
        end   = reference_ecg.shape[1]

    beat       = reference_ecg[:, start:end]
    t_ms       = np.arange(beat.shape[1]) * 1000.0 / fs
    qrs_anchor = central_peak - start
    return beat, t_ms, qrs_anchor, central_peak, peaks

# ---------------------------------------------------------------------------
# Allineamento QRS
# ---------------------------------------------------------------------------
def find_qrs_anchor(ecg, fs, search_window_ms=250):
    n   = min(ecg.shape[1], int(round(search_window_ms * fs / 1000.0)))
    ref = np.mean(np.abs(ecg[:, :n]), axis=0)
    return int(np.argmax(ref))

def align_qrs_anchor(reference_ecg, reference_t_ms, reference_qrs_anchor,
                     simulated_ecg, simulated_t_ms, simulated_qrs_anchor, fs):
    shift = int(reference_qrs_anchor - simulated_qrs_anchor)
    if shift > 0:
        reference_ecg  = reference_ecg[:, shift:]
        reference_t_ms = np.arange(reference_ecg.shape[1]) * 1000.0 / fs
    elif shift < 0:
        reference_ecg  = np.pad(reference_ecg, ((0, 0), (-shift, 0)), mode='constant')
        reference_t_ms = np.arange(reference_ecg.shape[1]) * 1000.0 / fs
    n = min(reference_ecg.shape[1], simulated_ecg.shape[1])
    return reference_ecg[:, :n], reference_t_ms[:n], simulated_ecg[:, :n], simulated_t_ms[:n]

# ---------------------------------------------------------------------------
# Scaling simulazione target -> riferimento (solo 8 lead)
# ---------------------------------------------------------------------------
def get_signed_qrs_amplitudes(ecg, qrs_onset_idx, qrs_window_ms, fs):
    qrs_len = max(1, int(round(qrs_window_ms * fs / 1000.0)))
    end     = min(ecg.shape[1], qrs_onset_idx + qrs_len)
    seg     = ecg[:, qrs_onset_idx:end]
    maxv    = np.max(seg, axis=1)
    minv    = np.min(seg, axis=1)
    use_pos = np.abs(maxv) >= np.abs(minv)
    return np.where(use_pos, maxv, minv)

def scale_simulated_to_reference_8leads(reference_ecg, simulated_ecg, fs,
                                        qrs_onset_idx, qrs_window_ms=150,
                                        mode='two_groups', eps=1e-8):
    """
    Scala simulated_ecg (8 lead) per adattarne l'ampiezza a reference_ecg (8 lead).

    mode:
      'global'     - unico fattore di scala per tutti gli 8 lead
      'two_groups' - fattore separato per limb leads (I, II) e precordiali (V1-V6)
      'per_lead'   - fattore individuale per ciascun lead
    """
    if reference_ecg.shape[0] != 8 or simulated_ecg.shape[0] != 8:
        raise ValueError("Questa funzione e' hardcodata per 8 lead: I, II, V1-V6.")

    ref_amp = get_signed_qrs_amplitudes(reference_ecg, qrs_onset_idx, qrs_window_ms, fs)
    sim_amp = get_signed_qrs_amplitudes(simulated_ecg, qrs_onset_idx, qrs_window_ms, fs)

    if mode == 'global':
        s       = np.mean(np.abs(ref_amp)) / max(np.mean(np.abs(sim_amp)), eps)
        factors = np.full(8, s)
    elif mode == 'two_groups':
        limb_s  = np.mean(np.abs(ref_amp[:2])) / max(np.mean(np.abs(sim_amp[:2])), eps)
        prec_s  = np.mean(np.abs(ref_amp[2:])) / max(np.mean(np.abs(sim_amp[2:])), eps)
        factors = np.array([limb_s, limb_s,
                            prec_s, prec_s, prec_s, prec_s, prec_s, prec_s], dtype=float)
    elif mode == 'per_lead':
        factors = np.abs(ref_amp) / np.maximum(np.abs(sim_amp), eps)
    else:
        raise ValueError("mode deve essere 'global', 'two_groups' oppure 'per_lead'.")

    scaled = simulated_ecg * factors[:, None]
    return scaled, factors, ref_amp, sim_amp

# ---------------------------------------------------------------------------
# Resample
# ---------------------------------------------------------------------------
def resample_ecg(ecg, original_fs, target_fs):
    if abs(original_fs - target_fs) < 1e-9:
        return ecg.copy()
    n_target = int(round(ecg.shape[1] * target_fs / original_fs))
    return signal.resample(ecg, n_target, axis=1)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def visualise_ecgs(reference_ecg, simulated_ecgs, simulated_t_ms, lead_names,
                   casename, ref_t_ms=None, ylim=None, out_png=None):
    nb_leads = len(lead_names)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=False)
    axes = axes.ravel()
    if ref_t_ms is None:
        ref_t_ms = np.arange(reference_ecg.shape[1])
    for i in range(nb_leads):
        axes[i].plot(ref_t_ms, reference_ecg[i, :],
                     label='Reference ECG', color='limegreen', linewidth=2)
        axes[i].plot(simulated_t_ms, simulated_ecgs[i, :],
                     label='Monodomain translation (scaled)', color='darkred', linewidth=1.5)
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
    print('\n=== Scaling report (8 lead) ===')
    print('Mode:', mode)
    for i, lead in enumerate(lead_names):
        print(f"{lead:>2s} | ref_amp={ref_amp[i]: .4f} mV | "
              f"sim_amp={sim_amp[i]: .4f} mV | scale={factors[i]: .4f}")

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
        for i in range(ecg.shape[0]):
            writer.writerow(ecg[i, :].tolist())

###############################################################################
## MAIN SCRIPT – parametri da modificare
###############################################################################
filtering      = False
normalise      = True
zero_align     = True
frequency      = 1000        # Hz campionamento simulazione target
high_freq_cut  = 150
low_freq_cut   = 0.5
qrs_window_ms  = 200.0
scaling_mode   = 'two_groups'   # 'global', 'two_groups', 'per_lead'

# --- Simulazione TARGET (da preprocessare e scalare) ---
casename             = 'sb4301'
sim_folder           = os.path.join(".", casename, casename + "_mono_torord_LBBB_translation_best_densesparse")
sim_ecg_igb_filename = casename + '_phie_recovery.igb'
sim_ecg_igb_path     = os.path.join(sim_folder, sim_ecg_igb_filename)
sim_ecg_filename     = casename + '_phierec_ascii.txt'
sim_ecg_path         = os.path.join(sim_folder, sim_ecg_filename)

# --- ECG di RIFERIMENTO: altra simulazione monodominio gia' preprocessata (CSV 8 lead) ---
# Formati accettati:
#   - intestazione con nomi colonne (time_ms, I, II, V1, ..., V6)
#   - 8 colonne senza intestazione (campioni x lead, ordine LEAD_NAMES_8)
#   - 9 colonne senza intestazione (time_ms + 8 lead)
#   - 8 righe x N colonne (row-per-lead)
reference_csv_path = os.path.join(".", casename, "sb4301_LBBB_scaled_simulated_ecg_8leads.csv")
reference_fs       = 1000.0  # Hz del CSV di riferimento

# Lunghezza del beat da estrarre dal riferimento (se multicycle)
ref_target_len_ms = 500.0
ref_qrs_pre_ms    = 0.0
ref_qrs_start = 0.0
# ---------------------------------------------------------------------------
# 1. Converti e importa simulazione TARGET
# ---------------------------------------------------------------------------
print('Importando e preprocessando la simulazione target...')
cmd_convert_igb_txt = [
    'opencarp-docker',
    'igbextract', '-l', "0-9", "-o", "asciiTm", "-O",
    sim_ecg_path,
    sim_ecg_igb_path
]
print("Eseguendo:", " ".join(cmd_convert_igb_txt))
subprocess.run(cmd_convert_igb_txt, check=True)

simulated_t_ms, simulated_ecgs_8 = import_simulated_ecg_8leads_raw(
    filename=sim_ecg_path, monoalg_activation_offset=20
)

sim_freq = 1000.0 / np.mean(np.diff(simulated_t_ms)) if len(simulated_t_ms) > 1 else frequency
print(f"Shape ECG target (8 lead): {simulated_ecgs_8.shape}  |  fs = {sim_freq:.2f} Hz")

# ---------------------------------------------------------------------------
# 2. Importa ECG di riferimento dal CSV della simulazione di riferimento
# ---------------------------------------------------------------------------
print(f'\nImportando ECG di riferimento da CSV: {reference_csv_path}')
reference_t_ms_full, reference_ecg_full, reference_fs = import_reference_ecg_from_csv(
    csv_path=reference_csv_path,
    fs_reference=reference_fs,
    lead_names=LEAD_NAMES_8,
    reference_start_ms=ref_qrs_start,
)
print(f"Shape ECG riferimento (8 lead): {reference_ecg_full.shape}  |  fs = {reference_fs:.2f} Hz")
reference_ecg_length = reference_ecg_full.shape[1]
simulated_ecgs_8 = simulated_ecgs_8[:reference_ecg_length]
# ---------------------------------------------------------------------------
# 3. Preprocessing (filtro + zero-align) di entrambi
# ---------------------------------------------------------------------------
reference_ecg_proc = preprocess_ecg_without_normalise(
    original_ecg=reference_ecg_full,
    filtering=filtering, zero_align=zero_align,
    frequency=reference_fs,
    high_freq_cut=high_freq_cut, low_freq_cut=low_freq_cut,
)

simulated_ecg_proc = preprocess_ecg_without_normalise(
    original_ecg=simulated_ecgs_8,
    filtering=filtering, zero_align=zero_align,
    frequency=sim_freq,
    high_freq_cut=high_freq_cut, low_freq_cut=low_freq_cut,
)

# ---------------------------------------------------------------------------
# 4. Estrai beat centrale dal riferimento
# ---------------------------------------------------------------------------
# reference_beat, reference_t_ms, reference_qrs_anchor, central_peak, all_peaks = select_central_beat(
#     reference_ecg_proc,
#     fs=reference_fs,
#     target_len_ms=ref_target_len_ms,
#     qrs_pre_ms=ref_qrs_pre_ms,
# )
# print(f"Picchi R trovati nel riferimento: {len(all_peaks)}")
# print(f"Beat estratto shape: {reference_ecg_full.shape}")

# Resample il beat di riferimento alla frequenza della simulazione target se necessario
# if abs(reference_fs - sim_freq) > 1e-9:
#     reference_beat       = resample_ecg(reference_beat, original_fs=reference_fs, target_fs=sim_freq)
#     reference_t_ms       = np.arange(reference_beat.shape[1]) * 1000.0 / sim_freq
#     reference_qrs_anchor = int(round(reference_qrs_anchor * sim_freq / reference_fs))
#     reference_fs         = sim_freq

# ---------------------------------------------------------------------------
# 5. Allinea QRS target -> riferimento
# ---------------------------------------------------------------------------
# n = min(reference_ecg_full.shape[1], simulated_ecgs_8.shape[1])
# reference_beat     = reference_ecg_full[:, :n]
# reference_t_ms     = reference_t_ms_full[:n]
# simulated_ecg_proc = simulated_ecgs_8[:, :n]
# simulated_t_ms     = simulated_t_ms[:n]

# sim_qrs_anchor = find_qrs_anchor(simulated_ecgs_8, sim_freq)
# reference_qrs_anchor = find_qrs_anchor(reference_ecg_full,reference_fs)
# print("sim_qrs_anchor", sim_qrs_anchor)
# print("ref_qrs_anchor", reference_qrs_anchor)
# reference_beat, reference_t_ms, simulated_ecg_proc, simulated_t_ms = align_qrs_anchor(
#     reference_ecg_full, reference_t_ms_full, reference_qrs_anchor,
#     simulated_ecgs_8, simulated_t_ms, sim_qrs_anchor,
#     fs=sim_freq,
# )

# ---------------------------------------------------------------------------
# 6. Scala la simulazione target verso il riferimento (8 lead, two_groups)
# ---------------------------------------------------------------------------
#qrs_onset_idx = min(int(sim_qrs_anchor), reference_beat.shape[1] - 1)

scaled_simulated_ecgs_8, scale_factors_8, ref_amp_8, sim_amp_8 = scale_simulated_to_reference_8leads(
    reference_ecg=reference_ecg_proc,
    simulated_ecg=simulated_ecg_proc,
    fs=sim_freq,
    qrs_onset_idx=0,
    qrs_window_ms=qrs_window_ms,
    mode=scaling_mode,
)

print_scaling_report(scale_factors_8, ref_amp_8, sim_amp_8, scaling_mode, LEAD_NAMES_8)

# ---------------------------------------------------------------------------
# 7. Salva output CSV
# ---------------------------------------------------------------------------
#ref_csv_out    = os.path.join('.', f'{casename}_reference_sim_beat_8leads.csv')
scaled_csv_out = os.path.join(sim_folder, f'{casename}_scaled_simulated_ecg_8leads.csv')
out_png        = os.path.join(sim_folder, f'{casename}_scaled_to_refsim_{scaling_mode}_8leads.png')

#ave_ecg_csv(ref_csv_out,    reference_t_ms, reference_beat,          LEAD_NAMES_8)
save_ecg_csv_per_row(scaled_csv_out, simulated_t_ms, scaled_simulated_ecgs_8, LEAD_NAMES_8)

#print(f'\nSalvato beat di riferimento CSV: {ref_csv_out}')
print(f'Salvato ECG scalato CSV:          {scaled_csv_out}')

# ---------------------------------------------------------------------------
# 8. Plot
# ---------------------------------------------------------------------------
visualise_ecgs(
    reference_ecg=reference_ecg_proc,
    simulated_ecgs=scaled_simulated_ecgs_8,
    simulated_t_ms=simulated_t_ms,
    ref_t_ms=reference_t_ms_full,
    lead_names=LEAD_NAMES_8,
    casename=f'{casename} | Monodomain translation scaled vs reference ECG',
    ylim=None,
    out_png=out_png,
)
