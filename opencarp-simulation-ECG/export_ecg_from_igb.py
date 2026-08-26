import os
#from carputils import ecg
import ecg_module_carputils as ecg
import matplotlib
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
import matplotlib.pyplot as plt
#plt.switch_backend('Agg')# backend non-interattivo per ambiente headless

subject_name = "sb4101"
sim_folder_name = "test_mono_torord_sb4101reg_norm1_CVcampsv2_fast4x_prep0"
igb_path = os.path.join(".", subject_name+'_reg', sim_folder_name, "sb4101_phie_recovery.igb")

# 1. Leggi il file igb prodotto dalla phie recovery
t, datadict = ecg.readfile(igb_path, sample_time=1.0, silent=False)
# t        → array (T,) in ms
# datadict → dict con chiavi: 'I','II','III','aVR','aVL','aVF','V1'–'V6','vm'
#            'vm' è il Vm medio (da Vm.igb se disponibile)
temp_LA = datadict['LA']
temp_RA = datadict['RA']
datadict['LA'] = temp_RA
datadict['RA'] = temp_LA
# 2. Calcola le 12 derivazioni standard
outdata = ecg.calc_std_ecg(datadict, vl3D_method='lead')

# 3. Filtra (opzionale ma consigliato per confronto con ECG clinico)
outdata_filt = ecg.filter_std_ecg(
    outdata,
    filter_order=2,
    freq_lowpass=150.0,    # Hz
    freq_highpass=0.05,    # Hz
    freq_bandstop=None
)

out_path = os.path.join(".", subject_name+'_reg', sim_folder_name, "output_ecg")
if not os.path.exists(out_path):
    os.makedirs(out_path)

# 4. Salva derivazioni
ecg.save_std_ecg(t, outdata, outdata_filt,
                  filepath=out_path,
                  filebase="sb3901_phierec_ecgexport")

# 5. Calcola features (QRS, QT, PR, asse...)
features = ecg.calc_ecg_features(t, outdata_filt, datadict['vm'],
                                   params={'tlim': [min(t), max(t)], 'ylim': None},
                                   silent=False)

ecg.save_features(features, filepath=out_path)

# 6. Accedi direttamente alle derivazioni come numpy array
print(outdata_filt['V1'].shape)   # (T,) in mV
print(f"I max: {outdata_filt['I'].max()}")    # picco R in mV
print(f"II max: {outdata_filt['II'].max()}")   # picco R in mV
params = {'tlim': [min(t), max(t)], 'ylim': None}

ecg.savefig_einthoven(t, outdata, outdata_filt, params=params,filepath=out_path, silent=False)
ecg.savefig_goldberger(t, outdata, outdata_filt, params=params, filepath=out_path, silent=False)
ecg.savefig_wilson(t, outdata, outdata_filt,params=params, filepath=out_path, silent=False)
ecg.savefig_vectorloop(t, outdata, outdata_filt, params=params, filepath=out_path, loop_meth='lead', silent=False)
ecg.vectorloop_plot2D(t, outdata, outdata_filt, features=features, params=params, filepath=out_path, silent=False)
ecg.vectorloop_plot3D(t, outdata, outdata_filt, features=features, params=params, filepath=out_path, silent=False)
ecg.periodogram_plot(t, outdata, outdata_filt, params=params, filepath=out_path, silent=False)