import os

import numpy as np

case_name = "sb3901"
# Carica i file coordinate cobiveco ab e tm estratti dalla mesh a 500um 
ab_filepath =os.path.join("..","cardiac-data","meta_data","geometric_data", case_name, case_name+'_fine500um', case_name+'_fine500um_nodefield_ab_cut.csv')
ab_values = np.loadtxt(ab_filepath) #coordinate apicobasali (Valori da 0 apice a 1 base)
tm_filepath =os.path.join("..","cardiac-data","meta_data","geometric_data", case_name, case_name+'_fine500um', case_name+'_fine500um_nodefield_tm.csv')
tm_values = np.loadtxt(tm_filepath) # Coordinate transmurali (Valori da 0 epicardio a  endocardio)
n_nodes = len(ab_values)

# valori base e apice di GKs
valore_base_endo = 0.0011 
valore_apice_endo = valore_base_endo * 1.05

#calcolo gradiente ab
gks_ab = valore_apice_endo - (ab_values * (valore_apice_endo - valore_base_endo))

# applica il moltiplicatore Transmurale di Gks
moltiplicatore_epi = 1.4
gks_finale = gks_ab * (1.0 + (moltiplicatore_epi - 1.0) * (-tm_values))

export_filename_noext = f'{case_name}_gks_gradient5abtm'
# salva file dat per visualizzazione con meshalyzer
with open(os.path.join(".", case_name, case_name+'_meshes', f'{export_filename_noext}.dat'), 'w') as f:
     #f.write("1\n")                   
     #f.write(f"{n_nodes}\n")
     for val in gks_finale:
         f.write(f"{val:.6f}\n")

# salva file adj per simulazione opencarp
with open(os.path.join(".", case_name, case_name+'_meshes', f'{export_filename_noext}.adj'), 'w') as f:
    f.write(f"{n_nodes}\n")
    f.write("intra\n") 
    for i, val in enumerate(gks_finale):
        f.write(f"{i} {val:.8f}\n")
        
print("File .adj e .dat gradiente generati")