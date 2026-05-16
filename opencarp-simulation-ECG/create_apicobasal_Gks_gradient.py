import numpy as np

# 1. Carica il file Cobiveco 'ab' della tua MESH 500um (Valori da 0.0 Apice a 1.0 Base)
ab_values = np.loadtxt('../cardiac-data/meta_data/geometric_data/sb3901/sb3901_fine500um/sb3901_fine500um_nodefield_ab.csv')
n_nodes = len(ab_values)
# 2. Crea il gradiente di GKs (Aumenta dalla base verso l'apice)
# Tomek dice che l'aumento causa 25ms di differenza. Spesso questo si ottiene
# moltiplicando GKs all'apice per circa 1.5x - 2.5x rispetto alla base.
valore_base = 0.0011 
valore_apice = valore_base * 1.5  

# Formula lineare: se ab=1 (Base) -> risultato=1.0. Se ab=0 (Apice) -> risultato=2.0
gks_gradient = valore_apice - (ab_values * (valore_apice - valore_base))

# 3. Salva per openCARP
# with open('./sb3901/sb3901_meshes_v2/gks_tomek_gradient.dat', 'w') as f:
#     f.write("1\n")                   
#     f.write(f"{len(gks_gradient)}\n")
#     for val in gks_gradient:
#         f.write(f"{val:.6f}\n")

with open('./sb3901/sb3901_meshes/gks_tomek_gradient2.adj', 'w') as f:
    f.write(f"{n_nodes}\n")
    f.write("intra\n") 
    for i, val in enumerate(gks_gradient):
        f.write(f"{i} {val:.8f}\n") # Scrive l'indice del nodo e il valore
        
print("File .adj generato con successo!")