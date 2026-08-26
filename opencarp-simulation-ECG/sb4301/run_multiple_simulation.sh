#!/bin/bash

SIMS=(    
    "run_monodomain_torord_sb4301_LBBB_CVcampsv2_fast4x_grad10abtm_purkinje300_prep0.py"
    "run_monodomain_torord_sb4301_RBBB_CVcampsv2_fast4x_grad10abtm_purkinje300_prep0.py"
)

for sim in "${SIMS[@]}"; do
    echo "=== Avvio simulazione: $sim ==="
    python $sim --np 20
    echo "=== Fine: $sim ==="
done