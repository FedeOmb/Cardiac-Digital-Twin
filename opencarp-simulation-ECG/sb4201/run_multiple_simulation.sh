#!/bin/bash

SIMS=(    
    "run_mono_torord_sb4201_norm9_CVcampsv2_RNv3_fast3x_prep0.py"
    "run_mono_torord_sb4201_norm9_CVcampsv2_RNv3_fast4x_prep0.py"
    "run_mono_torord_sb4201_norm9_CVcampsv2_RNv3_fast4x_prep0_grad10abtm.py"
)

for sim in "${SIMS[@]}"; do
    echo "=== Avvio simulazione: $sim ==="
    python $sim --np 20
    echo "=== Fine: $sim ==="
done