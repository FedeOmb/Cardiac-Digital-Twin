#!/bin/bash

SIMS=(    
    "run_mono_torord_sb4101reg_norm3_CVcampsv2_fast3x_RN7_grad30abtm.py"
    "run_mono_torord_sb4101reg_norm3_CVcampsv2_fast3x_RN7_grad30abtm08.py"
    "run_mono_torord_sb4101reg_norm3_CVcampsv2_fast4x_RN7_grad10abtm.py"
    "run_mono_torord_sb4101reg_norm3_CVcampsv2_fast4x_RN7_grad30abtm.py"
)

for sim in "${SIMS[@]}"; do
    echo "=== Avvio simulazione: $sim ==="
    python $sim --np 20
    echo "=== Fine: $sim ==="
done