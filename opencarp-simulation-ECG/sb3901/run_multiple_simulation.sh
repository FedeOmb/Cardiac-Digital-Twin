#!/bin/bash

SIMS=(    
    "run_mono_torord_sb3901_LBBB_fast4x_prep0.py"
    "run_mono_torord_sb3901_RBBB_fast4x_prep0.py"
)

for sim in "${SIMS[@]}"; do
    echo "=== Avvio simulazione: $sim ==="
    python $sim --np 12
    echo "=== Fine: $sim ==="
done