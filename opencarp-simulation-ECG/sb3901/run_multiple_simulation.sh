#!/bin/bash

SIMS=(    
    "run_mono_torord_sb3901_norm5_CVmincole_fast4x_realprep_nomid_grad10abtm.py"
    "run_mono_torord_sb3901_norm5_CVmincole_fast4x_realprep_nomid_grad30abtm.py"
    "run_mono_torord_sb3901_norm5_CVmincole_fast4x_realprep_nomid_grad50abtm.py"
)

for sim in "${SIMS[@]}"; do
    echo "=== Avvio simulazione: $sim ==="
    python $sim --np 12
    echo "=== Fine: $sim ==="
done