#!/bin/bash

SIMS=(
    "run_monodomain_torord_sb3901_norm2.py"
    "run_monodomain_torord_sb3901_norm3.py"
    "run_monodomain_torord_sb3901_norm4.py"
    "run_monodomain_torord_sb3901_norm4.py"
    "run_monodomain_torord_sb4101_norm1.py"
    "run_monodomain_torord_sb4101_norm2.py"
    "run_monodomain_torord_sb4101_norm3.py"
    "run_monodomain_torord_sb4101_norm4.py"
)

for sim in "${SIMS[@]}"; do
    echo "=== Avvio simulazione: $sim ==="
    python $sim --np 12
    echo "=== Fine: $sim ==="
done