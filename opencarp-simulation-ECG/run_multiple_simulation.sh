#!/bin/bash

SIMS=(
    "run_monodomain_torord_sb3901_9rn_norm2_prepace.py"
    "run_monodomain_torord_sb301_7rn_hfv4_prepace.py"
    "run_monodomain_torord_sb301_7rn_hfv5_prepace.py"
)

for sim in "${SIMS[@]}"; do
    echo "=== Avvio simulazione: $sim ==="
    python $sim --np 12
    echo "=== Fine: $sim ==="
done