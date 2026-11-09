#!/bin/bash
set -e

echo "Starting Jupyter Server for Opencarp Monodomain Launcher - $(date)"

JUPYTER_PORT="${JUPYTER_PORT:-9090}"
MONODOMAIN_PATH="${MONODOMAIN_PATH:-/data/Cardiac-Digital-Twin/cdt-monodomain-translation}"
echo "Starting JupyterLab on port ${JUPYTER_PORT}..."

exec jupyter lab \
    --ip=0.0.0.0 \
    --port="${JUPYTER_PORT}" \
    --no-browser \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --ServerApp.root_dir="${MONODOMAIN_PATH}" \
    --ServerApp.base_url="/monodomain/"