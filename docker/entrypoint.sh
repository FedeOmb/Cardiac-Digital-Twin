#!/bin/bash
set -e

echo "[CDT] Starting CDT Inference container - $(date)"

# Verify bind mount for CDT framework
if [ ! -d "${CDT_FRAMEWORK_PATH}" ]; then
    echo "ERROR: CDT framework not found in ${CDT_FRAMEWORK_PATH}"
    echo "Check that volume for CDT Framework is mounted correctly in Docker container"
    exit 1
fi

# Verify bind mount for NFS data volume
if [ ! -d "${DATA_PATH}" ] || [ -z "$(ls -A ${DATA_PATH} 2>/dev/null)" ]; then
    echo "[CDT] Warning: ${DATA_PATH} is empty or not mounted"
fi
JUPYTER_PORT="${JUPYTER_PORT:-8888}"
echo "Starting JupyterLab on port ${JUPYTER_PORT}..."

exec jupyter lab \
    --ip=0.0.0.0 \
    --port="${JUPYTER_PORT}" \
    --no-browser \
    --NotebookApp.token='' \
    --NotebookApp.password='' \
    --ServerApp.root_dir="${CDT_FRAMEWORK_PATH}" \
    --ServerApp.base_url="/cdt-framework/"