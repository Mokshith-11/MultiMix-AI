#!/usr/bin/env bash
set -e

echo "=========================================="
echo "MultiMix AI — Cloud Container Startup"
echo "=========================================="

# 1. Verify and create essential runtime directories
echo "[1/3] Ensuring audio & runtime directories exist..."
mkdir -p assets/audio/generated
mkdir -p assets/audio/uploads
mkdir -p models

# 2. Optionally run model provisioning only when explicitly requested
# Set PROVISION_MODELS=1 or PROVISION_MODELS=true to provision models into /app/models
if [ "${PROVISION_MODELS}" = "1" ] || [ "${PROVISION_MODELS}" = "true" ] || [ "${DOWNLOAD_MODELS}" = "1" ]; then
    echo "[2/3] Explicit model provisioning enabled. Running download_models.py..."
    python scripts/download_models.py
else
    echo "[2/3] Startup model auto-provisioning is OFF (using existing/mounted models in models/)."
    echo "      To auto-provision at startup, set environment variable PROVISION_MODELS=1."
fi

# Sanity check for model directory content
if [ ! -d "models" ] || [ -z "$(ls -A models 2>/dev/null)" ]; then
    echo "NOTE: models/ directory appears empty or unmounted."
    echo "      Ensure persistent storage volume is mounted to /app/models or run download_models.py."
fi

# 3. Launch Streamlit server
PORT_NUM="${PORT:-8501}"
echo "[3/3] Launching Streamlit on 0.0.0.0:${PORT_NUM} (headless=true)..."

exec streamlit run app.py \
  --server.port="${PORT_NUM}" \
  --server.address=0.0.0.0 \
  --server.headless=true
