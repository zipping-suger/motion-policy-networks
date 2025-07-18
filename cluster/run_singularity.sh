#!/usr/bin/env bash

# run_singularity.sh
# Optimized version for existing container environment

set -euo pipefail

# --- Hardcoded Paths ---
CONTAINER_IMAGE="/cluster/scratch/yixili/mpinets"
CODE_DIR="/cluster/home/yixili/motion-policy-networks"
DATA_DIR="/cluster/home/yixili/motion_policy/pretrain_data"
CHECKPOINT_DIR="/cluster/home/yixili/motion-policy-networks/checkpoints"

# --- Environment Validation ---
echo "Validating container environment..."
if [[ ! -f "$CONTAINER_IMAGE" ]]; then
    echo "Error: Container image not found at $CONTAINER_IMAGE"
    exit 1
fi

# --- System Preparation ---
echo "Preparing Python environment..."
singularity exec --writable-tmpfs "$CONTAINER_IMAGE" \
  bash -c "find /usr/lib/python3.7 /usr/local/lib/python3.7 -name '*.pyc' -delete 2>/dev/null || true && \
           python3 -c 'import git; import wandb; import numpy; print(\"Core packages verified\")'"

# --- Dependency Management ---
echo "Ensuring package compatibility..."
singularity exec --writable-tmpfs "$CONTAINER_IMAGE" \
  bash -c "pip install --upgrade --no-cache-dir pip && \
           pip install --force-reinstall --no-deps --no-cache-dir \
           gitpython==3.1.44 \
           wandb==0.18.7 \
           numpy==1.21.6 && \
           pip check"

# --- Training Execution ---
echo "Starting training workflow..."
singularity exec \
  --nv \
  --containall --writable-tmpfs \
  --bind "${CODE_DIR}:/root/mpinets" \
  --bind "${DATA_DIR}:/data" \
  --bind "${CHECKPOINT_DIR}:/workspace" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH="/root/mpinets" \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env ACCEPT_EULA=Y \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env GIT_PYTHON_REFRESH=quiet \
  --env WANDB_SILENT=true \
  "${CONTAINER_IMAGE}" \
  bash -c "cd /root/mpinets && \
           python3 -B -m pip install --no-compile --no-deps -e . && \
           python3 -B /root/mpinets/mpinets/run_training.py /root/mpinets/train_configs/pretrain.yaml"

echo "=== Training Completed ==="