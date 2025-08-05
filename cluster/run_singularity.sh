#!/usr/bin/env bash

# run_singularity.sh
# Modified version without network/X11 requirements

set -euo pipefail

# --- Paths ---
CONTAINER_IMAGE="/cluster/scratch/yixili/mpinets"
CODE_DIR="/cluster/home/yixili/motion-policy-networks"
DATA_DIR="/cluster/home/yixili/motion_policy/pretrain_data"
CHECKPOINT_DIR="/cluster/home/yixili/motion-policy-networks/checkpoints"

echo "Starting Singularity container..."

# --- Targeted Python bytecode cleanup ---
echo "Cleaning Python 3.7 bytecode caches..."
if singularity exec --writable-tmpfs "$CONTAINER_IMAGE" \
   find /usr/lib/python3.7 -name "*.pyc" -delete 2>/dev/null ; then
    echo "Bytecode cleanup successful"
else
    echo "Bytecode cleanup completed (some files may not exist)"
fi

# --- Dependency Management ---
echo "Ensuring package compatibility..."
singularity exec --writable-tmpfs "$CONTAINER_IMAGE" \
  bash -c "pip install --upgrade --no-cache-dir pip && \
           pip install --force-reinstall --no-deps --no-cache-dir \
           gitpython==3.1.44 \
           wandb==0.18.7 \
           numpy==1.21.6 && \
           pip check"

singularity exec \
  --nv \
  --cleanenv \
  --writable-tmpfs \
  --bind "${CODE_DIR}:/root/mpinets" \
  --bind "${DATA_DIR}:/data" \
  --bind "${CHECKPOINT_DIR}:/workspace" \
  --env NVIDIA_DRIVER_CAPABILITIES="all" \
  --env ACCEPT_EULA="Y" \
  --env PYTHONPATH="/root/mpinets" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONDONTWRITEBYTECODE=1 \
  "${CONTAINER_IMAGE}" \
  bash -c "export PYTHONPATH=/root/mpinets:\$PYTHONPATH && \
           git config --global --add safe.directory /root/mpinets && \
           cd /root/mpinets && \
           wandb login e69097b8c1bd646d9218e652823487632097445d && \
           python3 -B /root/mpinets/mpinets/run_training.py /root/mpinets/train_configs/pretrain.yaml --no-checkpointing"