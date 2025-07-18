#!/usr/bin/env bash

# run_singularity.sh
# Optimized version with precise bytecode cleanup

set -euo pipefail

# --- Hardcoded Paths ---
CONTAINER_IMAGE="/cluster/scratch/yixili/mpinets"
CODE_DIR="/cluster/home/yixili/motion-policy-networks"
DATA_DIR="/cluster/home/yixili/motion_policy/pretrain_data"
CHECKPOINT_DIR="/cluster/home/yixili/motion-policy-networks/checkpoints"

echo "Starting Singularity container: $CONTAINER_IMAGE"

# --- Targeted Python bytecode cleanup ---
echo "Cleaning Python 3.7 bytecode caches..."
singularity exec --writable-tmpfs "$CONTAINER_IMAGE" \
  find /usr/lib/python3.7 -name "*.pyc" -delete 2>/dev/null || true

# --- Execute with bytecode prevention ---
singularity exec \
  --nv \
  --containall --writable-tmpfs \
  --bind "${CODE_DIR}:/root/mpinets" \
  --bind "${DATA_DIR}:/data" \
  --bind "${CHECKPOINT_DIR}:/workspace" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH="/root/mpinets:\${PYTHONPATH:-}" \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env ACCEPT_EULA=Y \
  --env PYTHONDONTWRITEBYTECODE=1 \
  "${CONTAINER_IMAGE}" \
  bash -c "wandb login e69097b8c1bd646d9218e652823487632097445d && \
           cd /root/mpinets && \
           python3 -B -m pip install --no-compile -e . && \
           cd / && \
           python3 -B /root/mpinets/mpinets/run_training.py /root/mpinets/train_configs/pretrain.yaml"

echo "=== Completed ==="