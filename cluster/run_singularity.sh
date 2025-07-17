#!/usr/bin/env bash

# run_singularity.sh
# Hardcoded paths and arguments for singularity execution

set -euo pipefail

# --- Hardcoded Paths ---
CONTAINER_IMAGE="/cluster/scratch/yixili/mpinets"
CODE_DIR="/cluster/home/yixili/motion-policy-networks"
DATA_DIR="/cluster/home/yixili/motion_policy/pretrain_data"
CHECKPOINT_DIR="/cluster/home/yixili/motion-policy-networks/checkpoints"  # Directory for saving checkpoints

echo "Starting Singularity container: $CONTAINER_IMAGE"

# --- Run Singularity  ---
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
  "${CONTAINER_IMAGE}" \
  bash -c "wandb login e69097b8c1bd646d9218e652823487632097445d && python3 mpinets/mpinets/run_training.py mpinets/train_configs/pretrain.yaml"

echo "=== Completed ==="
