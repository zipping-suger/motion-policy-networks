#!/usr/bin/env bash

# run_singularity.sh
# Modified version with bytecode corruption prevention

set -euo pipefail

# --- Hardcoded Paths ---
CONTAINER_IMAGE="/cluster/scratch/yixili/mpinets"
CODE_DIR="/cluster/home/yixili/motion-policy-networks"
DATA_DIR="/cluster/home/yixili/motion_policy/pretrain_data"
CHECKPOINT_DIR="/cluster/home/yixili/motion-policy-networks/checkpoints"

echo "Starting Singularity container: $CONTAINER_IMAGE"

# --- Clean Python bytecode caches ---
echo "Cleaning Python bytecode caches..."
singularity exec --writable-tmpfs "$CONTAINER_IMAGE" \
  find /usr -name "*.pyc" -delete

# --- Run Singularity with bytecode prevention ---
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
  --env PYTHONDONTWRITEBYTECODE=1 \  # Prevents new .pyc creation
  "${CONTAINER_IMAGE}" \
  bash -c "wandb login e69097b8c1bd646d9218e652823487632097445d && \
           cd /root/mpinets && \
           python -B -m pip install -e . && \  # -B prevents bytecode
           cd / && \
           python -B /root/mpinets/mpinets/run_training.py /root/mpinets/train_configs/pretrain.yaml"

echo "=== Completed ==="