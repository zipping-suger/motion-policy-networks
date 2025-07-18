#!/usr/bin/env bash

# run_singularity.sh
# Fixed version with proper variable naming and robust error handling

set -euo pipefail

# --- Hardcoded Paths ---
CONTAINER_IMAGE="/cluster/scratch/yixili/mpinets"  # FIXED: Correct spelling
CODE_DIR="/cluster/home/yixili/motion-policy-networks"
DATA_DIR="/cluster/home/yixili/motion_policy/pretrain_data"
CHECKPOINT_DIR="/cluster/home/yixili/motion-policy-networks/checkpoints"

echo "Starting Singularity container: $CONTAINER_IMAGE"

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