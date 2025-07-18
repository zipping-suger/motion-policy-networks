#!/usr/bin/env bash

# run_singularity.sh
# Matches Docker functionality and runs training script

set -euo pipefail

# --- Paths ---
CONTAINER_IMAGE="/cluster/scratch/yixili/mpinets"
CODE_DIR="/cluster/home/yixili/motion-policy-networks"
DATA_DIR="/cluster/home/yixili/motion_policy/pretrain_data"
CHECKPOINT_DIR="/cluster/home/yixili/motion-policy-networks/checkpoints"

# --- Display settings (matching Docker's X11 forwarding) ---
export DISPLAY=${DISPLAY:-:0}
export XAUTHORITY=${XAUTHORITY:-$HOME/.Xauthority}

echo "Starting Singularity container "

singularity exec \
  --nv \
  --cleanenv \
  --writable-tmpfs \
  --containall \
  --net \
  --network-args "portmap=host" \
  --bind "${CODE_DIR}:/root/mpinets" \
  --bind "${DATA_DIR}:/data" \
  --bind "${CHECKPOINT_DIR}:/workspace" \
  --bind "/tmp/.X11-unix:/tmp/.X11-unix" \
  --bind "${XAUTHORITY}:${XAUTHORITY}" \
  --env DISPLAY="$DISPLAY" \
  --env XAUTHORITY="$XAUTHORITY" \
  --env NVIDIA_DRIVER_CAPABILITIES="all" \
  --env ACCEPT_EULA="Y" \
  --env PYTHONPATH="/root/mpinets" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONDONTWRITEBYTECODE=1 \
  "${CONTAINER_IMAGE}" \
  bash -c "export PYTHONPATH=/root/mpinets:\$PYTHONPATH && \
           git config --global --add safe.directory /root/mpinets && \
           cd /root/mpinets && \
           python3 mpinets/mpinets/run_training.py mpinets/train_configs/pretrain.yaml"