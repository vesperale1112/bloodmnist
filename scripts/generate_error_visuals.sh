#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-bloodmnist.npz}
DEVICE=${DEVICE:-auto}
NUM_WORKERS=${NUM_WORKERS:-2}

python -m src.error_visuals \
  --data "$DATA" \
  --device "$DEVICE" \
  --num-workers "$NUM_WORKERS"
