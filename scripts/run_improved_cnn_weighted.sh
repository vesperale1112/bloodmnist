#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-bloodmnist.npz}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-128}
DEVICE=${DEVICE:-auto}
NUM_WORKERS=${NUM_WORKERS:-2}

python -m src.train \
  --data "$DATA" \
  --model improved_cnn \
  --loss weighted_ce \
  --augment \
  --run-name improved_cnn_weighted_ce \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --num-workers "$NUM_WORKERS"
