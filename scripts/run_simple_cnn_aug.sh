#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-bloodmnist.npz}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-128}
DEVICE=${DEVICE:-auto}
NUM_WORKERS=${NUM_WORKERS:-2}

python -m src.train \
  --data "$DATA" \
  --model simple_cnn \
  --loss ce \
  --augment \
  --run-name simple_cnn_aug_ce \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --num-workers "$NUM_WORKERS"
