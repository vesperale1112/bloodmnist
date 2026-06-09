#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-bloodmnist.npz}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-128}
DEVICE=${DEVICE:-auto}
NUM_WORKERS=${NUM_WORKERS:-2}

python -m src.train \
  --data "$DATA" \
  --model resnet18 \
  --loss ce \
  --augment \
  --lr 5e-4 \
  --weight-decay 1e-3 \
  --patience 10 \
  --run-name resnet18_compare \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --num-workers "$NUM_WORKERS"
