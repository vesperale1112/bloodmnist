#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-bloodmnist.npz}
SPLIT=${SPLIT:-test}
BATCH_SIZE=${BATCH_SIZE:-256}
DEVICE=${DEVICE:-auto}
NUM_WORKERS=${NUM_WORKERS:-2}
CALIBRATION_BINS=${CALIBRATION_BINS:-15}

runs=(
  simple_cnn_ce
  simple_cnn_aug_ce
  improved_cnn_ce
  improved_cnn_weighted_ce
  resnet18_compare
)

for run_name in "${runs[@]}"; do
  checkpoint="outputs/runs/${run_name}/checkpoints/best.pt"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Skip ${run_name}: checkpoint not found at ${checkpoint}"
    continue
  fi

  python -m src.evaluate \
    --data "$DATA" \
    --checkpoint "$checkpoint" \
    --split "$SPLIT" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --num-workers "$NUM_WORKERS" \
    --calibration-bins "$CALIBRATION_BINS" \
    --output-dir "outputs/runs/${run_name}/eval_${SPLIT}"
done
