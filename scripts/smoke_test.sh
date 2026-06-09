#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-bloodmnist.npz}
DEVICE=${DEVICE:-auto}

python -m src.analyze_data --data "$DATA" --output-dir outputs/smoke_dataset
python -m src.train \
  --data "$DATA" \
  --model simple_cnn \
  --loss ce \
  --run-name smoke_simple_cnn \
  --epochs 2 \
  --batch-size 64 \
  --num-workers 0 \
  --device "$DEVICE" \
  --max-train-batches 3 \
  --max-val-batches 2
