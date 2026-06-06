#!/usr/bin/env bash
set -euo pipefail

python -m src.analyze_data --data bloodmnist.npz --output-dir outputs/smoke_dataset
python -m src.train \
  --data bloodmnist.npz \
  --model simple_cnn \
  --loss ce \
  --run-name smoke_simple_cnn \
  --epochs 2 \
  --batch-size 64 \
  --num-workers 0 \
  --device auto \
  --max-train-batches 3 \
  --max-val-batches 2

