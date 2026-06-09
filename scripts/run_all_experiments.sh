#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-bloodmnist.npz}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-128}
DEVICE=${DEVICE:-auto}
NUM_WORKERS=${NUM_WORKERS:-2}

export DATA EPOCHS BATCH_SIZE DEVICE NUM_WORKERS

bash scripts/analyze_data.sh
bash scripts/run_simple_cnn.sh
bash scripts/run_simple_cnn_aug.sh
bash scripts/run_improved_cnn.sh
bash scripts/run_improved_cnn_weighted.sh
bash scripts/summarize_results.sh
