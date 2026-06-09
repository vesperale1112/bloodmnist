#!/usr/bin/env bash
set -euo pipefail

DATA=${DATA:-bloodmnist.npz}

python -m src.analyze_data --data "$DATA" --output-dir outputs/dataset
