#!/usr/bin/env bash
set -euo pipefail

python -m src.summarize_runs --runs-dir outputs/runs --output-dir outputs/summary
