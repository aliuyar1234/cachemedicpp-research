#!/usr/bin/env bash
set -euo pipefail

# Prevent Python from emitting __pycache__/ and .pyc files inside the repo.
export PYTHONDONTWRITEBYTECODE=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="second_model_$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="$ROOT_DIR/empty_dirs_for_codex/outputs/runs/$RUN_ID"
mkdir -p "$RUN_DIR"

python -m cachemedicpp.train --config "$ROOT_DIR/configs/second_model_smoke.yaml" --run_dir "$RUN_DIR"
python -m cachemedicpp.eval --config "$ROOT_DIR/configs/second_model_smoke.yaml" --run_dir "$RUN_DIR"
python -m cachemedicpp.stability --config "$ROOT_DIR/configs/second_model_smoke.yaml" --run_dir "$RUN_DIR"
python -m cachemedicpp.plots --run_dir "$RUN_DIR"

echo "Second model smoke complete. Run directory: $RUN_DIR"
