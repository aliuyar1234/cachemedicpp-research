#!/usr/bin/env bash
set -euo pipefail

# Prevent Python from emitting __pycache__/ and .pyc files inside the repo.
export PYTHONDONTWRITEBYTECODE=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="$ROOT_DIR/outputs/runs"
mkdir -p "$OUT_ROOT"

python -m cachemedicpp.sweep --sweep_config "$ROOT_DIR/configs/full_gpt2_matrix.yaml" --out_root "$OUT_ROOT"
