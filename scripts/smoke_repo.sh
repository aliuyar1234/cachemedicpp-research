#!/usr/bin/env bash
set -euo pipefail

# Prevent Python from emitting __pycache__/ and .pyc files inside the repo.
export PYTHONDONTWRITEBYTECODE=1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 "$ROOT_DIR/scripts/check_junk.py"
python3 "$ROOT_DIR/scripts/forbid_placeholders.py"
python3 "$ROOT_DIR/scripts/check_links.py"
python3 "$ROOT_DIR/scripts/check_paper_contracts.py"
python3 "$ROOT_DIR/scripts/validate_configs.py"
python3 "$ROOT_DIR/scripts/verify_manifest.py" || {
  echo "MANIFEST verification failed. If you changed files intentionally, run: python3 scripts/make_manifest.py";
  exit 2;
}

echo "OK: smoke_repo passed."
