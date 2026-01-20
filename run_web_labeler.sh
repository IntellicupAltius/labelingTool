#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Prefer the currently-active venv interpreter (usually `python`), but allow override.
PYTHON_BIN="${PYTHON:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: Cannot find python. Activate your venv first (or set PYTHON=/path/to/python)." >&2
  exit 1
fi

# Install deps if pip is available (typical in venvs). Otherwise, just try to run.
if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install -r requirements.txt
else
  echo "WARNING: pip not available for $PYTHON_BIN; skipping install. If it fails, install pip/deps first." >&2
fi

"$PYTHON_BIN" run_web_labeler.py


