#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Prefer the currently-active venv interpreter (usually `python`), but allow override.
# Many Linux distros don't ship `python` in PATH anymore, only `python3`.
PYTHON_BIN="${PYTHON:-}"

if [ -z "${PYTHON_BIN}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "ERROR: Cannot find python/python3. Activate your venv or set PYTHON=/path/to/python." >&2
    exit 1
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: PYTHON is set to '$PYTHON_BIN' but it's not found in PATH." >&2
  echo "Set PYTHON to an absolute path, e.g. PYTHON=/path/to/venv/bin/python" >&2
  exit 1
fi

# Install deps if pip is available (typical in venvs). Otherwise, just try to run.
if "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install -r requirements.txt
else
  echo "WARNING: pip not available for $PYTHON_BIN; skipping install. If it fails, install pip/deps first." >&2
fi

"$PYTHON_BIN" run_web_labeler.py


