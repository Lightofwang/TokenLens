#!/bin/bash
set -e
cd "$(dirname "$0")"

# Prioritize Apple Silicon Homebrew Python (/opt/homebrew/bin/python3), then Intel/system Python
PYTHON_BIN=""
for candidate in \
  "/opt/homebrew/bin/python3" \
  "/opt/homebrew/opt/python@3.14/bin/python3" \
  "/opt/homebrew/opt/python@3.13/bin/python3" \
  "/opt/homebrew/opt/python@3.12/bin/python3" \
  "/usr/local/bin/python3" \
  "$(command -v python3 2>/dev/null || true)"; do
  if [ -n "$candidate" ] && [ -x "$candidate" ]; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [ -z "$PYTHON_BIN" ]; then
  echo "[Error] Python 3.10+ not found. On Apple Silicon Mac, install via: brew install python"
  exit 1
fi

echo "Starting TokenLens using $PYTHON_BIN ($(uname -m))..."
"$PYTHON_BIN" -m tokenlens
