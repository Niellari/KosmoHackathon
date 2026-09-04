#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXECUTABLE="$PYTHON_BIN"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
  PYTHON_EXECUTABLE="$PROJECT_DIR/.venv/bin/python"
else
  PYTHON_EXECUTABLE="python3"
fi

cd "$PROJECT_DIR"
exec "$PYTHON_EXECUTABLE" -m api.submit "$@"

