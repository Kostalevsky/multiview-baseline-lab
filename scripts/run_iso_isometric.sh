#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="$PROJECT_ROOT/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Virtual environment not found."
    echo "Create it with:"
    echo "python3 -m venv .venv"
    exit 1
fi

MODEL="${1:-$PROJECT_ROOT/examples/models/part.obj}"

echo "Running ISO Isometric baseline"
echo "Model: $MODEL"

"$PYTHON" "$PROJECT_ROOT/run.py" \
    --baseline iso_isometric \
    --input "$MODEL"