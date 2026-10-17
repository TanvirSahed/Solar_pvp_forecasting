#!/usr/bin/env bash
# Linux/macOS (no CUDA on macOS -> CPU). Usage: bash setup_venv.sh [--no-fm]
set -euo pipefail
python3.11 -m venv .venv
PY=.venv/bin/python
$PY -m pip install --upgrade pip
$PY -m pip install -r requirements/torch-cu130.txt
if [ -f requirements/lock-linux.txt ]; then $PY -m pip install -r requirements/lock-linux.txt
else
  $PY -m pip install -r requirements/core.txt
  [ "${1:-}" = "--no-fm" ] || $PY -m pip install -r requirements/fm.txt
fi
$PY -m pip install --no-deps -e .
$PY -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
$PY -m pvf smoke
