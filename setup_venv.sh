#!/usr/bin/env bash
# One-command setup on Linux:  bash setup_venv.sh [cu130|cu126|cpu]
# Same pinned versions as Windows (requirements/lock-linux.txt); CUDA build picked from the NVIDIA driver.
set -euo pipefail
CUDA="${1:-auto}"
case "$CUDA" in
  auto|cu130|cu126|cpu) ;;
  *) echo "Unknown build: $CUDA (use auto, cu130, cu126 or cpu)" >&2; exit 1 ;;
esac
command -v python3.11 >/dev/null || { echo "python3.11 not found (e.g. sudo apt install python3.11 python3.11-venv)"; exit 1; }
if [ "$CUDA" = auto ]; then
  CUDA=cpu
  if command -v nvidia-smi >/dev/null &&
     DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1) &&
     [[ "$DRV" =~ ^[0-9]+$ ]]; then
    if [ "$DRV" -ge 580 ]; then CUDA=cu130; elif [ "$DRV" -ge 525 ]; then CUDA=cu126; fi
    echo "NVIDIA driver $DRV -> PyTorch build $CUDA"
  fi
fi
[ -d .venv ] || python3.11 -m venv .venv
PY=.venv/bin/python
$PY -m pip install --upgrade pip
$PY -m pip install -r "requirements/torch-$CUDA.txt"
$PY -m pip install -r requirements/lock-linux.txt
$PY -m pip install --no-deps -e .
$PY -c "import pvf; raise SystemExit(0 if pvf.check_environment() else 1)"
echo "Open notebooks/run_experiment.ipynb (e.g. .venv/bin/jupyter lab)"
