#!/usr/bin/env bash
# One-step setup on Linux:  bash setup_venv.sh [cu130|cu126|cpu]
# Nothing has to be installed first, not even Python: uv (a single-file Python manager) is downloaded into .tools/,
# it downloads a private Python 3.11 into .tools/python, and .venv gets the pinned versions from
# requirements/lock-linux.txt plus the PyTorch build that matches the NVIDIA driver. Safe to run again.
set -euo pipefail
cd "$(dirname "$0")"
CUDA="${1:-auto}"
case "$CUDA" in
  auto|cu130|cu126|cpu) ;;
  *) echo "Unknown build: $CUDA (use auto, cu130, cu126 or cpu)" >&2; exit 1 ;;
esac
UV_VERSION=0.13.7
TOOLS="$PWD/.tools"
UV="$TOOLS/uv"
export UV_PYTHON_INSTALL_DIR="$TOOLS/python"
if [ ! -x "$UV" ]; then
  echo "Downloading uv $UV_VERSION (one time) ..."
  mkdir -p "$TOOLS"
  curl -LsSf "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz" \
    | tar -xz -C "$TOOLS" --strip-components=1
fi
if [ "$CUDA" = auto ]; then
  CUDA=cpu
  if command -v nvidia-smi >/dev/null &&
     DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1) &&
     [[ "$DRV" =~ ^[0-9]+$ ]]; then
    if [ "$DRV" -ge 580 ]; then CUDA=cu130; elif [ "$DRV" -ge 525 ]; then CUDA=cu126; fi
    echo "NVIDIA driver $DRV -> PyTorch build $CUDA"
  fi
fi
PY=.venv/bin/python
[ -x "$PY" ] || "$UV" venv --seed --python 3.11 --python-preference only-managed .venv
"$UV" pip install --python "$PY" -r "requirements/torch-$CUDA.txt"
"$UV" pip install --python "$PY" -r requirements/lock-linux.txt
"$UV" pip install --python "$PY" --no-deps -e .
"$PY" -c "import pvf; raise SystemExit(0 if pvf.check_environment() else 1)"
echo "Ready. Open notebooks/run_experiment.ipynb (e.g. .venv/bin/jupyter lab notebooks/run_experiment.ipynb)"
