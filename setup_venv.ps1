# Windows: creates .venv with Python 3.11, CUDA 13.0 PyTorch, core + foundation-model packages.
# Usage (PowerShell, from repo root):   .\setup_venv.ps1          (add -NoFM to skip Chronos/TimesFM/TiRex)
param([switch]$NoFM)
$ErrorActionPreference = "Stop"
py -3.11 -m venv .venv
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements\torch-cu130.txt
if (Test-Path requirements\lock-windows.txt) { & $py -m pip install -r requirements\lock-windows.txt }
else {
  & $py -m pip install -r requirements\core.txt
  if (-not $NoFM) { & $py -m pip install -r requirements\fm.txt }
}
& $py -m pip install --no-deps -e .
& $py -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
& $py -m pvf smoke
