# One-command setup on Windows (run from the repo root in PowerShell):
#     powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1
# Creates .venv (Python 3.11), installs the exact package versions from requirements\lock-windows.txt,
# picks the PyTorch CUDA build that matches the installed NVIDIA driver, then checks the machine.
#   -Cuda cu130|cu126|cpu   force a build instead of auto-detect
param([ValidateSet("auto", "cu130", "cu126", "cpu")][string]$Cuda = "auto")
$ErrorActionPreference = "Stop"

if (-not (Get-Command py -ErrorAction SilentlyContinue) -or -not (py -0 | Select-String "3.11")) {
    Write-Host "Python 3.11 not found. Install it:  winget install -e --id Python.Python.3.11   then re-run." -ForegroundColor Red
    exit 1
}
if ($Cuda -eq "auto") {
    $Cuda = "cpu"
    try {
        $drv = [double]((nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Split(".")[0])
        if ($drv -ge 580) { $Cuda = "cu130" } elseif ($drv -ge 528) { $Cuda = "cu126" }
        Write-Host "NVIDIA driver $drv -> PyTorch build $Cuda"
    } catch { Write-Host "No NVIDIA driver found -> CPU build (training will be slow)" -ForegroundColor Yellow }
}
if (-not (Test-Path .venv)) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -m pip install -r "requirements\torch-$Cuda.txt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -m pip install -r requirements\lock-windows.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -c "import pvf; raise SystemExit(0 if pvf.check_environment() else 1)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "`nOpen notebooks\run_experiment.ipynb with the .venv kernel (VS Code: pick .venv as kernel, or run .\.venv\Scripts\jupyter lab)" -ForegroundColor Green
