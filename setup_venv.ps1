# One-step setup on Windows: double-click setup.bat, or run from the repo root in PowerShell:
#     powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1
# Nothing has to be installed first, not even Python. The script
#   1. downloads uv (a single-file Python manager) into .tools\
#   2. lets uv download a private Python 3.11 into .tools\python (other Pythons on the machine are ignored)
#   3. creates .venv with the exact package versions from requirements\lock-windows.txt and the PyTorch build that
#      matches the NVIDIA driver (CPU build without one)
#   4. checks the machine and the Data\ folder.
# Running it again is safe: it completes or repairs the setup. Delete .venv and .tools to start over.
#   -Cuda cu130|cu126|cpu   force a PyTorch build instead of auto-detect
param([ValidateSet("auto", "cu130", "cu126", "cpu")][string]$Cuda = "auto")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$UvVersion = "0.13.7"
$tools = Join-Path $PSScriptRoot ".tools"
$uv = Join-Path $tools "uv.exe"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $tools "python"   # keep the Python interpreter inside the project folder

if (-not (Test-Path $uv)) {
    Write-Host "Downloading uv $UvVersion (one time) ..."
    New-Item -ItemType Directory -Force $tools | Out-Null
    $zip = Join-Path $tools "uv.zip"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -UseBasicParsing -OutFile $zip `
        "https://github.com/astral-sh/uv/releases/download/$UvVersion/uv-x86_64-pc-windows-msvc.zip"
    Expand-Archive $zip -DestinationPath $tools -Force
    Remove-Item $zip
}

if ($Cuda -eq "auto") {
    $Cuda = "cpu"
    try {
        $drv = [double]((nvidia-smi --query-gpu=driver_version --format=csv,noheader | Select-Object -First 1).Split(".")[0])
        if ($drv -ge 580) { $Cuda = "cu130" } elseif ($drv -ge 528) { $Cuda = "cu126" }
        Write-Host "NVIDIA driver $drv -> PyTorch build $Cuda"
    } catch { Write-Host "No NVIDIA driver found -> CPU build (training will be slow)" -ForegroundColor Yellow }
}

$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    & $uv venv --seed --python 3.11 --python-preference only-managed .venv
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $uv pip install --python $py -r "requirements\torch-$Cuda.txt"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $uv pip install --python $py -r requirements\lock-windows.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $uv pip install --python $py --no-deps -e .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -c "import pvf; raise SystemExit(0 if pvf.check_environment() else 1)"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "`nReady. Double-click start_notebook.bat to open the experiment notebook in the browser," -ForegroundColor Green
Write-Host "or open this folder in VS Code and pick the .venv kernel for notebooks\run_experiment.ipynb." -ForegroundColor Green
