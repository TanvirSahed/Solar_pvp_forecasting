@echo off
rem Double-click to open the experiment notebook in the browser (JupyterLab). Close this window to stop JupyterLab.
cd /d "%~dp0"
if not exist ".venv\Scripts\jupyter.exe" (
    echo Setup is missing: double-click setup.bat first.
    pause
    exit /b 1
)
".venv\Scripts\jupyter.exe" lab notebooks\run_experiment.ipynb
