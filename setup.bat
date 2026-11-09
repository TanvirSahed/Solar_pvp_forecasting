@echo off
rem Double-click to set up everything. No Python installation is needed.
rem The first run downloads about 3 GB (mostly PyTorch) and takes 5 to 15 minutes; later runs are quick.
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_venv.ps1" %*
echo.
pause
