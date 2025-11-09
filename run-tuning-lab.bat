@echo off
REM Deforum Tuning Lab Launcher (Windows)
REM Starts Forge with the Deforum tuning tab enabled

setlocal

set SCRIPT_DIR=%~dp0
set FORGE_DIR=%SCRIPT_DIR%..\..

echo ========================================
echo Deforum Tuning Lab
echo ========================================
echo Starting Forge with tuning tab...
echo ========================================

cd /d "%FORGE_DIR%"

REM Use webui.bat if available, otherwise fallback to python webui.py
if exist "webui.bat" (
    call webui.bat --deforum-run-tuning %*
) else (
    python webui.py --deforum-run-tuning %*
)
