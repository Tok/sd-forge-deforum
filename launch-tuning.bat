@echo off
REM Deforum Tuning Mode Launcher (Windows)
REM Starts Forge with the Deforum tuning tab enabled

setlocal

set SCRIPT_DIR=%~dp0
set FORGE_DIR=%SCRIPT_DIR%..\..

echo ========================================
echo Deforum Tuning Mode
echo ========================================
echo Starting Forge with tuning tab...
echo ========================================

cd /d "%FORGE_DIR%"

REM Launch with tuning mode flag
python webui.py --deforum-run-tuning %*
