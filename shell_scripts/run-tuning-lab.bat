@echo off
REM run-tuning-lab.bat - Deforum Tuning Lab Launcher (Windows)
REM Starts Forge with the Deforum tuning tab enabled and optimization flags
REM
REM Usage:
REM   run-tuning-lab.bat           Start with optimizations
REM   run-tuning-lab.bat --no-opt  Start without optimizations

setlocal enabledelayedexpansion

REM Get Forge directory (three levels up from shell_scripts/)
set SCRIPT_DIR=%~dp0
set EXTENSION_DIR=%SCRIPT_DIR%..\
set FORGE_DIR=%SCRIPT_DIR%..\..\..\

echo ========================================
echo Deforum Tuning Lab
echo ========================================
echo.

REM Check for existing WebUI instances to prevent duplicate launches
REM This prevents VRAM exhaustion from multiple instances running simultaneously
tasklist /FI "IMAGENAME eq python.exe" /FI "WINDOWTITLE eq *launch.py*" 2>nul | find /I "python.exe" >nul
if %ERRORLEVEL%==0 (
    echo.
    echo ========================================
    echo ERROR: WebUI is already running!
    echo ========================================
    echo.
    echo Running multiple instances causes VRAM exhaustion and OOM errors.
    echo.
    echo To stop existing instances:
    echo   1. Close the WebUI browser tab
    echo   2. Press Ctrl+C in the WebUI terminal
    echo   3. Or: taskkill /F /IM python.exe /FI "WINDOWTITLE eq *launch.py*"
    echo.
    echo Then restart with:
    echo   .\shell_scripts\run-tuning-lab.bat
    echo.
    exit /b 1
)

cd /d "%FORGE_DIR%"

REM Clear Python bytecode cache to ensure latest code is loaded
echo Clearing Python bytecode cache...
del /s /q "extensions\sd-forge-deforum\*.pyc" >nul 2>&1
for /d /r "extensions\sd-forge-deforum" %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" 2>nul
echo Cache cleared
echo.

REM Check for --no-opt flag
set USE_OPT=1
set EXTRA_ARGS=

:parse_args
if "%~1"=="" goto :done_parsing
if "%~1"=="--no-opt" (
    set USE_OPT=0
) else (
    set EXTRA_ARGS=!EXTRA_ARGS! %~1
)
shift
goto :parse_args

:done_parsing

REM Build flags with optimization
if "%USE_OPT%"=="1" (
    echo Starting with optimizations:
    echo   --sage              SageAttention ^(RTX 30/40/50^)
    echo   --fast-fp16         Fast FP16 accumulation
    echo   --cuda-malloc       CUDA malloc optimization
    echo   --cuda-stream       CUDA stream optimization
    echo.
    echo Note: If --sage fails, run: setup.bat
    echo.

    set FLAGS=--sage --fast-fp16 --cuda-malloc --cuda-stream --deforum-api --deforum-run-tuning
) else (
    echo Starting without optimizations
    echo.
    set FLAGS=--deforum-api --deforum-run-tuning
)

REM Add extra args
if not "%EXTRA_ARGS%"=="" (
    set FLAGS=!FLAGS! !EXTRA_ARGS!
)

echo ========================================
echo Launching Forge...
echo ========================================
echo.

REM Execute - use webui-user.bat if available, otherwise venv\Scripts\python.exe
if exist "%FORGE_DIR%\webui-user.bat" (
    REM webui-user.bat handles venv activation
    call "%FORGE_DIR%\webui-user.bat" !FLAGS!
) else if exist "%FORGE_DIR%\venv\Scripts\python.exe" (
    REM Use venv python directly
    "%FORGE_DIR%\venv\Scripts\python.exe" webui.py !FLAGS!
) else (
    echo ERROR: Neither webui-user.bat nor venv\Scripts\python.exe found!
    echo Please run from Forge root directory with a valid venv.
    exit /b 1
)
