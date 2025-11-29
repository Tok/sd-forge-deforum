@echo off
REM start-forge.bat - Start Forge with Deforum optimizations
REM
REM This script launches Forge WebUI with recommended optimization flags:
REM   --sage              SageAttention (auto-installs on first run)
REM   --fast-fp16         Fast FP16 accumulation (PyTorch 2.7+)
REM   --cuda-malloc       CUDA malloc optimization
REM   --cuda-stream       CUDA stream optimization
REM
REM Usage:
REM   start-forge.bat                Start with optimizations
REM   start-forge.bat --listen       Add custom flags
REM   start-forge.bat --no-opt       Start without optimizations

setlocal enabledelayedexpansion

REM Get Forge directory
set FORGE_DIR=%~dp0..\..\

echo ========================================
echo Forge WebUI + Deforum
echo ========================================
echo.

cd /d "%FORGE_DIR%"

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

    set FLAGS=--sage --fast-fp16 --cuda-malloc --cuda-stream
) else (
    echo Starting without optimizations
    echo.
    set FLAGS=
)

REM Add extra args
if not "%EXTRA_ARGS%"=="" (
    echo Extra flags: !EXTRA_ARGS!
    echo.
    set FLAGS=!FLAGS! !EXTRA_ARGS!
)

echo ========================================
echo Launching Forge...
echo ========================================
echo.

REM Execute - use webui-user.bat if available, otherwise venv\Scripts\python.exe
if exist "webui-user.bat" (
    REM webui-user.bat handles venv activation
    call webui-user.bat !FLAGS!
) else if exist "venv\Scripts\python.exe" (
    REM Use venv python directly
    venv\Scripts\python.exe webui.py !FLAGS!
) else (
    echo ERROR: Neither webui-user.bat nor venv\Scripts\python.exe found!
    echo Please run from Forge root directory with a valid venv.
    exit /b 1
)
