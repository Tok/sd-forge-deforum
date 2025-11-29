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

REM Build command with optimization flags
if "%USE_OPT%"=="1" (
    echo Starting with optimizations:
    echo   --sage              SageAttention ^(RTX 30/40/50^)
    echo   --fast-fp16         Fast FP16 accumulation
    echo   --cuda-malloc       CUDA malloc optimization
    echo   --cuda-stream       CUDA stream optimization
    echo.

    set CMD=python webui.py --sage --fast-fp16 --cuda-malloc --cuda-stream
) else (
    echo Starting without optimizations
    echo.
    set CMD=python webui.py
)

REM Add extra args
if not "%EXTRA_ARGS%"=="" (
    echo Extra flags: !EXTRA_ARGS!
    echo.
    set CMD=!CMD! !EXTRA_ARGS!
)

echo ========================================
echo Launching Forge...
echo ========================================
echo.

REM Execute
!CMD!
