@echo off
REM run-tuning-lab.bat - Deforum Tuning Lab Launcher (Windows)
REM Starts Forge with the Deforum tuning tab enabled and optimization flags
REM
REM Usage:
REM   run-tuning-lab.bat           Start with optimizations
REM   run-tuning-lab.bat --no-opt  Start without optimizations

setlocal enabledelayedexpansion

REM Get Forge directory (two levels up from scripts/)
set SCRIPT_DIR=%~dp0
set FORGE_DIR=%SCRIPT_DIR%..\..\\

echo ========================================
echo Deforum Tuning Lab
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

    set CMD=python webui.py --sage --fast-fp16 --cuda-malloc --cuda-stream --deforum-api --deforum-run-tuning
) else (
    echo Starting without optimizations
    echo.
    set CMD=python webui.py --deforum-api --deforum-run-tuning
)

REM Add extra args
if not "%EXTRA_ARGS%"=="" (
    set CMD=!CMD! !EXTRA_ARGS!
)

echo ========================================
echo Launching Forge...
echo ========================================
echo.

REM Execute
!CMD!
