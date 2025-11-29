@echo off
REM install-sageattention.bat - Install SageAttention (Windows)
REM
REM Requires:
REM   - PyTorch installed in Forge venv
REM   - CUDA toolkit (nvcc, CUDA_HOME) for compilation

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set EXTENSION_DIR=%SCRIPT_DIR%..\
set FORGE_DIR=%SCRIPT_DIR%..\..\..\

echo ========================================
echo SageAttention Installation
echo ========================================
echo.

REM Check if already installed
"%FORGE_DIR%\venv\Scripts\python.exe" -c "import sageattention" >nul 2>&1
if %errorlevel% equ 0 (
    echo [92m✓ SageAttention already installed[0m
    exit /b 0
)

echo [94mPrerequisites check:[0m

REM Check PyTorch
"%FORGE_DIR%\venv\Scripts\python.exe" -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo [91m✗ PyTorch not installed[0m
    echo [93mRun: setup.bat[0m
    exit /b 1
)
for /f "delims=" %%i in ('"%FORGE_DIR%\venv\Scripts\python.exe" -c "import torch; print(torch.__version__)"') do set TORCH_VERSION=%%i
echo [92m✓ PyTorch !TORCH_VERSION![0m

REM Check CUDA toolkit
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo [91m✗ CUDA toolkit (nvcc) not found[0m
    echo [93mRun: install-cuda-toolkit.bat for instructions[0m
    exit /b 1
)
echo [92m✓ CUDA toolkit found[0m

REM Check CUDA_HOME
if not defined CUDA_HOME (
    echo [93m⚠ CUDA_HOME not set[0m
    echo [93mTrying default location...[0m
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6" (
        set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6
        echo [92m✓ Set CUDA_HOME=!CUDA_HOME![0m
    ) else (
        echo [91m✗ Could not find CUDA installation[0m
        exit /b 1
    )
) else (
    echo [92m✓ CUDA_HOME=!CUDA_HOME![0m
)

echo.
echo [96mInstalling SageAttention...[0m
echo [94mUsing --no-build-isolation to access torch during build[0m
echo [93mThis may take 5-10 minutes to compile...[0m
echo.

cd /d "%FORGE_DIR%"

REM Install with --no-build-isolation
"%FORGE_DIR%\venv\Scripts\pip.exe" install --no-build-isolation sageattention
if %errorlevel% equ 0 (
    echo.
    echo [92m✓ SageAttention installed successfully![0m
    echo.
    echo [94mYou can now use --sage flag when launching Forge:[0m
    echo [92mstart-forge.bat[0m
) else (
    echo.
    echo [91m✗ Installation failed[0m
    echo.
    echo [93mCommon issues:[0m
    echo   1. CUDA_HOME not set
    echo   2. nvcc not in PATH
    echo   3. Visual Studio Build Tools not installed
    echo.
    echo [94mCheck build log above for specific error[0m
    exit /b 1
)
