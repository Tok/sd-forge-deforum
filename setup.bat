@echo off
REM setup.bat - Deforum setup script for Windows
REM
REM Usage:
REM   setup.bat              Interactive menu
REM   setup.bat check        Check status only
REM   setup.bat install      Install dependencies
REM   setup.bat help         Show help

setlocal enabledelayedexpansion

REM Get directories
set FORGE_DIR=%~dp0..\..\
set EXTENSION_DIR=%~dp0

REM Parse arguments
set MODE=interactive
if "%1"=="check" set MODE=check
if "%1"=="install" set MODE=install
if "%1"=="help" goto :help
if "%1"=="/?" goto :help
if "%1"=="-h" goto :help

if "%MODE%"=="interactive" goto :menu
if "%MODE%"=="check" goto :check
if "%MODE%"=="install" goto :install

:help
echo Deforum Setup Script (Windows)
echo.
echo Usage:
echo   setup.bat              Interactive menu
echo   setup.bat check        Check current status
echo   setup.bat install      Install Deforum dependencies
echo   setup.bat help         Show this help
echo.
echo Modes:
echo   check     - Display Python version, dependencies, optimizations
echo   install   - Install Deforum requirements.txt
echo.
echo Examples:
echo   setup.bat check        Quick status check
echo   setup.bat install      Install missing deps
echo.
echo Documentation:
echo   docs\SETUP.md          Comprehensive setup guide
echo   CLAUDE.md              Development documentation
echo.
echo Note: venv migration not supported on Windows
echo       (manually delete venv and run webui-user.bat)
exit /b 0

:menu
echo ========================================
echo Deforum Setup
echo ========================================
echo.
echo What would you like to do?
echo.
echo   1) Check status
echo   2) Install Deforum dependencies
echo   3) Exit
echo.
set /p choice="Choose [1-3]: "

if "%choice%"=="1" goto :check
if "%choice%"=="2" goto :install
if "%choice%"=="3" exit /b 0
echo Invalid choice
exit /b 1

:check
echo ========================================
echo Deforum Status Check
echo ========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    exit /b 1
)
echo.

REM Check PyTorch
echo Checking PyTorch...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo PyTorch: NOT INSTALLED
)
echo.

REM Check Deforum dependencies
echo Checking Deforum dependencies...
python -c "deps=['pandas','rich','librosa','soundfile','plotly','numexpr','av','pims','gdown','easydict']; missing=[]; [missing.append(d) if not __import__('importlib').util.find_spec(d) else None for d in deps]; print('Missing:', missing) if missing else print('All dependencies installed')" 2>nul
echo.

REM Check optimizations
echo Checking Forge optimizations...
python -c "import sageattention" 2>nul && echo SageAttention: Installed || echo SageAttention: Not installed
python -c "import flash_attn" 2>nul && echo FlashAttention: Installed || echo FlashAttention: Not installed
echo.

echo ========================================
echo Status check complete
echo ========================================
echo.
echo See docs\SETUP.md for optimization flags
echo.
if "%MODE%"=="check" exit /b 0
pause
goto :menu

:install
echo ========================================
echo Installing Deforum Dependencies
echo ========================================
echo.

REM Check if in Forge directory
if not exist "%FORGE_DIR%\venv" (
    echo ERROR: Forge venv not found at %FORGE_DIR%\venv
    echo.
    echo Please run this from the Deforum extension directory
    echo inside a working Forge installation.
    exit /b 1
)

REM Activate venv and install
echo Installing from requirements.txt...
call "%FORGE_DIR%\venv\Scripts\activate.bat"
pip install -r "%EXTENSION_DIR%\requirements.txt"

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete
echo ========================================
echo.
if "%MODE%"=="install" exit /b 0
pause
goto :menu
