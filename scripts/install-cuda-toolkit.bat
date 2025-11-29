@echo off
REM install-cuda-toolkit.bat - CUDA Toolkit Installation Guide (Windows)
REM
REM On Windows, CUDA toolkit must be installed via NVIDIA's installer
REM This script provides instructions and download links

echo ========================================
echo CUDA Toolkit Installation (Windows)
echo ========================================
echo.

REM Check if nvcc is already available
where nvcc >nul 2>&1
if %errorlevel% equ 0 (
    echo [92m✓ CUDA toolkit already installed[0m
    echo.
    nvcc --version
    exit /b 0
)

echo [93mCUDA Toolkit is required to compile SageAttention from source[0m
echo.
echo [94mInstallation Steps:[0m
echo.
echo 1. Download CUDA Toolkit 12.6 from NVIDIA:
echo    [96mhttps://developer.nvidia.com/cuda-12-6-0-download-archive[0m
echo.
echo 2. Select your configuration:
echo    - Operating System: Windows
echo    - Architecture: x86_64
echo    - Version: Your Windows version (10 or 11)
echo    - Installer Type: exe (network) [recommended]
echo.
echo 3. Run the installer:
echo    - Choose "Custom" installation
echo    - Select only "CUDA Toolkit"
echo    - Deselect drivers (you already have working CUDA)
echo.
echo 4. After installation completes:
echo    - CUDA_HOME will be set automatically
echo    - nvcc will be added to PATH
echo    - Restart your terminal
echo.
echo 5. Verify installation:
echo    [92mnvcc --version[0m
echo.
echo 6. Install SageAttention:
echo    [92minstall-sageattention.bat[0m
echo.
echo ========================================
echo.
pause
