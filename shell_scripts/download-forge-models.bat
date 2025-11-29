@echo off
REM download-forge-models.bat - Download base Forge models (Windows)
REM
REM This script helps download the base models required by Forge WebUI.
REM For Flux and Deforum-specific models, use download-all-models.bat instead.
REM
REM Reference: https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set FORGE_ROOT=%SCRIPT_DIR%..\..\..\

echo ========================================
echo Forge Base Models Download Script
echo ========================================
echo.
echo This script helps you download SD1/SDXL models.
echo For Flux/Deforum models, use download-all-models.bat
echo.

cd /d "%FORGE_ROOT%"

REM Create directories
if not exist "models\Stable-diffusion" mkdir "models\Stable-diffusion"
if not exist "models\VAE" mkdir "models\VAE"
if not exist "models\text_encoder" mkdir "models\text_encoder"

REM Check for HuggingFace CLI
where huggingface-cli >nul 2>&1
if errorlevel 1 (
    echo ERROR: huggingface-cli not found!
    echo.
    echo Install with: pip install huggingface-hub
    exit /b 1
)

echo Which models would you like to download?
echo.
echo   1^) SD1 VAE ^(vae-ft-mse-840000^) - ~330MB
echo   2^) SDXL VAE ^(sdxl-vae-fp16-fix^) - ~320MB
echo   3^) Both VAEs
echo   4^) Skip ^(will use CivitAI for checkpoints^)
echo.
set /p choice="Enter choice [1-4]: "

if "%choice%"=="1" goto :sd1_vae
if "%choice%"=="2" goto :sdxl_vae
if "%choice%"=="3" goto :both_vaes
if "%choice%"=="4" goto :skip_vae
goto :invalid

:sd1_vae
echo.
echo Downloading SD1 VAE...
huggingface-cli download stabilityai/sd-vae-ft-mse vae-ft-mse-840000-ema-pruned.safetensors --local-dir models\VAE --local-dir-use-symlinks False --resume-download
echo Done: SD1 VAE downloaded
goto :checkpoint_info

:both_vaes
echo.
echo Downloading SD1 VAE...
huggingface-cli download stabilityai/sd-vae-ft-mse vae-ft-mse-840000-ema-pruned.safetensors --local-dir models\VAE --local-dir-use-symlinks False --resume-download
echo Done: SD1 VAE downloaded

:sdxl_vae
echo.
echo Downloading SDXL VAE...
huggingface-cli download madebyollin/sdxl-vae-fp16-fix sdxl_vae.safetensors --local-dir models\VAE --local-dir-use-symlinks False --resume-download
echo Done: SDXL VAE downloaded
goto :checkpoint_info

:skip_vae
echo Skipping VAE downloads
goto :checkpoint_info

:invalid
echo Invalid choice
exit /b 1

:checkpoint_info
echo.
echo ========================================
echo Checkpoint Download Information
echo ========================================
echo.
echo For SD1/SDXL checkpoints, visit CivitAI:
echo.
echo   SD 1.5: https://civitai.com/models/6424/chilloutmix
echo   SDXL:   https://civitai.com/models/101055/sd-xl
echo.
echo Download .safetensors files and place them in:
echo   %FORGE_ROOT%\models\Stable-diffusion\
echo.
echo ========================================
echo Base model setup complete!
echo ========================================
echo.
echo Next steps:
echo   1. Download checkpoints from CivitAI ^(if needed^)
echo   2. Run download-all-models.bat for Flux/Deforum models
echo   3. Launch Forge with: start-forge.bat
echo.
