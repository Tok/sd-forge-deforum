@echo off
REM Download Flux 2 Klein models for Forge Neo (Windows)
REM Based on: https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models

setlocal enabledelayedexpansion

REM BB0 Slopcore colors (ANSI escape codes for Windows 10+)
REM Enable ANSI colors in Windows console
for /F "tokens=1,2 delims=#" %%a in ('"prompt #$H#$E# & echo on & for %%b in (1) do rem"') do (
  set "ESC=%%b"
)

REM Color codes
set "BB0_ZENITH=%ESC%[38;2;23;167;254m"
set "BB0_GLITCH=%ESC%[38;2;255;20;147m"
set "BB0_VOID=%ESC%[38;2;86;6;255m"
set "BB0_MIDNIGHT=%ESC%[38;2;55;87;255m"
set "NC=%ESC%[0m"

REM Detect Forge Neo root
set "SCRIPT_DIR=%~dp0"
set "EXTENSION_ROOT=%SCRIPT_DIR%.."
for %%I in ("%EXTENSION_ROOT%\..") do set "EXTENSIONS_DIR=%%~fI"
for %%I in ("%EXTENSIONS_DIR%\..") do set "FORGE_ROOT=%%~fI"

echo %BB0_VOID%=== Flux 2 Klein Model Downloader ===%NC%
echo.
echo This script downloads Flux 2 Klein models for Forge Neo.
echo Forge Neo root: %FORGE_ROOT%
echo.

REM Check if we're in the right place
if not exist "%FORGE_ROOT%\models" (
    echo %BB0_GLITCH%ERROR: Cannot find models directory at %FORGE_ROOT%\models%NC%
    echo Please run this script from the sd-forge-deforum\shell_scripts directory
    pause
    exit /b 1
)

REM Create model directories
if not exist "%FORGE_ROOT%\models\Stable-diffusion" mkdir "%FORGE_ROOT%\models\Stable-diffusion"
if not exist "%FORGE_ROOT%\models\text_encoder" mkdir "%FORGE_ROOT%\models\text_encoder"
if not exist "%FORGE_ROOT%\models\VAE" mkdir "%FORGE_ROOT%\models\VAE"

echo %BB0_MIDNIGHT%Which Klein model do you want to download?%NC%
echo 1) Flux 2 Klein 4B (~13GB VRAM, Apache 2.0 - commercial use OK)
echo 2) Flux 2 Klein 9B (~29GB VRAM, non-commercial license)
echo 3) Both models
echo 4) Exit
set /p "model_choice=Choice [1-4]: "

if "%model_choice%"=="4" (
    echo Exiting...
    exit /b 0
)

echo.
echo %BB0_MIDNIGHT%Choose checkpoint precision:%NC%
echo 1) FP8 (recommended - smaller, faster)
echo 2) BF16 (full precision - larger)
set /p "precision_choice=Choice [1-2]: "

echo.
echo %BB0_MIDNIGHT%Choose text encoder format:%NC%
echo 1) Safetensors FP8 (recommended - smaller, compatible)
echo 2) Safetensors BF16 (full precision)
echo 3) GGUF (advanced - multiple quantization options)
set /p "encoder_choice=Choice [1-3]: "

REM Download based on choices
if "%model_choice%"=="1" call :download_klein_4b
if "%model_choice%"=="2" call :download_klein_9b
if "%model_choice%"=="3" (
    call :download_klein_4b
    echo.
    call :download_klein_9b
)

echo.
echo %BB0_ZENITH%=== Download Complete ===%NC%
echo.
echo %BB0_MIDNIGHT%Model Information:%NC%
echo   Flux 2 Klein 4B: ~13GB VRAM, Apache 2.0 (commercial use OK)
echo   Flux 2 Klein 9B: ~29GB VRAM, non-commercial license
echo.
echo %BB0_GLITCH%WARNING: Flux 2 uses a different VAE than Flux 1!%NC%
echo   %BB0_MIDNIGHT%Downloaded: flux2-vae.safetensors (Flux 2 specific)%NC%
echo.
echo %BB0_MIDNIGHT%To use with Deforum:%NC%
echo 1. Launch Forge: start-forge.bat
echo 2. Select Flux 2 Klein checkpoint from dropdown
echo 3. Ensure Flux 2 VAE is selected (not Flux 1 VAE)
echo 4. Use any Deforum render mode (3D, Flux + Interpolation, etc.)
echo.
echo %BB0_VOID%Sources:%NC%
echo   - Klein 4B BF16: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B
echo   - Klein 4B FP8: https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8
echo   - Klein 9B BF16: https://huggingface.co/black-forest-labs/FLUX.2-klein-9B
echo   - Klein 9B FP8: https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8
echo   - Text Encoders: Comfy-Org, jiangchengchengNLP, Qwen repos
echo   - Flux 2 VAE: https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b
echo   - Download guide: https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models

pause
exit /b 0

:download_klein_4b
echo.
echo %BB0_ZENITH%Downloading Flux 2 Klein 4B...%NC%

REM Download checkpoint based on precision choice
if "%precision_choice%"=="1" (
    echo %BB0_VOID%Downloading checkpoint FP8 (4.07 GB)...%NC%
    huggingface-cli download black-forest-labs/FLUX.2-klein-4b-fp8 flux-2-klein-4b-fp8.safetensors --local-dir "%FORGE_ROOT%\models\Stable-diffusion"
    set "checkpoint_file=flux-2-klein-4b-fp8.safetensors"
) else (
    echo %BB0_VOID%Downloading checkpoint BF16 (7.75 GB)...%NC%
    huggingface-cli download black-forest-labs/FLUX.2-klein-4B flux-2-klein-4b.safetensors --local-dir "%FORGE_ROOT%\models\Stable-diffusion"
    set "checkpoint_file=flux-2-klein-4b.safetensors"
)

REM Download text encoder based on choice
if "%encoder_choice%"=="1" (
    echo %BB0_VOID%Downloading text encoder FP8 (Qwen 3 4B)...%NC%
    huggingface-cli download jiangchengchengNLP/qwen3-4b-fp8-scaled qwen3_4b_fp8_scaled.safetensors --local-dir "%FORGE_ROOT%\models\text_encoder"
    set "encoder_file=qwen3_4b_fp8_scaled.safetensors"
) else if "%encoder_choice%"=="2" (
    echo %BB0_VOID%Downloading text encoder BF16 (Qwen 3 4B)...%NC%
    huggingface-cli download Comfy-Org/z_image_turbo split_files/text_encoders/qwen_3_4b.safetensors --local-dir "%FORGE_ROOT%\models\text_encoder"
    REM Move from nested directory to text_encoder root
    if exist "%FORGE_ROOT%\models\text_encoder\split_files\text_encoders\qwen_3_4b.safetensors" (
        move /y "%FORGE_ROOT%\models\text_encoder\split_files\text_encoders\qwen_3_4b.safetensors" "%FORGE_ROOT%\models\text_encoder\" >nul
        rmdir /s /q "%FORGE_ROOT%\models\text_encoder\split_files" 2>nul
    )
    set "encoder_file=qwen_3_4b.safetensors"
) else (
    echo %BB0_VOID%Downloading text encoder GGUF (Qwen 3 4B)...%NC%
    echo %BB0_MIDNIGHT%Available GGUF quantizations - download manually from:%NC%
    echo   https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/main
    set "encoder_file=(GGUF - manual download required)"
)

REM Download Flux 2 VAE
call :download_flux2_vae

echo %BB0_ZENITH%Flux 2 Klein 4B downloaded successfully!%NC%
echo   Checkpoint: models\Stable-diffusion\!checkpoint_file!
echo   Text Encoder: models\text_encoder\!encoder_file!
echo   VAE: models\VAE\flux2-vae.safetensors
goto :eof

:download_klein_9b
echo.
echo %BB0_ZENITH%Downloading Flux 2 Klein 9B...%NC%

REM Download checkpoint based on precision choice
if "%precision_choice%"=="1" (
    echo %BB0_VOID%Downloading checkpoint FP8 (~8 GB)...%NC%
    huggingface-cli download black-forest-labs/FLUX.2-klein-9b-fp8 flux-2-klein-9b-fp8.safetensors --local-dir "%FORGE_ROOT%\models\Stable-diffusion"
    set "checkpoint_file=flux-2-klein-9b-fp8.safetensors"
) else (
    echo %BB0_VOID%Downloading checkpoint BF16 (~15 GB)...%NC%
    huggingface-cli download black-forest-labs/FLUX.2-klein-9B flux-2-klein-9b.safetensors --local-dir "%FORGE_ROOT%\models\Stable-diffusion"
    set "checkpoint_file=flux-2-klein-9b.safetensors"
)

REM Download text encoder based on choice
if "%encoder_choice%"=="1" (
    echo %BB0_GLITCH%WARNING: FP8 text encoder not available for Qwen 3 8B%NC%
    echo %BB0_MIDNIGHT%Falling back to BF16...%NC%
    set "encoder_choice=2"
)

if "%encoder_choice%"=="2" (
    echo %BB0_VOID%Downloading text encoder BF16 (Qwen 3 8B)...%NC%
    huggingface-cli download Comfy-Org/vae-text-encorder-for-flux-klein-9b split_files/text_encoders/qwen_3_8b.safetensors --local-dir "%FORGE_ROOT%\models\text_encoder"
    REM Move from nested directory to text_encoder root
    if exist "%FORGE_ROOT%\models\text_encoder\split_files\text_encoders\qwen_3_8b.safetensors" (
        move /y "%FORGE_ROOT%\models\text_encoder\split_files\text_encoders\qwen_3_8b.safetensors" "%FORGE_ROOT%\models\text_encoder\" >nul
        rmdir /s /q "%FORGE_ROOT%\models\text_encoder\split_files" 2>nul
    )
    set "encoder_file=qwen_3_8b.safetensors"
) else (
    echo %BB0_VOID%Downloading text encoder GGUF (Qwen 3 8B)...%NC%
    echo %BB0_MIDNIGHT%Available GGUF quantizations - download manually from:%NC%
    echo   https://huggingface.co/Qwen/Qwen3-8B-GGUF/tree/main
    set "encoder_file=(GGUF - manual download required)"
)

REM Download Flux 2 VAE
call :download_flux2_vae

echo %BB0_ZENITH%Flux 2 Klein 9B downloaded successfully!%NC%
echo   Checkpoint: models\Stable-diffusion\!checkpoint_file!
echo   Text Encoder: models\text_encoder\!encoder_file!
echo   VAE: models\VAE\flux2-vae.safetensors
goto :eof

:download_flux2_vae
REM Download Flux 2 VAE (different from Flux 1!)
if not exist "%FORGE_ROOT%\models\VAE\flux2-vae.safetensors" (
    echo %BB0_VOID%Downloading Flux 2 VAE (different from Flux 1)...%NC%
    huggingface-cli download Comfy-Org/vae-text-encorder-for-flux-klein-9b split_files/vae/flux2-vae.safetensors --local-dir "%FORGE_ROOT%\models\VAE"

    REM Move from nested directory to VAE root
    if exist "%FORGE_ROOT%\models\VAE\split_files\vae\flux2-vae.safetensors" (
        move /y "%FORGE_ROOT%\models\VAE\split_files\vae\flux2-vae.safetensors" "%FORGE_ROOT%\models\VAE\" >nul
        rmdir /s /q "%FORGE_ROOT%\models\VAE\split_files" 2>nul
    )

    echo %BB0_MIDNIGHT%Checksum verification (manual check recommended)%NC%
) else (
    echo %BB0_ZENITH%Flux 2 VAE already exists, skipping...%NC%
)
goto :eof
