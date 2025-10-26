@echo off
REM download-all-models.bat
REM Downloads all required models for Deforum extension at once
REM This ensures a complete installation and tests all download hooks

setlocal enabledelayedexpansion

echo ========================================
echo Deforum Model Download Script
echo ========================================
echo.

REM Get script directory (extension root)
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%"

REM Navigate to Forge root (two levels up)
cd ..\..
set FORGE_ROOT=%CD%
echo Working from Forge root: %FORGE_ROOT%
echo.

REM Create model directories
echo Creating model directories...
if not exist "models\Stable-diffusion\Flux" mkdir "models\Stable-diffusion\Flux"
if not exist "models\VAE" mkdir "models\VAE"
if not exist "models\ControlNet" mkdir "models\ControlNet"
if not exist "models\Deforum\film_interpolation" mkdir "models\Deforum\film_interpolation"
if not exist "models\Deforum\wan" mkdir "models\Deforum\wan"
if not exist "models\Deforum\qwen" mkdir "models\Deforum\qwen"
echo [OK] Directories created
echo.

REM Check for HuggingFace CLI
where huggingface-cli >nul 2>nul
if errorlevel 1 (
    echo [ERROR] huggingface-cli not found!
    echo.
    echo Please install it with:
    echo   pip install huggingface-hub
    echo.
    exit /b 1
)

REM =====================================
REM 1. Flux.1 Dev BNB NF4 v2 (MOST IMPORTANT)
REM =====================================
echo === Flux.1 Dev Checkpoint (Quantized) ===
set FLUX_PATH=models\Stable-diffusion\Flux\flux1-dev-bnb-nf4-v2.safetensors
if exist "%FLUX_PATH%" (
    echo [OK] Flux checkpoint already exists
) else (
    echo Downloading Flux.1 Dev BNB NF4 v2 (~10GB quantized)...
    echo Note: This is a 4-bit quantized version optimized for lower VRAM usage
    huggingface-cli download lllyasviel/flux1-dev-bnb-nf4 flux1-dev-bnb-nf4-v2.safetensors --local-dir models\Stable-diffusion\Flux --resume-download
    if errorlevel 1 (
        echo [ERROR] Failed to download Flux checkpoint
    ) else (
        echo [OK] Flux checkpoint downloaded
    )
)
echo.

REM =====================================
REM 2. Flux VAE and Text Encoders
REM =====================================
echo === Flux VAE and Text Encoders ===

REM VAE (ae.safetensors)
set VAE_PATH=models\VAE\ae.safetensors
if exist "%VAE_PATH%" (
    echo [OK] VAE already exists
) else (
    echo Downloading Flux VAE (ae.safetensors)...
    huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors --local-dir models\VAE --resume-download
    if errorlevel 1 (
        echo [ERROR] Failed to download VAE
    ) else (
        echo [OK] VAE downloaded
    )
)

REM CLIP-L text encoder
set CLIP_L_PATH=models\VAE\clip_l.safetensors
if exist "%CLIP_L_PATH%" (
    echo [OK] CLIP-L already exists
) else (
    echo Downloading CLIP-L text encoder...
    huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models\VAE --resume-download
    if errorlevel 1 (
        echo [ERROR] Failed to download CLIP-L
    ) else (
        echo [OK] CLIP-L downloaded
    )
)

REM T5-XXL text encoder
set T5_PATH=models\VAE\t5xxl_fp16.safetensors
if exist "%T5_PATH%" (
    echo [OK] T5-XXL already exists
) else (
    echo Downloading T5-XXL text encoder (fp16)...
    huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir models\VAE --resume-download
    if errorlevel 1 (
        echo [ERROR] Failed to download T5-XXL
    ) else (
        echo [OK] T5-XXL downloaded
    )
)
echo.

REM =====================================
REM 3. Flux ControlNet V2
REM =====================================
echo === Flux ControlNet V2 ===
echo Choose which Flux ControlNet models to download:
echo   1) Canny (Edge detection, ~3.5GB)
echo   2) Depth (Depth conditioning, ~3.5GB)
echo   3) Both Canny and Depth
echo   4) Skip ControlNet models
echo.
set /p controlnet_choice="Enter choice [1-4]: "

if "%controlnet_choice%"=="1" goto :download_cn_canny
if "%controlnet_choice%"=="2" goto :download_cn_depth
if "%controlnet_choice%"=="3" goto :download_all_cn
if "%controlnet_choice%"=="4" goto :skip_controlnet
goto :skip_controlnet

:download_all_cn
call :download_cn_canny_func
call :download_cn_depth_func
goto :skip_controlnet

:download_cn_canny
call :download_cn_canny_func
goto :skip_controlnet

:download_cn_depth
call :download_cn_depth_func
goto :skip_controlnet

:download_cn_canny_func
echo Downloading Flux ControlNet Canny...
huggingface-cli download InstantX/FLUX.1-dev-Controlnet-Canny --local-dir models\ControlNet\FLUX.1-dev-Controlnet-Canny --resume-download
echo [OK] ControlNet Canny downloaded
exit /b 0

:download_cn_depth_func
echo Downloading Flux ControlNet Depth...
huggingface-cli download Shakker-Labs/FLUX.1-dev-ControlNet-Depth --local-dir models\ControlNet\FLUX.1-dev-ControlNet-Depth --resume-download
echo [OK] ControlNet Depth downloaded
exit /b 0

:skip_controlnet
if "%controlnet_choice%"=="4" echo Skipping ControlNet models
echo.

REM =====================================
REM 4. FILM Interpolation Model
REM =====================================
echo === FILM Interpolation Model ===
set FILM_PATH=models\Deforum\film_interpolation\film_net_fp16.pt
if exist "%FILM_PATH%" (
    echo [OK] FILM model already exists
) else (
    echo Downloading FILM model ^(film_net_fp16.pt^)...
    python -c "from torch.hub import download_url_to_file; download_url_to_file('https://github.com/hithereai/frame-interpolation-pytorch/releases/download/film_net_fp16.pt/film_net_fp16.pt', '%FILM_PATH%', progress=True)"
    if errorlevel 1 (
        echo [ERROR] Failed to download FILM model
    ) else (
        echo [OK] FILM model downloaded successfully
    )
)
echo.

REM =====================================
REM 5. Wan Models (HuggingFace)
REM =====================================
echo === Wan AI Video Models ===
echo Choose which Wan models to download:
echo   1) FLF2V-14B (Required for FLF2V interpolation, ~14GB)
echo   2) TI2V-5B (Recommended for T2V/I2V, 24GB VRAM, ~5GB)
echo   3) TI2V-A14B (Highest quality MoE, 32GB+ VRAM, ~14GB)
echo   4) All models (Downloads all 3)
echo   5) Skip Wan models
echo.
set /p wan_choice="Enter choice [1-5]: "

if "%wan_choice%"=="1" goto :download_flf2v
if "%wan_choice%"=="2" goto :download_ti2v_5b
if "%wan_choice%"=="3" goto :download_ti2v_14b
if "%wan_choice%"=="4" goto :download_all_wan
if "%wan_choice%"=="5" goto :skip_wan
goto :skip_wan

:download_all_wan
call :download_flf2v_func
call :download_ti2v_5b_func
call :download_ti2v_14b_func
goto :skip_wan

:download_flf2v
call :download_flf2v_func
goto :skip_wan

:download_ti2v_5b
call :download_ti2v_5b_func
goto :skip_wan

:download_ti2v_14b
call :download_ti2v_14b_func
goto :skip_wan

:download_flf2v_func
echo Downloading Wan2.1-FLF2V-14B...
huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers --local-dir models\Deforum\wan\Wan2.1-FLF2V-14B --resume-download
echo [OK] FLF2V-14B downloaded
exit /b 0

:download_ti2v_5b_func
echo Downloading Wan2.2-TI2V-5B...
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir models\Deforum\wan\Wan2.2-TI2V-5B --resume-download
echo [OK] TI2V-5B downloaded
exit /b 0

:download_ti2v_14b_func
echo Downloading Wan2.2-TI2V-A14B...
huggingface-cli download Wan-AI/Wan2.2-TI2V-A14B-Diffusers --local-dir models\Deforum\wan\Wan2.2-TI2V-A14B --resume-download
echo [OK] TI2V-A14B downloaded
exit /b 0

:skip_wan
if "%wan_choice%"=="5" echo Skipping Wan models
echo.

REM =====================================
REM 6. Qwen AI Prompt Enhancement Models
REM =====================================
echo === Qwen Prompt Enhancement Models ===
echo Choose which Qwen model to download (for AI prompt enhancement):
echo   1) Qwen2.5-3B-Instruct (Recommended for low VRAM, ~3GB)
echo   2) Qwen2.5-7B-Instruct (Better quality, ~7GB)
echo   3) Qwen2.5-14B-Instruct (Best quality, 32GB+ VRAM, ~14GB)
echo   4) All models (Downloads all 3)
echo   5) Skip Qwen models
echo.
set /p qwen_choice="Enter choice [1-5]: "

if "%qwen_choice%"=="1" goto :download_qwen_3b
if "%qwen_choice%"=="2" goto :download_qwen_7b
if "%qwen_choice%"=="3" goto :download_qwen_14b
if "%qwen_choice%"=="4" goto :download_all_qwen
if "%qwen_choice%"=="5" goto :skip_qwen
goto :skip_qwen

:download_all_qwen
call :download_qwen_3b_func
call :download_qwen_7b_func
call :download_qwen_14b_func
goto :skip_qwen

:download_qwen_3b
call :download_qwen_3b_func
goto :skip_qwen

:download_qwen_7b
call :download_qwen_7b_func
goto :skip_qwen

:download_qwen_14b
call :download_qwen_14b_func
goto :skip_qwen

:download_qwen_3b_func
echo Downloading Qwen2.5-3B-Instruct...
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models\Deforum\qwen\Qwen2.5-3B-Instruct --resume-download
echo [OK] Qwen2.5-3B-Instruct downloaded
exit /b 0

:download_qwen_7b_func
echo Downloading Qwen2.5-7B-Instruct...
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models\Deforum\qwen\Qwen2.5-7B-Instruct --resume-download
echo [OK] Qwen2.5-7B-Instruct downloaded
exit /b 0

:download_qwen_14b_func
echo Downloading Qwen2.5-14B-Instruct...
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir models\Deforum\qwen\Qwen2.5-14B-Instruct --resume-download
echo [OK] Qwen2.5-14B-Instruct downloaded
exit /b 0

:skip_qwen
if "%qwen_choice%"=="5" echo Skipping Qwen models
echo.

REM =====================================
REM Summary
REM =====================================
echo.
echo ========================================
echo [OK] Model Download Complete!
echo ========================================
echo.
echo Downloaded models are located in:
echo   * Flux: models\Stable-diffusion\Flux\
echo   * VAE ^& Text Encoders: models\VAE\
echo   * ControlNet: models\ControlNet\
echo   * FILM: models\Deforum\film_interpolation\
echo   * Wan AI Video: models\Deforum\wan\
echo   * Qwen Prompts: models\Deforum\qwen\
echo.
echo Note: Depth models (Depth-Anything V2) will be auto-downloaded
echo on first use. Gifski and Real-ESRGAN binaries are also auto-downloaded.
echo.
echo [OK] You can now use Deforum with Flux + Interpolation mode!
echo.

endlocal
