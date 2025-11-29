@echo off
REM download-all-models.bat - Download all Deforum models (Windows)
REM This ensures a complete installation of Flux, Wan, Qwen, and other Deforum models

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set FORGE_ROOT=%SCRIPT_DIR%..\..\\

echo ========================================
echo Deforum Model Download Script
echo ========================================
echo.

cd /d "%FORGE_ROOT%"

echo Working from Forge root: %CD%
echo.

REM Create model directories
echo Creating model directories...
if not exist "models\Stable-diffusion\Flux" mkdir "models\Stable-diffusion\Flux"
if not exist "models\VAE" mkdir "models\VAE"
if not exist "models\text_encoder" mkdir "models\text_encoder"
if not exist "models\ControlNet" mkdir "models\ControlNet"
if not exist "models\Deforum\film_interpolation" mkdir "models\Deforum\film_interpolation"
if not exist "models\Deforum\wan" mkdir "models\Deforum\wan"
if not exist "models\Deforum\qwen" mkdir "models\Deforum\qwen"
echo Done: Directories created
echo.

REM Check for HuggingFace CLI
where huggingface-cli >nul 2>&1
if errorlevel 1 (
    echo ERROR: huggingface-cli not found!
    echo.
    echo Please install it with:
    echo   pip install huggingface-hub
    echo.
    exit /b 1
)

REM =====================================
REM 1. Flux.1 Dev BNB NF4 v2
REM =====================================
echo === Flux.1 Dev Checkpoint ^(Quantized^) ===
if exist "models\Stable-diffusion\Flux\flux1-dev-bnb-nf4-v2.safetensors" (
    echo Done: Flux checkpoint already exists
) else (
    echo Downloading Flux.1 Dev BNB NF4 v2 ^(~10GB quantized^)...
    echo Note: This is a 4-bit quantized version optimized for lower VRAM usage
    huggingface-cli download lllyasviel/flux1-dev-bnb-nf4 flux1-dev-bnb-nf4-v2.safetensors --local-dir models\Stable-diffusion\Flux --resume-download
    echo Done: Flux checkpoint downloaded
)
echo.

REM =====================================
REM 2. Flux VAE and Text Encoders
REM =====================================
echo === Flux VAE and Text Encoders ===

REM VAE
if exist "models\VAE\ae.safetensors" (
    echo Done: VAE already exists
) else (
    echo Downloading Flux VAE ^(ae.safetensors^)...
    huggingface-cli download black-forest-labs/FLUX.1-dev ae.safetensors --local-dir models\VAE --resume-download
    echo Done: VAE downloaded
)

REM CLIP-L
if exist "models\VAE\clip_l.safetensors" (
    echo Done: CLIP-L already exists
) else (
    echo Downloading CLIP-L text encoder...
    huggingface-cli download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models\VAE --local-dir-use-symlinks False --resume-download
    echo Done: CLIP-L downloaded
)

REM T5-XXL
if exist "models\VAE\t5xxl_fp16.safetensors" (
    echo Done: T5-XXL already exists
) else (
    echo Downloading T5-XXL text encoder ^(fp16^)...
    huggingface-cli download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir models\VAE --local-dir-use-symlinks False --resume-download
    echo Done: T5-XXL downloaded
)
echo.

REM =====================================
REM 3. Flux ControlNet V2
REM =====================================
echo === Flux ControlNet V2 ===
echo Choose which Flux ControlNet models to download:
echo   1^) Canny ^(Edge detection, ~3.5GB^)
echo   2^) Depth ^(Depth conditioning, ~3.5GB^)
echo   3^) Both Canny and Depth
echo   4^) Skip ControlNet models
set /p controlnet_choice="Enter choice [1-4]: "

if "%controlnet_choice%"=="1" goto :canny_only
if "%controlnet_choice%"=="2" goto :depth_only
if "%controlnet_choice%"=="3" goto :both_cn
if "%controlnet_choice%"=="4" goto :skip_cn
goto :skip_cn

:canny_only
:both_cn
echo Downloading Flux ControlNet Canny...
huggingface-cli download InstantX/FLUX.1-dev-Controlnet-Canny --local-dir models\ControlNet\FLUX.1-dev-Controlnet-Canny --resume-download
echo Done: ControlNet Canny downloaded
if "%controlnet_choice%"=="1" goto :film_section

:depth_only
echo Downloading Flux ControlNet Depth...
huggingface-cli download Shakker-Labs/FLUX.1-dev-ControlNet-Depth --local-dir models\ControlNet\FLUX.1-dev-ControlNet-Depth --resume-download
echo Done: ControlNet Depth downloaded
goto :film_section

:skip_cn
echo Skipping ControlNet models

:film_section
echo.

REM =====================================
REM 4. FILM Interpolation Model
REM =====================================
echo === FILM Interpolation Model ===
if exist "models\Deforum\film_interpolation\film_net_fp16.pt" (
    echo Done: FILM model already exists
) else (
    echo Downloading FILM model...
    echo Note: Using python download since FILM is from GitHub releases
    python -c "from torch.hub import download_url_to_file; download_url_to_file('https://github.com/hithereai/frame-interpolation-pytorch/releases/download/film_net_fp16.pt/film_net_fp16.pt', 'models/Deforum/film_interpolation/film_net_fp16.pt', progress=True)"
    if errorlevel 1 (
        echo ERROR: Failed to download FILM model
    ) else (
        echo Done: FILM model downloaded
    )
)
echo.

REM =====================================
REM 5. Wan Models
REM =====================================
echo === Wan AI Video Models ===
echo Choose which Wan models to download:
echo   1^) FLF2V-14B ^(Required for FLF2V interpolation, ~14GB^)
echo   2^) TI2V-5B ^(Recommended for T2V/I2V, 24GB VRAM, ~5GB^)
echo   3^) TI2V-A14B ^(Highest quality MoE, 32GB+ VRAM, ~14GB^)
echo   4^) All models ^(Downloads all 3^)
echo   5^) Skip Wan models
set /p wan_choice="Enter choice [1-5]: "

if "%wan_choice%"=="1" goto :flf2v_only
if "%wan_choice%"=="2" goto :ti2v5b_only
if "%wan_choice%"=="3" goto :ti2va14b_only
if "%wan_choice%"=="4" goto :all_wan
if "%wan_choice%"=="5" goto :skip_wan
goto :skip_wan

:flf2v_only
:all_wan
echo Downloading Wan2.1-FLF2V-14B...
huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers --local-dir models\Deforum\wan\Wan2.1-FLF2V-14B --resume-download
echo Done: FLF2V-14B downloaded
if "%wan_choice%"=="1" goto :qwen_section

:ti2v5b_only
echo Downloading Wan2.2-TI2V-5B...
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir models\Deforum\wan\Wan2.2-TI2V-5B --resume-download
echo Done: TI2V-5B downloaded
if "%wan_choice%"=="2" goto :qwen_section

:ti2va14b_only
echo Downloading Wan2.2-TI2V-A14B...
huggingface-cli download Wan-AI/Wan2.2-TI2V-A14B-Diffusers --local-dir models\Deforum\wan\Wan2.2-TI2V-A14B --resume-download
echo Done: TI2V-A14B downloaded
goto :qwen_section

:skip_wan
echo Skipping Wan models

:qwen_section
echo.

REM =====================================
REM 6. Qwen Models
REM =====================================
echo === Qwen Prompt Enhancement Models ===
echo Choose which Qwen model to download:
echo   1^) Qwen2.5-3B-Instruct ^(Recommended for low VRAM, ~3GB^)
echo   2^) Qwen2.5-7B-Instruct ^(Better quality, ~7GB^)
echo   3^) Qwen2.5-14B-Instruct ^(Best quality, 32GB+ VRAM, ~14GB^)
echo   4^) All models ^(Downloads all 3^)
echo   5^) Skip Qwen models
set /p qwen_choice="Enter choice [1-5]: "

if "%qwen_choice%"=="1" goto :qwen3b_only
if "%qwen_choice%"=="2" goto :qwen7b_only
if "%qwen_choice%"=="3" goto :qwen14b_only
if "%qwen_choice%"=="4" goto :all_qwen
if "%qwen_choice%"=="5" goto :skip_qwen
goto :skip_qwen

:qwen3b_only
:all_qwen
echo Downloading Qwen2.5-3B-Instruct...
huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models\Deforum\qwen\Qwen2.5-3B-Instruct --resume-download
echo Done: Qwen2.5-3B-Instruct downloaded
if "%qwen_choice%"=="1" goto :summary

:qwen7b_only
echo Downloading Qwen2.5-7B-Instruct...
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models\Deforum\qwen\Qwen2.5-7B-Instruct --resume-download
echo Done: Qwen2.5-7B-Instruct downloaded
if "%qwen_choice%"=="2" goto :summary

:qwen14b_only
echo Downloading Qwen2.5-14B-Instruct...
huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir models\Deforum\qwen\Qwen2.5-14B-Instruct --resume-download
echo Done: Qwen2.5-14B-Instruct downloaded
goto :summary

:skip_qwen
echo Skipping Qwen models

:summary
echo.
echo ========================================
echo Model Download Complete!
echo ========================================
echo.
echo Downloaded models are located in:
echo   - Flux: models\Stable-diffusion\Flux\
echo   - VAE ^& Text Encoders: models\VAE\
echo   - ControlNet: models\ControlNet\
echo   - FILM: models\Deforum\film_interpolation\
echo   - Wan AI Video: models\Deforum\wan\
echo   - Qwen Prompts: models\Deforum\qwen\
echo.
echo Note: Depth models ^(Depth-Anything V2^) will be auto-downloaded
echo on first use. Gifski and Real-ESRGAN binaries are also auto-downloaded.
echo.
echo You can now use Deforum with Flux + Interpolation mode!
echo.
