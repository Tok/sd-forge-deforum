#!/bin/bash

# download-all-models.sh
# Downloads all required models for Deforum extension at once
# This ensures a complete installation and tests all download hooks

set -e  # Exit on error

echo "========================================"
echo "Deforum Model Download Script"
echo "========================================"
echo ""

# BB0 Slopcore colors (see docs/SLOPCORE.md)
BB0_ZENITH='\033[38;2;23;167;254m'    # #17A7FE - Cyan (info, success)
BB0_GLITCH='\033[38;2;255;20;147m'    # #FF1493 - Neon pink (warning, error)
BB0_VOID='\033[38;2;86;6;255m'        # #5606FF - Deep purple (emphasis)
BB0_MIDNIGHT='\033[38;2;55;87;255m'   # #3757FF - Mid blue (secondary)
NC='\033[0m' # No Color

# Get script directory (extension root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTENSION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# Navigate to Forge root (two levels up)
EXTENSION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FORGE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$FORGE_ROOT"

echo -e "${BB0_MIDNIGHT}Working from Forge root: $FORGE_ROOT${NC}"
echo ""

# Create model directories
echo -e "${BB0_GLITCH}Creating model directories...${NC}"
mkdir -p models/Stable-diffusion/Flux
mkdir -p models/VAE
mkdir -p models/text_encoder
mkdir -p models/ControlNet
mkdir -p models/Deforum/film_interpolation
mkdir -p models/Deforum/wan
mkdir -p models/Deforum/qwen
echo -e "${BB0_ZENITH}✓ Directories created${NC}"
echo ""

# Check for HuggingFace CLI
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${BB0_GLITCH}❌ huggingface-cli not found!${NC}"
    echo ""
    echo "Please install it with:"
    echo "  pip install huggingface-hub"
    echo ""
    exit 1
fi

# Function to download with progress
download_file() {
    local url=$1
    local dest=$2
    local name=$3

    echo -e "${BB0_GLITCH}Downloading $name...${NC}"
    python -c "
import sys
from torch.hub import download_url_to_file
download_url_to_file('$url', '$dest', progress=True)
"
    if [ $? -eq 0 ]; then
        echo -e "${BB0_ZENITH}✓ $name downloaded successfully${NC}"
    else
        echo -e "${BB0_GLITCH}❌ Failed to download $name${NC}"
        return 1
    fi
}

# =====================================
# 1. Flux.1 Dev BNB NF4 v2 (MOST IMPORTANT)
# =====================================
echo -e "${BB0_MIDNIGHT}=== Flux.1 Dev Checkpoint (Quantized) ===${NC}"
FLUX_PATH="models/Stable-diffusion/Flux/flux1-dev-bnb-nf4-v2.safetensors"
if [ -f "$FLUX_PATH" ]; then
    echo -e "${BB0_ZENITH}✓ Flux checkpoint already exists${NC}"
else
    echo -e "${BB0_GLITCH}Downloading Flux.1 Dev BNB NF4 v2 (~10GB quantized)...${NC}"
    echo "Note: This is a 4-bit quantized version optimized for lower VRAM usage"
    huggingface-cli download lllyasviel/flux1-dev-bnb-nf4 \
        flux1-dev-bnb-nf4-v2.safetensors \
        --local-dir models/Stable-diffusion/Flux \
        --resume-download
    echo -e "${BB0_ZENITH}✓ Flux checkpoint downloaded${NC}"
fi
echo ""

# =====================================
# 2. Shared VAE (ae.safetensors)
# =====================================
echo -e "${BB0_MIDNIGHT}=== Shared FLUX VAE ===${NC}"
echo "This VAE is shared by Flux, Lumina, and Z-Image"
echo ""

VAE_PATH="models/VAE/ae.safetensors"
if [ -f "$VAE_PATH" ]; then
    echo -e "${BB0_ZENITH}✓ FLUX VAE already exists (shared)${NC}"
else
    echo -e "${BB0_GLITCH}Downloading FLUX VAE (ae.safetensors, ~320MB)...${NC}"
    huggingface-cli download black-forest-labs/FLUX.1-dev \
        ae.safetensors \
        --local-dir models/VAE \
        --resume-download
    echo -e "${BB0_ZENITH}✓ FLUX VAE downloaded${NC}"
fi
echo ""

# =====================================
# 3. Flux Text Encoders
# =====================================
echo -e "${BB0_MIDNIGHT}=== Flux Text Encoders ===${NC}"
echo "Required for Flux models"
echo ""

# CLIP-L text encoder
CLIP_L_PATH="models/text_encoder/clip_l.safetensors"
if [ -f "$CLIP_L_PATH" ]; then
    echo -e "${BB0_ZENITH}✓ CLIP-L already exists${NC}"
else
    echo -e "${BB0_GLITCH}Downloading CLIP-L text encoder (~235MB)...${NC}"
    huggingface-cli download comfyanonymous/flux_text_encoders \
        clip_l.safetensors \
        --local-dir models/text_encoder \
        --local-dir-use-symlinks False \
        --resume-download
    echo -e "${BB0_ZENITH}✓ CLIP-L downloaded${NC}"
fi

# T5-XXL text encoder
T5_PATH="models/text_encoder/t5xxl_fp16.safetensors"
if [ -f "$T5_PATH" ]; then
    echo -e "${BB0_ZENITH}✓ T5-XXL already exists${NC}"
else
    echo -e "${BB0_GLITCH}Downloading T5-XXL text encoder (fp16, ~9.2GB)...${NC}"
    huggingface-cli download comfyanonymous/flux_text_encoders \
        t5xxl_fp16.safetensors \
        --local-dir models/text_encoder \
        --local-dir-use-symlinks False \
        --resume-download
    echo -e "${BB0_ZENITH}✓ T5-XXL downloaded${NC}"
fi
echo ""

# =====================================
# 4. Flux ControlNet V2
# =====================================
echo -e "${BB0_MIDNIGHT}=== Flux ControlNet V2 ===${NC}"
echo "Choose which Flux ControlNet models to download:"
echo "  1) Canny (Edge detection, ~3.5GB)"
echo "  2) Depth (Depth conditioning, ~3.5GB)"
echo "  3) Both Canny and Depth"
echo "  4) Skip ControlNet models"
read -p "Enter choice [1-4]: " controlnet_choice

case $controlnet_choice in
    1|3)
        echo -e "${BB0_GLITCH}Downloading Flux ControlNet Canny...${NC}"
        huggingface-cli download InstantX/FLUX.1-dev-Controlnet-Canny \
            --local-dir models/ControlNet/FLUX.1-dev-Controlnet-Canny \
            --resume-download
        echo -e "${BB0_ZENITH}✓ ControlNet Canny downloaded${NC}"
        ;&  # Fall through if choice was 3
esac

case $controlnet_choice in
    2|3)
        echo -e "${BB0_GLITCH}Downloading Flux ControlNet Depth...${NC}"
        huggingface-cli download Shakker-Labs/FLUX.1-dev-ControlNet-Depth \
            --local-dir models/ControlNet/FLUX.1-dev-ControlNet-Depth \
            --resume-download
        echo -e "${BB0_ZENITH}✓ ControlNet Depth downloaded${NC}"
        ;;
    4)
        echo -e "${BB0_GLITCH}Skipping ControlNet models${NC}"
        ;;
esac
echo ""

# =====================================
# 5. FILM Interpolation Model
# =====================================
echo -e "${BB0_MIDNIGHT}=== FILM Interpolation Model ===${NC}"
FILM_PATH="models/Deforum/film_interpolation/film_net_fp16.pt"
if [ -f "$FILM_PATH" ]; then
    echo -e "${BB0_ZENITH}✓ FILM model already exists${NC}"
else
    download_file \
        "https://github.com/hithereai/frame-interpolation-pytorch/releases/download/film_net_fp16.pt/film_net_fp16.pt" \
        "$FILM_PATH" \
        "FILM model (film_net_fp16.pt)"
fi
echo ""

# =====================================
# 6. Wan Models (HuggingFace)
# =====================================
echo -e "${BB0_MIDNIGHT}=== Wan AI Video Models ===${NC}"
echo "Choose which Wan models to download:"
echo "  1) FLF2V-14B (Required for FLF2V interpolation, ~14GB)"
echo "  2) TI2V-5B (Recommended for T2V/I2V, 24GB VRAM, ~5GB)"
echo "  3) TI2V-A14B (Highest quality MoE, 32GB+ VRAM, ~14GB)"
echo "  4) All models (Downloads all 3)"
echo "  5) Skip Wan models"
read -p "Enter choice [1-5]: " wan_choice

case $wan_choice in
    1|4)
        echo -e "${BB0_GLITCH}Downloading Wan2.1-FLF2V-14B...${NC}"
        huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers \
            --local-dir models/Deforum/wan/Wan2.1-FLF2V-14B \
            --resume-download
        echo -e "${BB0_ZENITH}✓ FLF2V-14B downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $wan_choice in
    2|4)
        echo -e "${BB0_GLITCH}Downloading Wan2.2-TI2V-5B...${NC}"
        huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
            --local-dir models/Deforum/wan/Wan2.2-TI2V-5B \
            --resume-download
        echo -e "${BB0_ZENITH}✓ TI2V-5B downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $wan_choice in
    3|4)
        echo -e "${BB0_GLITCH}Downloading Wan2.2-TI2V-A14B...${NC}"
        huggingface-cli download Wan-AI/Wan2.2-TI2V-A14B-Diffusers \
            --local-dir models/Deforum/wan/Wan2.2-TI2V-A14B \
            --resume-download
        echo -e "${BB0_ZENITH}✓ TI2V-A14B downloaded${NC}"
        ;;
    5)
        echo -e "${BB0_GLITCH}Skipping Wan models${NC}"
        ;;
esac
echo ""

# =====================================
# 7. Qwen AI Prompt Enhancement Models
# =====================================
echo -e "${BB0_MIDNIGHT}=== Qwen Prompt Enhancement Models ===${NC}"
echo "Choose which Qwen model to download (for AI prompt enhancement):"
echo "  1) Qwen2.5-3B-Instruct (Recommended for low VRAM, ~3GB)"
echo "  2) Qwen2.5-7B-Instruct (Better quality, ~7GB)"
echo "  3) Qwen2.5-14B-Instruct (Best quality, 32GB+ VRAM, ~14GB)"
echo "  4) All models (Downloads all 3)"
echo "  5) Skip Qwen models"
read -p "Enter choice [1-5]: " qwen_choice

case $qwen_choice in
    1|4)
        echo -e "${BB0_GLITCH}Downloading Qwen2.5-3B-Instruct...${NC}"
        huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
            --local-dir models/Deforum/qwen/Qwen2.5-3B-Instruct \
            --resume-download
        echo -e "${BB0_ZENITH}✓ Qwen2.5-3B-Instruct downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $qwen_choice in
    2|4)
        echo -e "${BB0_GLITCH}Downloading Qwen2.5-7B-Instruct...${NC}"
        huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
            --local-dir models/Deforum/qwen/Qwen2.5-7B-Instruct \
            --resume-download
        echo -e "${BB0_ZENITH}✓ Qwen2.5-7B-Instruct downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $qwen_choice in
    3|4)
        echo -e "${BB0_GLITCH}Downloading Qwen2.5-14B-Instruct...${NC}"
        huggingface-cli download Qwen/Qwen2.5-14B-Instruct \
            --local-dir models/Deforum/qwen/Qwen2.5-14B-Instruct \
            --resume-download
        echo -e "${BB0_ZENITH}✓ Qwen2.5-14B-Instruct downloaded${NC}"
        ;;
    5)
        echo -e "${BB0_GLITCH}Skipping Qwen models${NC}"
        ;;
esac
echo ""

# =====================================
# 8. Depth Anything V3 (Advanced Depth Estimation)
# =====================================
echo -e "${BB0_MIDNIGHT}=== Depth Anything V3 (DA3) ===${NC}"
echo "DA3 provides enhanced depth estimation for 3D depth warping"
echo ""
echo -e "${BB0_ZENITH}Standard Models (for 3D depth warp modes):${NC}"
echo "  1) DA3MONO-LARGE (Recommended, single-view depth, ~350MB)"
echo ""
echo -e "${BB0_GLITCH}GIANT Models (for DA3-3DGS interpolation only, 24GB+ VRAM required):${NC}"
echo "  2) DA3-GIANT (3DGS capable, 1.15B params, ~3GB)"
echo "  3) DA3NESTED-GIANT-LARGE (Recommended for 3DGS, 1.40B params, ~4GB)"
echo ""
echo "  4) All standard + GIANT models"
echo "  5) Skip DA3 models (package auto-installs but models won't cache)"
echo ""
echo -e "${BB0_VOID}NOTE: Only DA3MONO-LARGE exists for mono depth (no Small/Base variants)${NC}"
echo -e "${BB0_VOID}      GIANT models ONLY work with DA3-3DGS FLF2V interpolation mode${NC}"
read -p "Enter choice [1-5]: " da3_choice

# Create DA3 directory
mkdir -p models/Deforum/depth-anything-v3

case $da3_choice in
    1|4)
        echo -e "${BB0_GLITCH}Downloading DA3MONO-LARGE (single-view depth)...${NC}"
        huggingface-cli download depth-anything/DA3MONO-LARGE \
            --local-dir models/Deforum/depth-anything-v3/DA3MONO-LARGE \
            --resume-download
        echo -e "${BB0_ZENITH}✓ DA3MONO-LARGE downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $da3_choice in
    2|4)
        echo -e "${BB0_GLITCH}Downloading DA3-GIANT (3DGS capable, 1.15B params, ~3GB)...${NC}"
        echo -e "${BB0_VOID}⚠️  Requires 24GB+ VRAM for DA3-3DGS interpolation mode${NC}"
        huggingface-cli download depth-anything/DA3-GIANT \
            --local-dir models/Deforum/depth-anything-v3/DA3-GIANT \
            --resume-download
        echo -e "${BB0_ZENITH}✓ DA3-GIANT downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $da3_choice in
    3|4)
        echo -e "${BB0_GLITCH}Downloading DA3NESTED-GIANT-LARGE (Recommended for 3DGS, 1.40B params, ~4GB)...${NC}"
        echo -e "${BB0_VOID}⚠️  Requires 24GB+ VRAM for DA3-3DGS interpolation mode${NC}"
        huggingface-cli download depth-anything/DA3NESTED-GIANT-LARGE \
            --local-dir models/Deforum/depth-anything-v3/DA3NESTED-GIANT-LARGE \
            --resume-download
        echo -e "${BB0_ZENITH}✓ DA3NESTED-GIANT-LARGE downloaded${NC}"
        ;;
    5)
        echo -e "${BB0_GLITCH}Skipping DA3 models (package auto-installs, models will download on first use)${NC}"
        ;;
esac
echo ""

# =====================================
# 9. Lumina 2.0 (Anime-Optimized)
# =====================================
echo -e "${BB0_MIDNIGHT}=== Lumina 2.0 (Anime-Optimized Fine-Tune) ===${NC}"
echo "Lumina 2.0 is a 2B parameter model with 1024x1024 native resolution"
echo "This is the neta-art anime-optimized fine-tune of Alpha-VLLM/Lumina-Image-2.0"
echo ""
echo -e "${BB0_GLITCH}Dependencies: Shares FLUX VAE (models/VAE/ae.safetensors)${NC}"
echo ""
read -p "Download Lumina 2.0? [y/N]: " download_lumina
if [[ $download_lumina =~ ^[Yy]$ ]]; then
    # Create Lumina subdirectories
    mkdir -p models/Stable-diffusion/Lumina/Unet
    mkdir -p "models/Stable-diffusion/Lumina/Text Encoder"
    mkdir -p models/Stable-diffusion/Lumina/VAE

    # Download UNet
    LUMINA_UNET_PATH="models/Stable-diffusion/Lumina/Unet/neta-lumina-v1.0.safetensors"
    if [ -f "$LUMINA_UNET_PATH" ]; then
        echo -e "${BB0_ZENITH}✓ Lumina UNet already exists${NC}"
    else
        echo -e "${BB0_GLITCH}Downloading Lumina UNet (~4.9GB)...${NC}"
        huggingface-cli download neta-art/Neta-Lumina \
            neta-lumina-v1.0.safetensors \
            --local-dir models/Stable-diffusion/Lumina/Unet \
            --resume-download
        echo -e "${BB0_ZENITH}✓ Lumina UNet downloaded${NC}"
    fi

    # Download Gemma-2-2B text encoder
    GEMMA_PATH="models/Stable-diffusion/Lumina/Text Encoder/gemma_2_2b_fp16.safetensors"
    if [ -f "$GEMMA_PATH" ]; then
        echo -e "${BB0_ZENITH}✓ Gemma-2-2B text encoder already exists${NC}"
    else
        echo -e "${BB0_GLITCH}Downloading Gemma-2-2B text encoder (~4.9GB)...${NC}"
        huggingface-cli download neta-art/Neta-Lumina \
            gemma_2_2b_fp16.safetensors \
            --local-dir "models/Stable-diffusion/Lumina/Text Encoder" \
            --resume-download
        echo -e "${BB0_ZENITH}✓ Gemma-2-2B downloaded${NC}"
    fi

    # Symlink or copy shared FLUX VAE
    LUMINA_VAE_PATH="models/Stable-diffusion/Lumina/VAE/ae.safetensors"
    if [ -f "$LUMINA_VAE_PATH" ]; then
        echo -e "${BB0_ZENITH}✓ Lumina VAE already exists${NC}"
    else
        if [ -f "$VAE_PATH" ]; then
            echo -e "${BB0_GLITCH}Creating symlink to shared FLUX VAE...${NC}"
            ln -s "../../../VAE/ae.safetensors" "$LUMINA_VAE_PATH" 2>/dev/null || cp "$VAE_PATH" "$LUMINA_VAE_PATH"
            echo -e "${BB0_ZENITH}✓ Lumina VAE linked (shared with Flux)${NC}"
        else
            echo -e "${BB0_GLITCH}✗ FLUX VAE not found, please download it first${NC}"
        fi
    fi
    echo ""
    echo -e "${BB0_ZENITH}✓ Lumina 2.0 setup complete${NC}"
else
    echo -e "${BB0_GLITCH}Skipping Lumina 2.0${NC}"
fi
echo ""

# =====================================
# 9. Z-Image-Turbo
# =====================================
echo -e "${BB0_MIDNIGHT}=== Z-Image-Turbo ===${NC}"
echo "Z-Image-Turbo is a fast image generation model based on SD3"
echo ""
echo -e "${BB0_GLITCH}Dependencies: Shares FLUX VAE (models/VAE/ae.safetensors)${NC}"
echo -e "${BB0_GLITCH}              Requires Qwen-3-4B text encoder${NC}"
echo ""
read -p "Download Z-Image-Turbo? [y/N]: " download_zimage
if [[ $download_zimage =~ ^[Yy]$ ]]; then
    # Download DiT model
    ZIMAGE_PATH="models/Stable-diffusion/Z-Image/diffusion_pytorch_model.safetensors"
    if [ -f "$ZIMAGE_PATH" ]; then
        echo -e "${BB0_ZENITH}✓ Z-Image DiT already exists${NC}"
    else
        echo -e "${BB0_GLITCH}Downloading Z-Image DiT model...${NC}"
        mkdir -p models/Stable-diffusion/Z-Image
        huggingface-cli download stabilityai/stable-diffusion-3-medium \
            --local-dir models/Stable-diffusion/Z-Image \
            --resume-download
        echo -e "${BB0_ZENITH}✓ Z-Image DiT downloaded${NC}"
    fi

    # Download Qwen-3-4B text encoder
    QWEN_PATH="models/text_encoder/qwen_3_4b.safetensors"
    if [ -f "$QWEN_PATH" ]; then
        echo -e "${BB0_ZENITH}✓ Qwen-3-4B text encoder already exists${NC}"
    else
        echo -e "${BB0_GLITCH}Downloading Qwen-3-4B text encoder (~7.5GB)...${NC}"
        echo "Required for Z-Image-Turbo"
        huggingface-cli download stabilityai/stable-diffusion-3-medium \
            text_encoders/qwen_3_4b.safetensors \
            --local-dir models/text_encoder \
            --resume-download
        echo -e "${BB0_ZENITH}✓ Qwen-3-4B downloaded${NC}"
    fi

    echo ""
    echo -e "${BB0_ZENITH}✓ Z-Image-Turbo setup complete${NC}"
    echo -e "${BB0_MIDNIGHT}Note:${NC} Z-Image shares FLUX VAE from models/VAE/ae.safetensors"
else
    echo -e "${BB0_GLITCH}Skipping Z-Image-Turbo${NC}"
fi
echo ""

# =====================================
# Summary
# =====================================
echo ""
echo -e "${BB0_ZENITH}========================================"
echo "✅ Model Download Complete!"
echo "========================================${NC}"
echo ""
echo "Downloaded models are located in:"
echo "  • Flux: models/Stable-diffusion/Flux/"
echo "  • Lumina: models/Stable-diffusion/Lumina/ (UNet, Text Encoder, VAE)"
echo "  • Z-Image: models/Stable-diffusion/Z-Image/"
echo "  • Text Encoders: models/text_encoder/ (clip_l, t5xxl, qwen_3_4b, gemma)"
echo "  • VAE (shared): models/VAE/ (ae.safetensors used by Flux, Lumina, Z-Image)"
echo "  • ControlNet: models/ControlNet/"
echo "  • FILM: models/Deforum/film_interpolation/"
echo "  • Wan AI Video: models/Deforum/wan/"
echo "  • Qwen Prompts: models/Deforum/qwen/"
echo ""
echo -e "${BB0_MIDNIGHT}Note:${NC} Depth-Anything V3 package auto-installs from GitHub (requirements.txt)"
echo "Models download from HuggingFace on first use if not cached locally."
echo "Gifski and Real-ESRGAN binaries are also auto-downloaded."
echo ""
echo -e "${BB0_ZENITH}✅ You can now use Deforum with all render modes!${NC}"
echo ""
