#!/bin/bash

# download-all-models.sh
# Downloads all required models for Deforum extension at once
# This ensures a complete installation and tests all download hooks

set -e  # Exit on error

echo "========================================"
echo "Deforum Model Download Script"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory (extension root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Navigate to Forge root (two levels up)
FORGE_ROOT="$(cd ../.. && pwd)"
cd "$FORGE_ROOT"

echo -e "${BLUE}Working from Forge root: $FORGE_ROOT${NC}"
echo ""

# Create model directories
echo -e "${YELLOW}Creating model directories...${NC}"
mkdir -p models/Stable-diffusion/Flux
mkdir -p models/VAE
mkdir -p models/text_encoder
mkdir -p models/ControlNet
mkdir -p models/Deforum/film_interpolation
mkdir -p models/Deforum/wan
mkdir -p models/Deforum/qwen
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Check for HuggingFace CLI
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${RED}❌ huggingface-cli not found!${NC}"
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

    echo -e "${YELLOW}Downloading $name...${NC}"
    python -c "
import sys
from torch.hub import download_url_to_file
download_url_to_file('$url', '$dest', progress=True)
"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ $name downloaded successfully${NC}"
    else
        echo -e "${RED}❌ Failed to download $name${NC}"
        return 1
    fi
}

# =====================================
# 1. Flux.1 Dev BNB NF4 v2 (MOST IMPORTANT)
# =====================================
echo -e "${BLUE}=== Flux.1 Dev Checkpoint (Quantized) ===${NC}"
FLUX_PATH="models/Stable-diffusion/Flux/flux1-dev-bnb-nf4-v2.safetensors"
if [ -f "$FLUX_PATH" ]; then
    echo -e "${GREEN}✓ Flux checkpoint already exists${NC}"
else
    echo -e "${YELLOW}Downloading Flux.1 Dev BNB NF4 v2 (~10GB quantized)...${NC}"
    echo "Note: This is a 4-bit quantized version optimized for lower VRAM usage"
    huggingface-cli download lllyasviel/flux1-dev-bnb-nf4 \
        flux1-dev-bnb-nf4-v2.safetensors \
        --local-dir models/Stable-diffusion/Flux \
        --resume-download
    echo -e "${GREEN}✓ Flux checkpoint downloaded${NC}"
fi
echo ""

# =====================================
# 2. Flux VAE and Text Encoders
# =====================================
echo -e "${BLUE}=== Flux VAE and Text Encoders ===${NC}"

# VAE (ae.safetensors)
VAE_PATH="models/VAE/ae.safetensors"
if [ -f "$VAE_PATH" ]; then
    echo -e "${GREEN}✓ VAE already exists${NC}"
else
    echo -e "${YELLOW}Downloading Flux VAE (ae.safetensors)...${NC}"
    huggingface-cli download black-forest-labs/FLUX.1-dev \
        ae.safetensors \
        --local-dir models/VAE \
        --resume-download
    echo -e "${GREEN}✓ VAE downloaded${NC}"
fi

# CLIP-L text encoder
CLIP_L_PATH="models/VAE/clip_l.safetensors"
if [ -f "$CLIP_L_PATH" ]; then
    echo -e "${GREEN}✓ CLIP-L already exists${NC}"
else
    echo -e "${YELLOW}Downloading CLIP-L text encoder...${NC}"
    huggingface-cli download comfyanonymous/flux_text_encoders \
        clip_l.safetensors \
        --local-dir models/VAE \
        --local-dir-use-symlinks False \
        --resume-download
    echo -e "${GREEN}✓ CLIP-L downloaded${NC}"
fi

# T5-XXL text encoder
T5_PATH="models/VAE/t5xxl_fp16.safetensors"
if [ -f "$T5_PATH" ]; then
    echo -e "${GREEN}✓ T5-XXL already exists${NC}"
else
    echo -e "${YELLOW}Downloading T5-XXL text encoder (fp16)...${NC}"
    huggingface-cli download comfyanonymous/flux_text_encoders \
        t5xxl_fp16.safetensors \
        --local-dir models/VAE \
        --local-dir-use-symlinks False \
        --resume-download
    echo -e "${GREEN}✓ T5-XXL downloaded${NC}"
fi
echo ""

# =====================================
# 3. Flux ControlNet V2
# =====================================
echo -e "${BLUE}=== Flux ControlNet V2 ===${NC}"
echo "Choose which Flux ControlNet models to download:"
echo "  1) Canny (Edge detection, ~3.5GB)"
echo "  2) Depth (Depth conditioning, ~3.5GB)"
echo "  3) Both Canny and Depth"
echo "  4) Skip ControlNet models"
read -p "Enter choice [1-4]: " controlnet_choice

case $controlnet_choice in
    1|3)
        echo -e "${YELLOW}Downloading Flux ControlNet Canny...${NC}"
        huggingface-cli download InstantX/FLUX.1-dev-Controlnet-Canny \
            --local-dir models/ControlNet/FLUX.1-dev-Controlnet-Canny \
            --resume-download
        echo -e "${GREEN}✓ ControlNet Canny downloaded${NC}"
        ;&  # Fall through if choice was 3
esac

case $controlnet_choice in
    2|3)
        echo -e "${YELLOW}Downloading Flux ControlNet Depth...${NC}"
        huggingface-cli download Shakker-Labs/FLUX.1-dev-ControlNet-Depth \
            --local-dir models/ControlNet/FLUX.1-dev-ControlNet-Depth \
            --resume-download
        echo -e "${GREEN}✓ ControlNet Depth downloaded${NC}"
        ;;
    4)
        echo -e "${YELLOW}Skipping ControlNet models${NC}"
        ;;
esac
echo ""

# =====================================
# 4. FILM Interpolation Model
# =====================================
echo -e "${BLUE}=== FILM Interpolation Model ===${NC}"
FILM_PATH="models/Deforum/film_interpolation/film_net_fp16.pt"
if [ -f "$FILM_PATH" ]; then
    echo -e "${GREEN}✓ FILM model already exists${NC}"
else
    download_file \
        "https://github.com/hithereai/frame-interpolation-pytorch/releases/download/film_net_fp16.pt/film_net_fp16.pt" \
        "$FILM_PATH" \
        "FILM model (film_net_fp16.pt)"
fi
echo ""

# =====================================
# 5. Wan Models (HuggingFace)
# =====================================
echo -e "${BLUE}=== Wan AI Video Models ===${NC}"
echo "Choose which Wan models to download:"
echo "  1) FLF2V-14B (Required for FLF2V interpolation, ~14GB)"
echo "  2) TI2V-5B (Recommended for T2V/I2V, 24GB VRAM, ~5GB)"
echo "  3) TI2V-A14B (Highest quality MoE, 32GB+ VRAM, ~14GB)"
echo "  4) All models (Downloads all 3)"
echo "  5) Skip Wan models"
read -p "Enter choice [1-5]: " wan_choice

case $wan_choice in
    1|4)
        echo -e "${YELLOW}Downloading Wan2.1-FLF2V-14B...${NC}"
        huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers \
            --local-dir models/Deforum/wan/Wan2.1-FLF2V-14B \
            --resume-download
        echo -e "${GREEN}✓ FLF2V-14B downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $wan_choice in
    2|4)
        echo -e "${YELLOW}Downloading Wan2.2-TI2V-5B...${NC}"
        huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers \
            --local-dir models/Deforum/wan/Wan2.2-TI2V-5B \
            --resume-download
        echo -e "${GREEN}✓ TI2V-5B downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $wan_choice in
    3|4)
        echo -e "${YELLOW}Downloading Wan2.2-TI2V-A14B...${NC}"
        huggingface-cli download Wan-AI/Wan2.2-TI2V-A14B-Diffusers \
            --local-dir models/Deforum/wan/Wan2.2-TI2V-A14B \
            --resume-download
        echo -e "${GREEN}✓ TI2V-A14B downloaded${NC}"
        ;;
    5)
        echo -e "${YELLOW}Skipping Wan models${NC}"
        ;;
esac
echo ""

# =====================================
# 6. Qwen AI Prompt Enhancement Models
# =====================================
echo -e "${BLUE}=== Qwen Prompt Enhancement Models ===${NC}"
echo "Choose which Qwen model to download (for AI prompt enhancement):"
echo "  1) Qwen2.5-3B-Instruct (Recommended for low VRAM, ~3GB)"
echo "  2) Qwen2.5-7B-Instruct (Better quality, ~7GB)"
echo "  3) Qwen2.5-14B-Instruct (Best quality, 32GB+ VRAM, ~14GB)"
echo "  4) All models (Downloads all 3)"
echo "  5) Skip Qwen models"
read -p "Enter choice [1-5]: " qwen_choice

case $qwen_choice in
    1|4)
        echo -e "${YELLOW}Downloading Qwen2.5-3B-Instruct...${NC}"
        huggingface-cli download Qwen/Qwen2.5-3B-Instruct \
            --local-dir models/Deforum/qwen/Qwen2.5-3B-Instruct \
            --resume-download
        echo -e "${GREEN}✓ Qwen2.5-3B-Instruct downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $qwen_choice in
    2|4)
        echo -e "${YELLOW}Downloading Qwen2.5-7B-Instruct...${NC}"
        huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
            --local-dir models/Deforum/qwen/Qwen2.5-7B-Instruct \
            --resume-download
        echo -e "${GREEN}✓ Qwen2.5-7B-Instruct downloaded${NC}"
        ;&  # Fall through if choice was 4
esac

case $qwen_choice in
    3|4)
        echo -e "${YELLOW}Downloading Qwen2.5-14B-Instruct...${NC}"
        huggingface-cli download Qwen/Qwen2.5-14B-Instruct \
            --local-dir models/Deforum/qwen/Qwen2.5-14B-Instruct \
            --resume-download
        echo -e "${GREEN}✓ Qwen2.5-14B-Instruct downloaded${NC}"
        ;;
    5)
        echo -e "${YELLOW}Skipping Qwen models${NC}"
        ;;
esac
echo ""

# =====================================
# Summary
# =====================================
echo ""
echo -e "${GREEN}========================================"
echo "✅ Model Download Complete!"
echo "========================================${NC}"
echo ""
echo "Downloaded models are located in:"
echo "  • Flux: models/Stable-diffusion/Flux/"
echo "  • VAE & Text Encoders: models/VAE/"
echo "  • ControlNet: models/ControlNet/"
echo "  • FILM: models/Deforum/film_interpolation/"
echo "  • Wan AI Video: models/Deforum/wan/"
echo "  • Qwen Prompts: models/Deforum/qwen/"
echo ""
echo -e "${BLUE}Note:${NC} Depth models (Depth-Anything V2) will be auto-downloaded"
echo "on first use. Gifski and Real-ESRGAN binaries are also auto-downloaded."
echo ""
echo -e "${GREEN}✅ You can now use Deforum with Flux + Interpolation mode!${NC}"
echo ""
