#!/bin/bash
# Download Flux 2 Klein models for Forge Neo
# Based on: https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models

set -e

# BB0 Slopcore colors (see docs/SLOPCORE.md)
BB0_ZENITH='\033[38;2;23;167;254m'    # #17A7FE - Cyan (info, success)
BB0_GLITCH='\033[38;2;255;20;147m'    # #FF1493 - Neon pink (warning, error)
BB0_VOID='\033[38;2;86;6;255m'        # #5606FF - Deep purple (emphasis)
BB0_MIDNIGHT='\033[38;2;55;87;255m'   # #3757FF - Mid blue (secondary)
NC='\033[0m' # No Color

# Detect Forge Neo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTENSION_ROOT="$(dirname "$SCRIPT_DIR")"
FORGE_ROOT="$(dirname "$(dirname "$EXTENSION_ROOT")")"

echo -e "${BB0_VOID}=== Flux 2 Klein Model Downloader ===${NC}"
echo ""
echo "This script downloads Flux 2 Klein models for Forge Neo."
echo "Forge Neo root: $FORGE_ROOT"
echo ""

# Check if we're in the right place
if [ ! -d "$FORGE_ROOT/models" ]; then
    echo -e "${BB0_GLITCH}ERROR: Cannot find models directory at $FORGE_ROOT/models${NC}"
    echo "Please run this script from the sd-forge-deforum/shell_scripts directory"
    exit 1
fi

# Create model directories
mkdir -p "$FORGE_ROOT/models/Stable-diffusion"
mkdir -p "$FORGE_ROOT/models/text_encoder"
mkdir -p "$FORGE_ROOT/models/VAE"

echo -e "${BB0_MIDNIGHT}Which Klein model do you want to download?${NC}"
echo "1) Flux 2 Klein 4B (fp8, ~13GB VRAM, Apache 2.0 - commercial use OK)"
echo "2) Flux 2 Klein 9B (fp8, ~29GB VRAM, non-commercial license)"
echo "3) Both models"
echo "4) Exit"
read -p "Choice [1-4]: " choice

download_klein_4b() {
    echo -e "\n${BB0_ZENITH}Downloading Flux 2 Klein 4B (FP8)...${NC}"

    # Download checkpoint (single safetensors file)
    echo -e "${BB0_VOID}Downloading checkpoint (4.07 GB)...${NC}"
    huggingface-cli download black-forest-labs/FLUX.2-klein-4b-fp8 \
        flux-2-klein-4b-fp8.safetensors \
        --local-dir "$FORGE_ROOT/models/Stable-diffusion" \
        --local-dir-use-symlinks False

    # Download text encoder (Qwen 3 4B fp8_scaled)
    echo -e "${BB0_VOID}Downloading text encoder (Qwen 3 4B fp8_scaled)...${NC}"
    huggingface-cli download jiangchengchengNLP/qwen3-4b-fp8-scaled \
        qwen3_4b_fp8_scaled.safetensors \
        --local-dir "$FORGE_ROOT/models/text_encoder" \
        --local-dir-use-symlinks False

    # Download VAE (shared with 9B)
    if [ ! -f "$FORGE_ROOT/models/VAE/flux2-vae.safetensors" ]; then
        echo -e "${BB0_VOID}Downloading Flux 2 VAE...${NC}"
        huggingface-cli download Comfy-Org/vae-text-encorder-for-flux-klein-9b \
            split_files/vae/flux2-vae.safetensors \
            --local-dir "$FORGE_ROOT/models/VAE" \
            --local-dir-use-symlinks False
        # Move from nested directory to VAE root
        mv "$FORGE_ROOT/models/VAE/split_files/vae/flux2-vae.safetensors" "$FORGE_ROOT/models/VAE/" 2>/dev/null || true
        rm -rf "$FORGE_ROOT/models/VAE/split_files" 2>/dev/null || true
    else
        echo -e "${BB0_ZENITH}Flux 2 VAE already exists, skipping...${NC}"
    fi

    echo -e "${BB0_ZENITH}✓ Flux 2 Klein 4B downloaded successfully!${NC}"
    echo -e "  Checkpoint: models/Stable-diffusion/flux-2-klein-4b-fp8.safetensors"
    echo -e "  Text Encoder: models/text_encoder/qwen3_4b_fp8_scaled.safetensors"
    echo -e "  VAE: models/VAE/flux2-vae.safetensors"
}

download_klein_9b() {
    echo -e "\n${BB0_ZENITH}Downloading Flux 2 Klein 9B (FP8)...${NC}"

    # Download checkpoint (single safetensors file)
    echo -e "${BB0_VOID}Downloading checkpoint (~8 GB)...${NC}"
    huggingface-cli download black-forest-labs/FLUX.2-klein-9b-fp8 \
        flux-2-klein-9b-fp8.safetensors \
        --local-dir "$FORGE_ROOT/models/Stable-diffusion" \
        --local-dir-use-symlinks False

    # Download text encoder (Qwen 3 8B bf16)
    echo -e "${BB0_VOID}Downloading text encoder (Qwen 3 8B bf16)...${NC}"
    huggingface-cli download Comfy-Org/vae-text-encorder-for-flux-klein-9b \
        split_files/text_encoders/qwen_3_8b.safetensors \
        --local-dir "$FORGE_ROOT/models/text_encoder" \
        --local-dir-use-symlinks False
    # Move from nested directory to text_encoder root
    mv "$FORGE_ROOT/models/text_encoder/split_files/text_encoders/qwen_3_8b.safetensors" "$FORGE_ROOT/models/text_encoder/" 2>/dev/null || true
    rm -rf "$FORGE_ROOT/models/text_encoder/split_files" 2>/dev/null || true

    # Download VAE (shared with 4B)
    if [ ! -f "$FORGE_ROOT/models/VAE/flux2-vae.safetensors" ]; then
        echo -e "${BB0_VOID}Downloading Flux 2 VAE...${NC}"
        huggingface-cli download Comfy-Org/vae-text-encorder-for-flux-klein-9b \
            split_files/vae/flux2-vae.safetensors \
            --local-dir "$FORGE_ROOT/models/VAE" \
            --local-dir-use-symlinks False
        # Move from nested directory to VAE root
        mv "$FORGE_ROOT/models/VAE/split_files/vae/flux2-vae.safetensors" "$FORGE_ROOT/models/VAE/" 2>/dev/null || true
        rm -rf "$FORGE_ROOT/models/VAE/split_files" 2>/dev/null || true
    else
        echo -e "${BB0_ZENITH}Flux 2 VAE already exists, skipping...${NC}"
    fi

    echo -e "${BB0_ZENITH}✓ Flux 2 Klein 9B downloaded successfully!${NC}"
    echo -e "  Checkpoint: models/Stable-diffusion/flux-2-klein-9b-fp8.safetensors"
    echo -e "  Text Encoder: models/text_encoder/qwen_3_8b.safetensors"
    echo -e "  VAE: models/VAE/flux2-vae.safetensors"
}

case $choice in
    1)
        download_klein_4b
        ;;
    2)
        download_klein_9b
        ;;
    3)
        download_klein_4b
        download_klein_9b
        ;;
    4)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${BB0_GLITCH}Invalid choice${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BB0_ZENITH}=== Download Complete ===${NC}"
echo ""
echo -e "${BB0_MIDNIGHT}Model Information:${NC}"
echo "  Flux 2 Klein 4B: ~13GB VRAM, Apache 2.0 (commercial use OK)"
echo "  Flux 2 Klein 9B: ~29GB VRAM, non-commercial license"
echo ""
echo -e "${BB0_MIDNIGHT}To use with Deforum:${NC}"
echo "1. Launch Forge: ./start-forge.sh"
echo "2. Select Flux 2 Klein model from checkpoint dropdown"
echo "3. Use any Deforum render mode (3D, Flux + Interpolation, etc.)"
echo ""
echo -e "${BB0_VOID}Sources:${NC}"
echo "  - Klein 4B: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
echo "  - Klein 9B: https://huggingface.co/black-forest-labs/FLUX.2-klein-9B"
echo "  - Download guide: https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models"
