#!/bin/bash
# download-forge-models.sh - Download base Forge models (SD, SDXL, etc.)
#
# This script helps download the base models required by Forge WebUI.
# For Flux and Deforum-specific models, use download-all-models.sh instead.
#
# Reference: https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models

set -e

# BB0 Slopcore colors (see docs/SLOPCORE.md)
BB0_ZENITH='\033[38;2;23;167;254m'    # #17A7FE - Cyan (info, success)
BB0_GLITCH='\033[38;2;255;20;147m'    # #FF1493 - Neon pink (warning, error)
BB0_VOID='\033[38;2;86;6;255m'        # #5606FF - Deep purple (emphasis)
BB0_MIDNIGHT='\033[38;2;55;87;255m'   # #3757FF - Mid blue (secondary)
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTENSION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FORGE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo -e "${BB0_ZENITH}========================================"
echo "Forge Base Models Download Script"
echo "========================================${NC}"
echo ""
echo "This script helps you download SD1/SDXL models."
echo "For Flux/Deforum models, use download-all-models.sh"
echo ""

cd "$FORGE_ROOT"

# Create directories
mkdir -p models/Stable-diffusion
mkdir -p models/VAE
mkdir -p models/text_encoder

# Check for HuggingFace CLI
if ! command -v huggingface-cli &> /dev/null; then
    echo -e "${BB0_GLITCH}❌ huggingface-cli not found!${NC}"
    echo ""
    echo "Install with: pip install huggingface-hub"
    exit 1
fi

echo "Which models would you like to download?"
echo ""
echo "  ${BB0_MIDNIGHT}1)${NC} SD1 VAE (vae-ft-mse-840000) - ~330MB"
echo "  ${BB0_MIDNIGHT}2)${NC} SDXL VAE (sdxl-vae-fp16-fix) - ~320MB"
echo "  ${BB0_MIDNIGHT}3)${NC} Both VAEs"
echo "  ${BB0_MIDNIGHT}4)${NC} Skip (will use CivitAI for checkpoints)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1|3)
        echo ""
        echo -e "${BB0_GLITCH}Downloading SD1 VAE...${NC}"
        huggingface-cli download stabilityai/sd-vae-ft-mse \
            vae-ft-mse-840000-ema-pruned.safetensors \
            --local-dir models/VAE \
            --local-dir-use-symlinks False \
            --resume-download
        echo -e "${BB0_ZENITH}✓ SD1 VAE downloaded${NC}"

        if [ "$choice" = "1" ]; then
            break
        fi
        ;&
esac

case $choice in
    2|3)
        echo ""
        echo -e "${BB0_GLITCH}Downloading SDXL VAE...${NC}"
        huggingface-cli download madebyollin/sdxl-vae-fp16-fix \
            sdxl_vae.safetensors \
            --local-dir models/VAE \
            --local-dir-use-symlinks False \
            --resume-download
        echo -e "${BB0_ZENITH}✓ SDXL VAE downloaded${NC}"
        ;;
    4)
        echo -e "${BB0_GLITCH}Skipping VAE downloads${NC}"
        ;;
esac

echo ""
echo -e "${BB0_ZENITH}========================================"
echo "Checkpoint Download Information"
echo "========================================${NC}"
echo ""
echo "For SD1/SDXL checkpoints, visit CivitAI:"
echo ""
echo "  ${BB0_MIDNIGHT}SD 1.5:${NC} https://civitai.com/models/6424/chilloutmix"
echo "  ${BB0_MIDNIGHT}SDXL:${NC}   https://civitai.com/models/101055/sd-xl"
echo ""
echo "Download .safetensors files and place them in:"
echo "  ${BB0_ZENITH}$FORGE_ROOT/models/Stable-diffusion/${NC}"
echo ""
echo -e "${BB0_ZENITH}✓ Base model setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Download checkpoints from CivitAI (if needed)"
echo "  2. Run download-all-models.sh for Flux/Deforum models"
echo "  3. Launch Forge with: ${BB0_MIDNIGHT}./start-forge.sh${NC}"
echo ""
