#!/bin/bash
# download-forge-models.sh - Download base Forge models (SD, SDXL, etc.)
#
# This script helps download the base models required by Forge WebUI.
# For Flux and Deforum-specific models, use download-all-models.sh instead.
#
# Reference: https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTENSION_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FORGE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo -e "${CYAN}========================================"
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
    echo -e "${RED}❌ huggingface-cli not found!${NC}"
    echo ""
    echo "Install with: pip install huggingface-hub"
    exit 1
fi

echo "Which models would you like to download?"
echo ""
echo "  ${BLUE}1)${NC} SD1 VAE (vae-ft-mse-840000) - ~330MB"
echo "  ${BLUE}2)${NC} SDXL VAE (sdxl-vae-fp16-fix) - ~320MB"
echo "  ${BLUE}3)${NC} Both VAEs"
echo "  ${BLUE}4)${NC} Skip (will use CivitAI for checkpoints)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1|3)
        echo ""
        echo -e "${YELLOW}Downloading SD1 VAE...${NC}"
        huggingface-cli download stabilityai/sd-vae-ft-mse \
            vae-ft-mse-840000-ema-pruned.safetensors \
            --local-dir models/VAE \
            --local-dir-use-symlinks False \
            --resume-download
        echo -e "${GREEN}✓ SD1 VAE downloaded${NC}"

        if [ "$choice" = "1" ]; then
            break
        fi
        ;&
esac

case $choice in
    2|3)
        echo ""
        echo -e "${YELLOW}Downloading SDXL VAE...${NC}"
        huggingface-cli download madebyollin/sdxl-vae-fp16-fix \
            sdxl_vae.safetensors \
            --local-dir models/VAE \
            --local-dir-use-symlinks False \
            --resume-download
        echo -e "${GREEN}✓ SDXL VAE downloaded${NC}"
        ;;
    4)
        echo -e "${YELLOW}Skipping VAE downloads${NC}"
        ;;
esac

echo ""
echo -e "${CYAN}========================================"
echo "Checkpoint Download Information"
echo "========================================${NC}"
echo ""
echo "For SD1/SDXL checkpoints, visit CivitAI:"
echo ""
echo "  ${BLUE}SD 1.5:${NC} https://civitai.com/models/6424/chilloutmix"
echo "  ${BLUE}SDXL:${NC}   https://civitai.com/models/101055/sd-xl"
echo ""
echo "Download .safetensors files and place them in:"
echo "  ${GREEN}$FORGE_ROOT/models/Stable-diffusion/${NC}"
echo ""
echo -e "${GREEN}✓ Base model setup complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Download checkpoints from CivitAI (if needed)"
echo "  2. Run download-all-models.sh for Flux/Deforum models"
echo "  3. Launch Forge with: ${BLUE}./start-forge.sh${NC}"
echo ""
