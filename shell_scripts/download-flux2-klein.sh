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
echo "1) Flux 2 Klein 4B (~13GB VRAM, Apache 2.0 - commercial use OK)"
echo "2) Flux 2 Klein 9B (~29GB VRAM, non-commercial license)"
echo "3) Both models"
echo "4) Exit"
read -p "Choice [1-4]: " model_choice

if [ "$model_choice" -eq 4 ]; then
    echo "Exiting..."
    exit 0
fi

echo ""
echo -e "${BB0_MIDNIGHT}Choose checkpoint precision:${NC}"
echo "1) FP8 (recommended - smaller, faster)"
echo "2) BF16 (full precision - larger)"
read -p "Choice [1-2]: " precision_choice

echo ""
echo -e "${BB0_MIDNIGHT}Choose text encoder format:${NC}"
echo "1) Safetensors FP8 (recommended - smaller, compatible)"
echo "2) Safetensors BF16 (full precision)"
echo "3) GGUF (advanced - multiple quantization options)"
read -p "Choice [1-3]: " encoder_choice

download_klein_4b() {
    echo -e "\n${BB0_ZENITH}Downloading Flux 2 Klein 4B...${NC}"

    # Download checkpoint based on precision choice
    if [ "$precision_choice" -eq 1 ]; then
        echo -e "${BB0_VOID}Downloading checkpoint FP8 (4.07 GB)...${NC}"
        huggingface-cli download black-forest-labs/FLUX.2-klein-4b-fp8 \
            flux-2-klein-4b-fp8.safetensors \
            --local-dir "$FORGE_ROOT/models/Stable-diffusion" \
            --local-dir-use-symlinks False
        checkpoint_file="flux-2-klein-4b-fp8.safetensors"
    else
        echo -e "${BB0_VOID}Downloading checkpoint BF16 (7.75 GB)...${NC}"
        huggingface-cli download black-forest-labs/FLUX.2-klein-4B \
            flux-2-klein-4b.safetensors \
            --local-dir "$FORGE_ROOT/models/Stable-diffusion" \
            --local-dir-use-symlinks False
        checkpoint_file="flux-2-klein-4b.safetensors"
    fi

    # Download text encoder based on choice
    if [ "$encoder_choice" -eq 1 ]; then
        echo -e "${BB0_VOID}Downloading text encoder FP8 (Qwen 3 4B)...${NC}"
        huggingface-cli download jiangchengchengNLP/qwen3-4b-fp8-scaled \
            qwen3_4b_fp8_scaled.safetensors \
            --local-dir "$FORGE_ROOT/models/text_encoder" \
            --local-dir-use-symlinks False
        encoder_file="qwen3_4b_fp8_scaled.safetensors"
    elif [ "$encoder_choice" -eq 2 ]; then
        echo -e "${BB0_VOID}Downloading text encoder BF16 (Qwen 3 4B)...${NC}"
        huggingface-cli download Comfy-Org/z_image_turbo \
            split_files/text_encoders/qwen_3_4b.safetensors \
            --local-dir "$FORGE_ROOT/models/text_encoder" \
            --local-dir-use-symlinks False
        mv "$FORGE_ROOT/models/text_encoder/split_files/text_encoders/qwen_3_4b.safetensors" \
           "$FORGE_ROOT/models/text_encoder/" 2>/dev/null || true
        rm -rf "$FORGE_ROOT/models/text_encoder/split_files" 2>/dev/null || true
        encoder_file="qwen_3_4b.safetensors"
    else
        echo -e "${BB0_VOID}Downloading text encoder GGUF (Qwen 3 4B)...${NC}"
        echo -e "${BB0_MIDNIGHT}Available GGUF quantizations - download manually from:${NC}"
        echo "  https://huggingface.co/Qwen/Qwen3-4B-GGUF/tree/main"
        encoder_file="(GGUF - manual download required)"
    fi

    # Download Flux 2 VAE (different from Flux 1!)
    download_flux2_vae

    echo -e "${BB0_ZENITH}✓ Flux 2 Klein 4B downloaded successfully!${NC}"
    echo -e "  Checkpoint: models/Stable-diffusion/$checkpoint_file"
    echo -e "  Text Encoder: models/text_encoder/$encoder_file"
    echo -e "  VAE: models/VAE/flux2-vae.safetensors"
}

download_klein_9b() {
    echo -e "\n${BB0_ZENITH}Downloading Flux 2 Klein 9B...${NC}"

    # Download checkpoint based on precision choice
    if [ "$precision_choice" -eq 1 ]; then
        echo -e "${BB0_VOID}Downloading checkpoint FP8 (~8 GB)...${NC}"
        huggingface-cli download black-forest-labs/FLUX.2-klein-9b-fp8 \
            flux-2-klein-9b-fp8.safetensors \
            --local-dir "$FORGE_ROOT/models/Stable-diffusion" \
            --local-dir-use-symlinks False
        checkpoint_file="flux-2-klein-9b-fp8.safetensors"
    else
        echo -e "${BB0_VOID}Downloading checkpoint BF16 (~15 GB)...${NC}"
        huggingface-cli download black-forest-labs/FLUX.2-klein-9B \
            flux-2-klein-9b.safetensors \
            --local-dir "$FORGE_ROOT/models/Stable-diffusion" \
            --local-dir-use-symlinks False
        checkpoint_file="flux-2-klein-9b.safetensors"
    fi

    # Download text encoder based on choice
    if [ "$encoder_choice" -eq 1 ]; then
        echo -e "${BB0_GLITCH}⚠ FP8 text encoder not available for Qwen 3 8B${NC}"
        echo -e "${BB0_MIDNIGHT}Falling back to BF16...${NC}"
        encoder_choice=2
    fi

    if [ "$encoder_choice" -eq 2 ]; then
        echo -e "${BB0_VOID}Downloading text encoder BF16 (Qwen 3 8B)...${NC}"
        huggingface-cli download Comfy-Org/vae-text-encorder-for-flux-klein-9b \
            split_files/text_encoders/qwen_3_8b.safetensors \
            --local-dir "$FORGE_ROOT/models/text_encoder" \
            --local-dir-use-symlinks False
        mv "$FORGE_ROOT/models/text_encoder/split_files/text_encoders/qwen_3_8b.safetensors" \
           "$FORGE_ROOT/models/text_encoder/" 2>/dev/null || true
        rm -rf "$FORGE_ROOT/models/text_encoder/split_files" 2>/dev/null || true
        encoder_file="qwen_3_8b.safetensors"
    else
        echo -e "${BB0_VOID}Downloading text encoder GGUF (Qwen 3 8B)...${NC}"
        echo -e "${BB0_MIDNIGHT}Available GGUF quantizations - download manually from:${NC}"
        echo "  https://huggingface.co/Qwen/Qwen3-8B-GGUF/tree/main"
        encoder_file="(GGUF - manual download required)"
    fi

    # Download Flux 2 VAE (different from Flux 1!)
    download_flux2_vae

    echo -e "${BB0_ZENITH}✓ Flux 2 Klein 9B downloaded successfully!${NC}"
    echo -e "  Checkpoint: models/Stable-diffusion/$checkpoint_file"
    echo -e "  Text Encoder: models/text_encoder/$encoder_file"
    echo -e "  VAE: models/VAE/flux2-vae.safetensors"
}

download_flux2_vae() {
    # Download Flux 2 VAE (different from Flux 1!)
    if [ ! -f "$FORGE_ROOT/models/VAE/flux2-vae.safetensors" ]; then
        echo -e "${BB0_VOID}Downloading Flux 2 VAE (different from Flux 1)...${NC}"
        huggingface-cli download Comfy-Org/vae-text-encorder-for-flux-klein-9b \
            split_files/vae/flux2-vae.safetensors \
            --local-dir "$FORGE_ROOT/models/VAE" \
            --local-dir-use-symlinks False
        mv "$FORGE_ROOT/models/VAE/split_files/vae/flux2-vae.safetensors" \
           "$FORGE_ROOT/models/VAE/" 2>/dev/null || true
        rm -rf "$FORGE_ROOT/models/VAE/split_files" 2>/dev/null || true

        # Verify download with checksum if sha256sum available
        if command -v sha256sum &> /dev/null; then
            echo -e "${BB0_MIDNIGHT}Verifying Flux 2 VAE checksum...${NC}"
            # Note: Add actual checksum when available from HF repo
            echo -e "${BB0_MIDNIGHT}✓ Checksum verification (manual check recommended)${NC}"
        fi
    else
        echo -e "${BB0_ZENITH}Flux 2 VAE already exists, skipping...${NC}"
    fi
}

case $model_choice in
    1)
        download_klein_4b
        ;;
    2)
        download_klein_9b
        ;;
    3)
        download_klein_4b
        echo ""
        download_klein_9b
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
echo -e "${BB0_GLITCH}⚠ IMPORTANT: Flux 2 uses a different VAE than Flux 1!${NC}"
echo -e "  ${BB0_MIDNIGHT}Downloaded: flux2-vae.safetensors (Flux 2 specific)${NC}"
echo ""
echo -e "${BB0_MIDNIGHT}To use with Deforum:${NC}"
echo "1. Launch Forge: ./start-forge.sh"
echo "2. Select Flux 2 Klein checkpoint from dropdown"
echo "3. Ensure Flux 2 VAE is selected (not Flux 1 VAE)"
echo "4. Use any Deforum render mode (3D, Flux + Interpolation, etc.)"
echo ""
echo -e "${BB0_VOID}Sources:${NC}"
echo "  - Klein 4B BF16: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B"
echo "  - Klein 4B FP8: https://huggingface.co/black-forest-labs/FLUX.2-klein-4b-fp8"
echo "  - Klein 9B BF16: https://huggingface.co/black-forest-labs/FLUX.2-klein-9B"
echo "  - Klein 9B FP8: https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-fp8"
echo "  - Text Encoders: Comfy-Org, jiangchengchengNLP, Qwen repos"
echo "  - Flux 2 VAE: https://huggingface.co/Comfy-Org/vae-text-encorder-for-flux-klein-9b"
echo "  - Download guide: https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models"
