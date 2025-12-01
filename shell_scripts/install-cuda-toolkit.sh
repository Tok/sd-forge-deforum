#!/bin/bash
# install-cuda-toolkit.sh - Install CUDA Toolkit for SageAttention compilation
#
# This script installs the CUDA toolkit (nvcc, CUDA_HOME) required for
# compiling SageAttention from source.
#
# PyTorch uses CUDA 12.8, so we install a compatible toolkit version (12.x)
#
# Usage:
#   ./install-cuda-toolkit.sh

set -e

# BB0 Slopcore colors (see docs/SLOPCORE.md)
BB0_ZENITH='\033[38;2;23;167;254m'    # #17A7FE - Cyan (info, success)
BB0_GLITCH='\033[38;2;255;20;147m'    # #FF1493 - Neon pink (warning, error)
BB0_VOID='\033[38;2;86;6;255m'        # #5606FF - Deep purple (emphasis)
BB0_MIDNIGHT='\033[38;2;55;87;255m'   # #3757FF - Mid blue (secondary)
NC='\033[0m' # No Color

echo -e "${BB0_ZENITH}========================================"
echo "CUDA Toolkit Installation"
echo "========================================${NC}"
echo ""

# Check if already installed
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${BB0_ZENITH}✓ CUDA toolkit already installed (nvcc $NVCC_VERSION)${NC}"
    echo ""
    nvcc --version
    exit 0
fi

echo -e "${BB0_MIDNIGHT}This will install CUDA Toolkit 12.x${NC}"
echo -e "${BB0_GLITCH}Required for compiling SageAttention from source${NC}"
echo ""
echo -e "${BB0_MIDNIGHT}Installation steps:${NC}"
echo "  1. Add NVIDIA CUDA repository"
echo "  2. Install CUDA toolkit (cuda-toolkit-12-6)"
echo "  3. Set CUDA_HOME environment variable"
echo ""
echo -e "${BB0_GLITCH}This requires sudo and will install ~3GB of packages${NC}"
echo ""

read -p "$(echo -e ${BB0_GLITCH}Continue? [y/N]: ${NC})" -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BB0_MIDNIGHT}Installation cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${BB0_ZENITH}Step 1/4: Add NVIDIA CUDA Repository${NC}"

# Install prerequisites
sudo apt-get update
sudo apt-get install -y wget gnupg

# Add NVIDIA package repository
# Using Ubuntu 22.04 x86_64 repository (adjust if needed)
DISTRO=ubuntu2204
ARCH=x86_64

wget https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm cuda-keyring_1.1-1_all.deb

echo -e "${BB0_ZENITH}✓ Repository added${NC}"
echo ""

echo -e "${BB0_ZENITH}Step 2/4: Update Package List${NC}"
sudo apt-get update
echo -e "${BB0_ZENITH}✓ Package list updated${NC}"
echo ""

echo -e "${BB0_ZENITH}Step 3/4: Install CUDA Toolkit${NC}"
echo -e "${BB0_GLITCH}This may take 5-10 minutes...${NC}"

# Install CUDA toolkit 12.6 (compatible with PyTorch cu128)
# Only install toolkit, not drivers (we already have working CUDA via PyTorch)
sudo apt-get install -y cuda-toolkit-12-6

echo -e "${BB0_ZENITH}✓ CUDA toolkit installed${NC}"
echo ""

echo -e "${BB0_ZENITH}Step 4/4: Set Environment Variables${NC}"

# Add CUDA_HOME to shell profile
CUDA_HOME_LINE='export CUDA_HOME=/usr/local/cuda-12.6'
PATH_LINE='export PATH=$CUDA_HOME/bin:$PATH'
LD_LIBRARY_PATH_LINE='export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH'

# Add to ~/.bashrc if not already present
if ! grep -q "CUDA_HOME" ~/.bashrc; then
    echo "" >> ~/.bashrc
    echo "# CUDA Toolkit" >> ~/.bashrc
    echo "$CUDA_HOME_LINE" >> ~/.bashrc
    echo "$PATH_LINE" >> ~/.bashrc
    echo "$LD_LIBRARY_PATH_LINE" >> ~/.bashrc
    echo -e "${BB0_ZENITH}✓ Environment variables added to ~/.bashrc${NC}"
else
    echo -e "${BB0_GLITCH}⚠ CUDA_HOME already in ~/.bashrc${NC}"
fi

# Set for current session
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo ""
echo -e "${BB0_ZENITH}========================================${NC}"
echo -e "${BB0_ZENITH}✓ CUDA Toolkit Installation Complete!${NC}"
echo -e "${BB0_ZENITH}========================================${NC}"
echo ""

# Verify installation
if command -v nvcc &> /dev/null; then
    echo -e "${BB0_ZENITH}✓ nvcc installed successfully${NC}"
    nvcc --version
    echo ""
    echo -e "${BB0_MIDNIGHT}CUDA_HOME: $CUDA_HOME${NC}"
else
    echo -e "${BB0_GLITCH}✗ nvcc not found in PATH${NC}"
    echo -e "${BB0_GLITCH}You may need to restart your shell or run: source ~/.bashrc${NC}"
fi

echo ""
echo -e "${BB0_MIDNIGHT}Next steps:${NC}"
echo "  1. Restart your shell (or run: source ~/.bashrc)"
echo "  2. Install SageAttention: ${BB0_ZENITH}./install-sageattention.sh${NC}"
