# Issue #24 Fix: Repair Broken Forge Neo Installation

## Problem Summary

After installing sd-forge-deforum, Forge Neo fails to launch with:
```
ImportError: DLL load failed while importing _C: The specified module could not be found.
```

This occurs because the deforum extension previously auto-installed Depth-Anything V3, which tried to install xformers/flash-attn. When these compiled CUDA extensions fail to install properly, they break the entire Forge installation.

**Note:** The fix is on the **dev** branch (main branch is unaffected as it doesn't include DA3).

## Solution: Step-by-Step Repair

### Recommended Approach (Using Extension Setup Scripts)

```bash
# Step 1: Update Forge Neo from neo branch
cd /path/to/forge-neo
git pull origin neo

# Step 2: Switch Deforum to dev branch (where fix is available)
cd extensions/sd-forge-deforum
git checkout dev
git pull origin dev

# Step 3: Use automated setup script
./setup.sh --install  # Handles cleanup + dependency reinstall

# Step 4: Test launch using extension's start script
./start-forge.sh      # Linux/Mac
# OR: start-forge.bat on Windows
```

Forge should now launch successfully!

### Manual Repair (If Automated Approach Fails)

#### Step 1: Update Forge Neo from neo branch

```bash
cd /path/to/forge-neo
git pull origin neo
```

This updates Forge Neo with the latest fixes (includes FLUX.2 Klein support).

#### Step 2: Remove Problematic Packages

```bash
# Activate venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows

# Remove broken packages
pip uninstall xformers flash-attn flash-attn-3 depth-anything-3 -y
```

#### Step 3: Reinstall Forge Requirements

```bash
# Still in forge-neo root directory
pip install -r requirements.txt --force-reinstall
```

#### Step 4: Update Deforum to dev branch

```bash
cd extensions/sd-forge-deforum
git checkout dev  # IMPORTANT: Fix is on dev branch (main unaffected)!
git pull origin dev

# Install fixed requirements
pip install -r requirements.txt
```

#### Step 5: Test Launch

```bash
cd extensions/sd-forge-deforum
./start-forge.sh      # Linux/Mac
# OR: start-forge.bat on Windows
```

Forge should now launch successfully!

## What Changed in the Fix

### requirements.txt Changes:

**Before:**
```txt
git+https://github.com/huggingface/diffusers.git  # Bleeding-edge main
transformers>=4.36.0,<5.0.0                        # Wide range
git+https://github.com/ByteDance-Seed/Depth-Anything-3.git  # Auto-installed
```

**After:**
```txt
diffusers>=0.36.0,<0.37.0                 # Match Forge Neo 0.36.0
transformers>=4.36.0,<=4.56.2             # Match Forge Neo 4.56.2
# DA3 is now OPTIONAL (commented out)
```

### Why This Fixes the Issue:

1. **Version Pinning:** Matches Forge Neo's exact dependency versions (no conflicts)
2. **No DA3 Auto-Install:** Prevents xformers/flash-attn from being installed
3. **Graceful Degradation:** Core features work perfectly without DA3

## What You Can Still Do (Everything!)

✅ **All Animation Modes:**
- 3D mode with depth warping
- Flux + Interpolation mode
- All keyframe distribution modes

✅ **All Depth Features:**
- Depth-Anything V2 (default, better temporal consistency for video)
- All depth-based transformations
- Camera movement and warping

✅ **All Video Features:**
- Wan video generation (TI2V, FLF2V)
- FILM interpolation
- Qwen prompt enhancement
- Camera Shakify

✅ **All Models:**
- Flux.1 Dev/Schnell
- Lumina 2.0
- Z-Image Turbo
- SDXL

## What Requires DA3 (Optional Advanced Features)

❌ **Only if you need:**
- Multi-view depth estimation
- 3D Gaussian Splatting (3DGS) - **currently in incubation/tuning lab**
- Depth-ray pose estimation

**Why choose DA3 over DA2?**
- **DA2 (default):** Better temporal consistency (smoother video), always available
- **DA3 (optional):** More VRAM efficient, required for 3DGS experiments

**To install DA3 (optional):**
```bash
# Only if you have Python 3.11 + CUDA toolkit!
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

## Verification

After completing the steps, verify your installation:

```bash
# Check Python version (should be 3.11.x)
python --version

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check Deforum dependencies
cd extensions/sd-forge-deforum
./setup.sh --check
```

## Expected Output:

```
Current Python: 3.11.9
✓ Using Python 3.11 (recommended)

PyTorch:
  Version: 2.5.1+cu121
  CUDA: True
  GPU: NVIDIA GeForce RTX 4090

✓ All dependencies installed

Checking Forge optimizations...
✓ SageAttention installed (optional)
⚠ FlashAttention not installed (optional)
```

## Still Having Issues?

See `docs/TROUBLESHOOTING.md` for comprehensive troubleshooting.

**Common issues:**
- Python 3.12: Some features unavailable (see migration steps)
- Out of Memory: Reduce resolution, use quantized models
- Model not found: Run `./shell_scripts/download-all-models.sh`

## Clean Reinstall (Nuclear Option)

If repair fails, perform a clean reinstall:

```bash
# 1. Backup settings
cp extensions/sd-forge-deforum/deforum/config/default_settings.txt ~/deforum_backup.txt

# 2. Remove extension
rm -rf extensions/sd-forge-deforum

# 3. Clone fresh on dev branch
cd extensions
git clone -b dev https://github.com/Tok/sd-forge-deforum.git

# 4. Install with fixes
cd sd-forge-deforum
./setup.sh --install

# 5. Launch using extension's start script
./start-forge.sh      # Linux/Mac
# OR: start-forge.bat on Windows
```

## Summary

This fix makes Depth-Anything V3 optional to prevent xformers/flash-attn dependency conflicts. Core Deforum functionality is unchanged - you can use all features with the default Depth-Anything V2 model.

**The key changes:**
- ✅ Version pinning matches Forge Neo exactly
- ✅ No forced installation of problematic dependencies
- ✅ Graceful error handling for missing optional packages
- ✅ Comprehensive troubleshooting documentation

**For more details:**
- Full guide: `docs/TROUBLESHOOTING.md`
- Commit: `3070acd5` - "fix: Make Depth-Anything V3 optional"
- Date: 2026-01-29
