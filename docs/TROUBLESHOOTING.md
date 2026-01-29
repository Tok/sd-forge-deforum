# Deforum Troubleshooting Guide

## Installation Issues

### Issue: "ImportError: DLL load failed while importing _C: The specified module could not be found"

**Symptom:** After installing sd-forge-deforum, Forge Neo fails to launch with an import error related to `flash_attn_3` or `xformers`.

**Root Cause:** The extension previously auto-installed Depth-Anything V3, which has optional dependencies (xformers, flash-attn) that require compiled CUDA extensions. When these fail to install properly, they break the entire Forge installation.

**Solution:**

1. **Uninstall problematic packages:**
   ```bash
   cd /path/to/forge-neo
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip uninstall xformers flash-attn flash-attn-3 depth-anything-3 -y
   ```

2. **Reinstall Forge Neo requirements:**
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

3. **Reinstall Deforum with fixed requirements:**
   ```bash
   cd extensions/sd-forge-deforum
   git pull  # Get the fixed requirements.txt
   pip install -r requirements.txt
   ```

4. **Test Forge launch:**
   ```bash
   cd /path/to/forge-neo
   python webui.py
   ```

**Prevention:** The fixed `requirements.txt` no longer auto-installs Depth-Anything V3 to prevent this issue. Core Deforum functionality works perfectly with Depth-Anything V2 (always available).

### Optional: Installing Depth-Anything V3 (Advanced)

Depth-Anything V3 provides advanced features (multi-view depth, 3D Gaussian Splatting) but has complex dependencies.

**Requirements:**
- Python 3.11 (3.12+ often has compatibility issues)
- CUDA toolkit installed (nvcc compiler)
- Compatible GPU drivers

**Installation:**
```bash
# Only attempt if you meet all requirements above!
pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git
```

**If installation fails:** Don't worry! Core Deforum works great without DA3. You'll still have:
- Depth-Anything V2 for monocular depth (excellent quality)
- All animation modes (3D, Flux + Interpolation)
- All rendering features

---

## Dependency Version Conflicts

### Issue: Diffusers or Transformers version conflicts

**Symptom:** After installing Deforum, you see warnings about incompatible package versions.

**Solution:**

The fixed `requirements.txt` now pins versions compatible with Forge Neo:
- `diffusers>=0.36.0,<0.37.0` (matches Forge Neo 0.36.0)
- `transformers>=4.36.0,<=4.56.2` (matches Forge Neo 4.56.2)

If you installed before this fix:
```bash
cd extensions/sd-forge-deforum
git pull
pip install -r requirements.txt --upgrade
```

---

## Python Version Issues

### Issue: "This program is tested with 3.11.9 Python"

**Symptom:** Forge warns about Python version mismatch.

**Why it matters:**
- Python 3.11: Best compatibility, prebuilt wheels for most packages
- Python 3.12+: Many packages (xformers, flash-attn) lack prebuilt wheels, require compilation

**Solution:**

**Option 1: Use Python 3.11 (Recommended)**
```bash
cd extensions/sd-forge-deforum
./setup.sh --migrate  # Full venv migration to Python 3.11
```

**Option 2: Stay on Python 3.12 (Limited Features)**
- Skip SageAttention (requires compilation)
- Skip Depth-Anything V3 (requires xformers)
- Core features still work fine!

---

## Runtime Errors

### Issue: Out of Memory (OOM) during generation

**Symptoms:**
- CUDA out of memory errors
- System freezes during generation

**Solutions:**

1. **Reduce resolution:**
   - Try 512x512 or 768x768 instead of 1024x1024

2. **Use quantized models:**
   - Flux: `flux1-dev-bnb-nf4-v2.safetensors` (12GB instead of 24GB)
   - Qwen: Select 3B instead of 7B/14B for prompt enhancement

3. **Lower max_frames:**
   - Reduce animation length
   - Use higher cadence (fewer diffusion steps)

4. **Enable cleanup:**
   - Use "Cleanup Qwen Cache" button after prompt enhancement
   - Close other GPU applications

### Issue: Generation fails with "Model not found"

**Symptom:** Error about missing Flux, Wan, or other model files.

**Solution:**

Download required models:
```bash
cd extensions/sd-forge-deforum
./shell_scripts/download-all-models.sh  # Interactive downloader
```

Or use UI download buttons in "Wan Models" tab.

**Note:** Flux and Wan are NOT auto-downloaded (prevent forced ~29GB download). Other models (Z-Image, Lumina, SDXL, Depth-Anything V2) work immediately.

---

## Clean Reinstall

If all else fails, perform a clean reinstall:

```bash
# 1. Backup your settings
cp extensions/sd-forge-deforum/deforum/config/default_settings.txt ~/deforum_backup.txt

# 2. Remove extension
cd /path/to/forge-neo
rm -rf extensions/sd-forge-deforum

# 3. Clone fresh copy
cd extensions
git clone https://github.com/Tok/sd-forge-deforum.git

# 4. Install dependencies (with fixes)
cd sd-forge-deforum
./setup.sh --install

# 5. Test launch
cd /path/to/forge-neo
python webui.py
```

---

## Getting Help

If you're still experiencing issues:

1. **Check logs:** Look for error messages in the console output
2. **Check Python version:** `python --version` (should be 3.11.x)
3. **Check CUDA:** `python -c "import torch; print(torch.cuda.is_available())"`
4. **Report issue:** https://github.com/Tok/sd-forge-deforum/issues
   - Include: Python version, OS, GPU model, error message, full console output

---

## Common Questions

**Q: Will disabling DA3 break existing projects?**
A: No! DA3 was only used if you explicitly selected a "Depth-Anything-V3" model in the UI. Default is V2, which continues to work perfectly.

**Q: Can I still use advanced features without DA3?**
A: Yes! All core features work:
- ✅ All animation modes (3D, Flux + Interpolation)
- ✅ Depth-Anything V2 (excellent quality)
- ✅ Camera Shakify
- ✅ Wan video generation
- ✅ Qwen prompt enhancement
- ✅ All interpolation methods

**Q: Should I install SageAttention?**
A: Optional but recommended for RTX 30/40/50 GPUs:
- Requires CUDA toolkit (~3GB)
- 5-10% speed improvement
- See `./shell_scripts/install-sageattention.sh`

**Q: Why not just fix xformers installation?**
A: xformers requires:
- Compiled C++ extensions (platform-specific)
- CUDA toolkit (nvcc compiler)
- Compatible Python version (often fails on 3.12+)
- Prebuilt wheels don't exist for all platforms

Making it truly optional is more reliable than forcing compilation.