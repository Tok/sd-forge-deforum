# Deforum Extension Setup Guide

## Quick Start

```bash
# 1. Check current status
./setup-deforum.sh --check

# 2. Install Deforum dependencies
./setup-deforum.sh --install

# 3. Launch Forge with optimizations
cd ../../
python webui.py --sage --fast-fp16 --cuda-malloc --cuda-stream
```

## Python Version

**Forge Neo Recommendation**: Python 3.11.9

**Current Compatibility**:
- ✅ Python 3.11.x: Fully supported
- ⚠️ Python 3.12.x: Works but not officially recommended
- ❌ Python 3.10.x or older: Not tested

### Switching to Python 3.11.9

If you're on Python 3.12 and want to use the recommended version:

```bash
# 1. Check if Python 3.11 is available
python3.11 --version

# If not, install it:
# Ubuntu/Debian:
sudo apt install python3.11 python3.11-venv python3.11-dev

# 2. Recreate Forge venv
cd <forge-neo-directory>
rm -rf venv
python3.11 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Reinstall Forge (takes 10-15 minutes)
python launch.py --skip-torch-cuda-test --exit

# 4. Install Deforum dependencies
cd extensions/sd-forge-deforum
pip install -r requirements.txt
```

## Dependencies

### Core Dependencies (Required)
- numexpr, matplotlib, pandas - Math and plotting
- av, pims, imageio_ffmpeg - Video processing
- rich - Console output
- gdown, easydict - Utilities
- diffusers (git main) - Hugging Face diffusers
- transformers (4.36.0 - 5.0.0) - For Wan/Qwen models
- accelerate (0.25.0 - 2.0.0) - Model acceleration

### Audio Dependencies (Optional)
- librosa >= 0.10.0
- soundfile >= 0.12.0
- scipy >= 1.10.0
- plotly >= 5.14.0

### Install All Dependencies

```bash
pip install -r requirements.txt
```

## Performance Optimizations

Forge Neo supports several optimization flags that significantly improve performance:

### Recommended Flags (RTX 30/40/50 series)

```bash
python webui.py \
  --sage              # SageAttention (best performance)
  --fast-fp16         # Fast FP16 accumulation (PyTorch 2.7+)
  --cuda-malloc       # CUDA malloc optimization
  --cuda-stream       # CUDA stream optimization
```

### Optional Flags

```bash
  --pin-shared-memory # May improve speed but can cause OOM
  --flash             # FlashAttention (alternative to --sage)
```

### Optimization Details

**SageAttention** (`--sage`)
- Auto-installs on first launch with flag
- Best performance for most GPUs
- Requires triton (auto-installed)
- For RTX 50 series: may need SageAttention 2 (manual install)

**FlashAttention** (`--flash`)
- Alternative to SageAttention
- Requires manual installation: `pip install flash-attn --no-build-isolation`
- May have better compatibility on some systems

**Fast FP16** (`--fast-fp16`)
- Requires PyTorch >= 2.7.0
- Significant speed boost for mixed-precision training

### Example Launch Commands

**Basic (no optimizations)**:
```bash
python webui.py
```

**Recommended (with optimizations)**:
```bash
python webui.py --sage --fast-fp16 --cuda-malloc --cuda-stream
```

**Maximum performance (may cause OOM)**:
```bash
python webui.py --sage --fast-fp16 --cuda-malloc --cuda-stream --pin-shared-memory
```

## Troubleshooting

### "SageAttention not installed" Warning

This is normal. SageAttention will be auto-installed when you launch with `--sage` flag:

```bash
python webui.py --sage
```

### Version Mismatch Warnings

If you see warnings about transformers or accelerate versions, this is usually safe. The version constraints in `requirements.txt` are conservative. Current versions (as of 2025):
- transformers 4.56.2+ works fine
- accelerate 1.10.1+ works fine

### Python Version Warning

If using Python 3.12.3:
- Most features work fine
- Some edge cases may have compatibility issues
- Recommended: switch to Python 3.11.9 (see above)

### Missing Dependencies

```bash
# Install all at once
pip install -r requirements.txt

# Or install individually
pip install pandas rich librosa soundfile plotly numexpr av pims gdown easydict
```

## Verification

After setup, verify everything works:

```bash
# 1. Check Deforum dependencies
./setup-deforum.sh --check

# 2. Run unit tests
./run-unit-tests.sh

# 3. Launch Forge
cd ../../
python webui.py --sage --fast-fp16

# 4. Navigate to Deforum tab in WebUI
# 5. Try generating a test animation
```

## Related Scripts

- `setup-deforum.sh` - Setup and check dependencies
- `download-all-models.sh` - Download required models (Flux, Wan, etc.)
- `run-unit-tests.sh` - Run Deforum unit tests
- `run-api-tests.sh` - Run API integration tests

## Getting Help

- Check CLAUDE.md for architecture and development guide
- Check README.md for feature documentation
- Run `./setup-deforum.sh --help` for script usage
