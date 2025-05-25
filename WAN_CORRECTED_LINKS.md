# WAN Integration - CORRECTED REPOSITORY LINKS

## ✅ FIXED: Using Correct Wan-AI Repository 

**Repository**: https://huggingface.co/Wan-AI/Wan2.1-VACE-14B

The integration now uses the correct and working Wan-AI repository instead of dead links.

## Required Files (Exact Names) 📁

Download from: **Wan-AI/Wan2.1-VACE-14B**

### Essential Files (Required)
```
models_t5_umt5-xxl-enc-bf16.pth          # T5 encoder (11.4 GB)
Wan2.1_VAE.pth                           # VAE (508 MB)
diffusion_pytorch_model-00001-of-00007.safetensors  # DiT model part 1 (9.89 GB)
diffusion_pytorch_model-00002-of-00007.safetensors  # DiT model part 2 (9.84 GB)
diffusion_pytorch_model-00003-of-00007.safetensors  # DiT model part 3 (9.84 GB)
diffusion_pytorch_model-00004-of-00007.safetensors  # DiT model part 4 (9.84 GB)
diffusion_pytorch_model-00005-of-00007.safetensors  # DiT model part 5 (9.84 GB)
diffusion_pytorch_model-00006-of-00007.safetensors  # DiT model part 6 (7.91 GB)
diffusion_pytorch_model-00007-of-00007.safetensors  # DiT model part 7 (6.1 GB)
diffusion_pytorch_model.safetensors.index.json     # DiT index (119 kB)
```

### Optional Files
```
config.json                              # Model config (325 B)
```

**Total Size**: ~75 GB

## 🚀 EASIEST Download Method

```bash
# Install HuggingFace CLI
pip install huggingface_hub

# Download entire repository (recommended)
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir /path/to/your/wan/models
```

## Manual Download Commands 📥

If you prefer to download individual files:

```bash
# Navigate to your WAN model directory
cd /path/to/webui-forge/webui/models/wan

# Download T5 encoder (Required - 11.4 GB)
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth

# Download VAE (Required - 508 MB)  
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/Wan2.1_VAE.pth

# Download DiT model parts (Required - ~63 GB total)
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00001-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00002-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00003-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00004-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00005-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00006-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00007-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model.safetensors.index.json

# Download config (Optional - 325 B)
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/config.json
```

## Model Capabilities 🎯

### ✅ Supported
- **Text-to-Video**: Full support with high-quality 720p generation
- **Multiple Resolutions**: 480p, 720p, and 1280x720
- **Long Videos**: Up to 30 seconds per clip  
- **High FPS**: Up to 60 FPS generation
- **Text Generation**: Can generate Chinese and English text in videos

### ❌ Not Supported (VACE-14B Model)
- **Image-to-Video**: Not available in VACE-14B model
  - For I2V, download separately: `Wan-AI/Wan2.1-I2V-14B-720P`

## Current Behavior 🎯

### ✅ Success Path
- All required files present with correct names → Real WAN video generation
- Uses actual WAN T2V model inference from Wan-AI repository
- Generates high-quality video frames
- No fallback or placeholder content

### ❌ Failure Path  
- Files missing or incorrect names → Clear error with corrected download instructions
- System fails immediately with actionable error messages showing working URLs
- No generation occurs (no placeholder frames)
- User gets exact wget commands and HuggingFace CLI instructions

## What Changed ✨

### ✅ Fixed
- **Corrected Repository**: Now uses working `Wan-AI/Wan2.1-VACE-14B` instead of dead links
- **Updated URLs**: All download links now point to existing files
- **Multi-part DiT**: Properly handles the 7-part DiT model structure
- **File Validation**: Checks for exact filenames from the actual repository
- **HuggingFace CLI**: Added easier download method for large files

### ❌ Removed
- Dead links to `wangfuyun/WAN2.1` repository
- References to non-existent CLIP model in VACE-14B
- Incorrect single-file DiT model expectations

## Next Steps 🚀

1. **Download**: Use the HuggingFace CLI method (easiest) or manual wget commands
2. **Verify**: Check that all files are in your model directory with exact names
3. **Restart**: Restart WebUI to clear Python module cache
4. **Test**: Generate WAN video - should now work with real models or fail clearly

## Model Directory Structure 📁

Your WAN model directory should look like this:
```
/path/to/webui-forge/webui/models/wan/
├── models_t5_umt5-xxl-enc-bf16.pth                    # 11.4 GB
├── Wan2.1_VAE.pth                                     # 508 MB
├── diffusion_pytorch_model-00001-of-00007.safetensors # 9.89 GB
├── diffusion_pytorch_model-00002-of-00007.safetensors # 9.84 GB
├── diffusion_pytorch_model-00003-of-00007.safetensors # 9.84 GB
├── diffusion_pytorch_model-00004-of-00007.safetensors # 9.84 GB
├── diffusion_pytorch_model-00005-of-00007.safetensors # 9.84 GB
├── diffusion_pytorch_model-00006-of-00007.safetensors # 7.91 GB
├── diffusion_pytorch_model-00007-of-00007.safetensors # 6.1 GB
├── diffusion_pytorch_model.safetensors.index.json    # 119 kB
└── config.json                                        # 325 B (optional)
```

**Repository**: https://huggingface.co/Wan-AI/Wan2.1-VACE-14B
