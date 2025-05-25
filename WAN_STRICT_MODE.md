# WAN Integration - STRICT MODE

**IMPORTANT: This WAN integration uses STRICT FAIL-FAST behavior. No fallbacks, no placeholders.**

## Current Status
- **OLD FILES REMOVED**: All fallback/placeholder generation removed
- **STRICT VALIDATION**: System fails immediately if WAN models are missing
- **REAL WAN MODELS ONLY**: Uses actual WAN T2V/I2V inference or fails completely

## Required Files

Place these files in your WAN model directory with **EXACT** filenames:

### Essential Files (Required)
1. **T5 Encoder**: `models_t5_umt5-xxl-enc-bf16.pth`
   - Download: https://huggingface.co/wangfuyun/WAN2.1/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
2. **VAE**: `Wan2.1_VAE.pth`  
   - Download: https://huggingface.co/wangfuyun/WAN2.1/resolve/main/Wan2.1_VAE.pth
3. **DiT Model**: `diffusion_pytorch_model.safetensors` (or similar)
   - Should already be present in your model directory

### Optional Files (For I2V support)
4. **CLIP**: `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
   - Download: https://huggingface.co/wangfuyun/WAN2.1/resolve/main/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth

## Download Instructions

```bash
# Navigate to your WAN model directory
cd /path/to/your/wan/models

# Download T5 encoder
wget https://huggingface.co/wangfuyun/WAN2.1/resolve/main/models_t5_umt5-xxl-enc-bf16.pth

# Download VAE
wget https://huggingface.co/wangfuyun/WAN2.1/resolve/main/Wan2.1_VAE.pth

# Download CLIP (optional, for I2V)
wget https://huggingface.co/wangfuyun/WAN2.1/resolve/main/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
```

Or download manually from browser and save with exact filenames shown above.

## What Changed

### ❌ REMOVED (Old Behavior)
- All placeholder/synthetic frame generation
- Automatic download attempts that were failing
- SD fallback modes for WAN
- `t5_encoder_wan_compatible.pth` references (incorrect filename)

### ✅ ADDED (New Behavior)  
- Strict validation for exact WAN model filenames
- Clear error messages with download URLs
- Real WAN T2V/I2V model inference calls
- Proper tensor-to-frame conversion for WAN output
- Complete failure if WAN models unavailable (no fallbacks)

## Error Handling

If you see errors like:
- `❌ Still missing files after download attempts: ['t5_encoder_wan_compatible.pth']`
- `🔄 Using fallback mode...`

This means the old cached files are still being used. **Restart the WebUI completely** to clear Python module cache.

## Behavior

- **SUCCESS**: WAN models work correctly → Real WAN video generation
- **FAILURE**: WAN models missing → Clear error with download instructions, no generation
- **NO FALLBACKS**: System does not generate placeholder videos or use SD fallbacks

This ensures you either get real WAN videos or clear guidance on what's missing.
