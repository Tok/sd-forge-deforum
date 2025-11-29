# Flux 2 Support Status

**Last Updated:** 2025-11-29
**Status:** ❌ Not working with current GGUF models

## Overview

Flux 2 is Black Forest Labs' latest model featuring:
- **8 double-stream blocks + 48 single-stream blocks** (vs Flux 1's 19/38)
- **Single text encoder**: Mistral Small 3.1 (vs Flux 1's dual T5-XXL + CLIP-L)
- **Improved efficiency** - Same quality with fewer transformer blocks

## Current Situation

Flux 2 is not currently working with Deforum on Forge Neo. Testing reveals an architectural mismatch between the model format and Forge's expectations.

## Technical Details

### Expected Architecture (Flux 2 Standard)
Based on official Flux 2 documentation:
- `in_channels`: 64
- `patch_size`: 1
- Resulting input dimension: 64

### Observed Behavior
When loading available GGUF models:
- Detected `in_channels`: 32
- Detected `patch_size`: 2
- Resulting expected dimension: 128 (from 32 × 2 × 2)

### Runtime Error
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (3600x64 and 128x6144)
```

**What this means:**
- Input from VAE: 64 features
- Model weights expect: 128 features
- These dimensions are incompatible

## What We've Implemented

The `flux2-experimental` work (now merged to dev) includes:

1. **✅ Flux GGUF Compatibility Patch**
   - Adds `vec_in_dim` parameter fallback (768)
   - Auto-detects Flux 2 vs Flux 1 models
   - Provides diagnostic logging
   - **Works perfectly for Flux 1 GGUF models**

2. **✅ Architecture Detection**
   - Detects model version by transformer block counts
   - Logs detected parameters for troubleshooting
   - Helps diagnose compatibility issues

3. **✅ Documentation**
   - Technical analysis of the incompatibility
   - Clear error messages
   - Guidance for future compatibility work

## Possible Paths Forward

1. **Wait for native Forge Neo support**
   - Forge Neo may add official Flux 2 support
   - Would bypass GGUF compatibility issues

2. **Alternative model formats**
   - Non-quantized safetensors might work if Forge supports them
   - Original diffusers format may be compatible

3. **GGUF format updates**
   - Future GGUF conversions may use different parameters
   - New conversion tools may handle architecture better

4. **Investigate parameter mapping**
   - There may be a way to remap the dimensions
   - Requires deep understanding of both formats

## Files Modified

- `deforum/integrations/flux2/compat_patch.py` - Compatibility patches
- `deforum/rendering/data/frame/diffusion_frame.py` - Classic 3D fix
- `scripts/deforum.py` - Patch application

## Testing Notes

If you're testing Flux 2:

1. The patch will detect and log the architecture:
   ```
   INFO: 🔍 Flux 2 model detected (depth=8, depth_single_blocks=48)
   INFO:   → Architecture: in_channels=32, patch_size=2
   ```

2. You'll see the model load successfully (vec_in_dim is added)

3. Generation will fail at first sampling step with dimension mismatch

4. This confirms the architectural incompatibility

## References

- [Flux 2 Official Announcement](https://huggingface.co/blog/flux-2)
- [Flux 2 Model Architecture](https://huggingface.co/black-forest-labs/FLUX.2-dev)
- Deforum compatibility patch: `deforum/integrations/flux2/`

## Conclusion

While Flux 2 support is not currently working, the infrastructure is in place:
- Flux 1 GGUF works great with the compatibility patch
- Architecture detection is working correctly
- Clear diagnostic output helps troubleshooting
- Ready for future Flux 2 compatibility when available

The work done on `flux2-experimental` provides value even without working Flux 2 support, and positions Deforum well for when proper Flux 2 compatibility becomes available.
