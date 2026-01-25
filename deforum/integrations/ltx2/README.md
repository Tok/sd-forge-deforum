# LTX-2 Integration Status

## Current State: NON-FUNCTIONAL (as of 2026-01-25)

LTX-2 integration is implemented but **not currently working** due to upstream issues in diffusers library.

## Issues

### 1. GGUF Loading Fails
- **Issue:** [diffusers #12981](https://github.com/huggingface/diffusers/issues/12981)
- **Error:** `Unable to load weights from checkpoint file`
- **Status:** Known bug, awaiting fix from diffusers team
- **Note:** GGUF files download successfully (12.8GB in ~50s) but cannot be loaded

### 2. BitsAndBytes NF4 OOMs
- **Issue:** Caching allocator warmup tries to allocate full model before quantization
- **Error:** `torch.OutOfMemoryError: Allocation on device` during warmup
- **Status:** Happens even with `DISABLE_WARMUP=1`, `low_cpu_mem_usage=True`, and `max_memory` constraints
- **Requires:** 24GB+ VRAM for full model, or successful quantization (which fails due to warmup)

## What's Implemented

✅ **GGUF Support**
- Model variants: Q4_K_M (13GB), Q3_K_M (10GB), Q2_K (8GB)
- Auto-download from [unsloth/LTX-2-GGUF](https://huggingface.co/unsloth/LTX-2-GGUF)
- Proper file caching via `hf_hub_download()`
- CPU offloading via `enable_model_cpu_offload()`

✅ **BitsAndBytes NF4 Fallback**
- Quantizes both transformer and text_encoder
- Device mapping with memory constraints
- Auto-fallback if GGUF fails

✅ **UI Integration**
- Variant selection dropdown
- Auto-selection based on VRAM
- VRAM requirement display
- Audio mode configuration

✅ **Generation Pipeline**
- Resolution validation with auto-letterboxing
- Audio segment extraction
- Seed scheduling
- Frame generation loop

## Tested VRAM Configurations

- **14.3GB VRAM (RTX 4070 Ti SUPER):** ❌ Fails (both GGUF and NF4 OOM)
- **24GB+ VRAM:** ⚠️ Untested (should work with full precision)

## Workarounds

**For 14GB VRAM users:**
1. **Use Wan FLF2V** (recommended) - AI-powered interpolation, works with 14GB VRAM
2. **Use FILM** - Google's interpolation, lightweight and fast

**For future when fixed:**
- GGUF Q4_K_M file is cached at `~/.cache/huggingface/hub/models--unsloth--LTX-2-GGUF/`
- Will work immediately once diffusers fixes GGUF loading
- No re-download needed

## Next Steps

1. Monitor [diffusers #12981](https://github.com/huggingface/diffusers/issues/12981) for GGUF fix
2. Test with 24GB+ VRAM card if available
3. Consider alternative: Use ComfyUI with LTX-2 GGUF (confirmed working)

## Files

- `ltx2_pipeline.py` - Main pipeline (GGUF + NF4 loading)
- `ltx2_model_discovery.py` - Model variants and auto-selection
- `ltx2_setup.py` - VRAM checking and initialization
- `../rendering/keyframe_interp.py` - Integration with Flux + Interpolation mode
- `../rendering/ltx2_setup.py` - Setup orchestration
- `../rendering/resolution_utils.py` - Resolution validation

## References

- [LTX-2 Paper](https://huggingface.co/papers/2601.03233)
- [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2) - Base model
- [unsloth/LTX-2-GGUF](https://huggingface.co/unsloth/LTX-2-GGUF) - GGUF quantized variants
- [Diffusers GGUF Docs](https://huggingface.co/docs/diffusers/en/quantization/gguf)
- [Issue #12981](https://github.com/huggingface/diffusers/issues/12981) - GGUF loading bug
