# LTX-2 Integration Status

## Current State: NOT VIABLE FOR 14GB VRAM (as of 2026-01-25)

LTX-2 integration is fully implemented but **not viable for <15GB VRAM** due to GGUF/CPU offload incompatibility.

## Critical Issue: GGUF + CPU Offload Incompatibility

### The Problem
GGUF quantization is **fundamentally incompatible** with accelerate's CPU offloading:
- GGUF tensors have `quant_type` metadata required for dequantization
- `enable_sequential_cpu_offload()` moves tensors to "meta" device during setup
- This causes `quant_type` to become `None`, breaking GGUF's dequantization lookup
- Error: `KeyError: None` in `GGML_QUANT_SIZES[quant_type]`

### Why This Matters
Without CPU offload, **all components must fit in VRAM simultaneously**:
- Transformer (GGUF): 8-13GB depending on quantization
- Text Encoder (Gemma): ~5GB
- VAE: ~2GB
- **Minimum Total: 15GB** (with Q2_K GGUF)

### VRAM Requirements (No CPU Offload)
| Variant | Transformer | + Text Encoder | + VAE | **Total VRAM** |
|---------|-------------|----------------|-------|----------------|
| Q4_K_M | 13GB | + 5GB | + 2GB | **20GB** |
| Q3_K_M | 10GB | + 5GB | + 2GB | **17GB** |
| Q2_K | 8GB | + 5GB | + 2GB | **15GB** |

**14.3GB VRAM is insufficient** even for Q2_K (lowest quality variant).

## Issues Fixed (But Still Not Enough)

### 1. GGUF Loading - FIXED ✅
- **Issue:** [diffusers #12981](https://github.com/huggingface/diffusers/issues/12981) - CLOSED
- **Fix:** [PR #12983](https://github.com/huggingface/diffusers/pull/12983) - Use `LTX2VideoTransformer3DModel`
- **Status:** GGUF loads successfully now
- **Note:** GGUF files download and load correctly, but OOM during generation

### 2. BitsAndBytes NF4 OOMs
- **Issue:** Caching allocator warmup tries to allocate full model before quantization
- **Error:** `torch.OutOfMemoryError: Allocation on device` during warmup
- **Status:** Happens even with `DISABLE_WARMUP=1`, `low_cpu_mem_usage=True`, and `max_memory` constraints
- **Requires:** 24GB+ VRAM for full model, or successful quantization (which fails due to warmup)

## Installation

Required packages (add to venv):
```bash
pip install 'gguf>=0.10.0' kernels
```

Already in requirements.txt but needs manual install if not using `./setup.sh --install`.

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

- **14.3GB VRAM (RTX 4070 Ti SUPER):** ❌ **INSUFFICIENT**
  - GGUF Q4_K_M (20GB): ❌ OOM (needs 20GB total)
  - GGUF Q3_K_M (17GB): ❌ OOM (needs 17GB total)
  - GGUF Q2_K (15GB): ❌ OOM (needs 15GB total)
  - BitsAndBytes NF4: ❌ OOM during warmup
  - **Conclusion:** LTX-2 requires minimum 15GB VRAM

- **15-16GB VRAM:** ⚠️ **Untested but should work**
  - GGUF Q2_K (15GB total) should fit
  - Quality will be lowest (Q2_K quantization)

- **17-19GB VRAM:** ⚠️ **Untested but should work**
  - GGUF Q3_K_M (17GB total) recommended
  - Better quality than Q2_K

- **20GB+ VRAM:** ⚠️ **Untested but should work**
  - GGUF Q4_K_M (20GB total) recommended
  - Best quality/VRAM balance

- **24GB+ VRAM:** ⚠️ **Untested**
  - Full precision BF16 (24GB+) available
  - Best quality but largest memory footprint

## Recommended Alternatives for <15GB VRAM

Since LTX-2 requires minimum 15GB VRAM, users with 14GB or less should use:

### 1. Wan FLF2V ⭐ (Recommended)
- **VRAM:** Works with 14GB VRAM
- **Quality:** AI-powered video generation with semantic understanding
- **Model:** Wan 2.2-TI2V-5B (~5GB)
- **Settings:** Guidance scale 3.5 for smooth transitions
- **Status:** Fully working and tested

### 2. FILM
- **VRAM:** Lightweight (<2GB)
- **Quality:** Google's Frame Interpolation for Large Motion
- **Speed:** Fast
- **Best for:** Dramatic scene changes
- **Status:** Fully working

### 3. Wait for Better VRAM Card
- LTX-2 GGUF files are already cached (12.8GB Q4_K_M)
- Will work immediately when you upgrade to 15GB+ VRAM
- No re-download needed

## Lessons Learned

1. **GGUF quantization cannot be combined with CPU offloading** - this is a fundamental limitation
2. **LTX-2 text encoder (Gemma) is large** (~5GB) - cannot be avoided
3. **Minimum VRAM = transformer + text_encoder + VAE** - all must fit simultaneously without offload
4. **BitsAndBytes also fails** due to warmup allocating full model before quantization
5. **ComfyUI may work differently** - uses different memory management, worth trying

## Next Steps / Future Work

1. ✅ **COMPLETED:** Identified GGUF/CPU offload incompatibility
2. ✅ **COMPLETED:** Documented VRAM requirements accurately
3. ⚠️ **TODO:** Test with 15GB+ VRAM hardware when available
4. ⚠️ **TODO:** Investigate if ComfyUI's GGUF loading works better (different memory approach)
5. ⚠️ **TODO:** Monitor diffusers for potential CPU offload fix for GGUF
6. ⚠️ **TODO:** Consider implementing direct GGUF loading without diffusers (complex)

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
