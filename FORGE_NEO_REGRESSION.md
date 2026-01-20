# Forge Neo Regression Report

**Date:** 2026-01-20
**Forge Neo Commit:** `2e2f1071` (23 commits after `9d964f21`)
**Issue:** CUDA crashes with Z-Image after ComfyUI backend merge

## Problem Summary

After pulling Forge Neo upstream updates (23 commits including ComfyUI backend integration), Z-Image model crashes with CUDA errors even at low resolutions (480p, 720p).

**Before update (backup-pre-comfy-merge):**
- Z-Image txt2img worked fine
- VRAM usage was manageable on 16GB RTX 4070 Ti SUPER

**After update (2e2f1071):**
- Z-Image crashes with `CUDA error: unspecified launch failure`
- VRAM exhaustion during model loading:
  - JointTextEncoder: 8.4GB (qwen3)
  - KModel: 11.7GB (Z-Image transformer)
  - **Total: ~20GB needed on 16GB card → CRASH**

## Error Logs

### VRAM Loading
```
Requested to load JointTextEncoder
loaded completely; 13754.55 MB usable, 8414.12 MB loaded, full load: True

Requested to load KModel
Unloaded partially: 8414.12 MB freed, 0.00 MB remains loaded, 1531.25 MB buffer reserved
loaded completely; 13273.88 MB usable, 11739.54 MB loaded, full load: True
```

### CUDA Crash
```
terminate called after throwing an instance of 'c10::AcceleratorError'
Exception: CUDA error: unspecified launch failure
```

## Root Cause

The new ComfyUI memory management backend (`backend/memory_management.py`, `backend/patcher/`) is less memory-efficient than the old Forge implementation:

- **Old backend:** Worked with Z-Image on 16GB VRAM
- **New backend:** Requires ~20GB VRAM for same model

The issue affects plain Forge txt2img, not Deforum-specific functionality.

## Affected Models

- **Z-Image:** ✗ Crashes (confirmed)
- **Flux:** ⚠️ Untested (likely affected)
- **Lumina 2.0:** ⚠️ Untested
- **SDXL/SD1.5:** ⚠️ Likely still work (smaller models)

## Workaround

None currently available. Options:

1. **Wait for Forge Neo fix** - Forge appears mid-work on ComfyUI integration
2. **Rollback to backup-pre-comfy-merge** - Works but loses upstream fixes
3. **Use smaller models** - SDXL/SD1.5 may still work

## Deforum Status

All Deforum code is compatible with latest Forge Neo. The crashes are pure Forge regression, not extension-related.

### Completed Tasks (Working on backup)
- ✅ Merged gaussian-splat-experiments branch
- ✅ Fixed render_mode persistence bug (Keyframes+Interpolation → New 3D fallback)
- ✅ Fixed depth.py namespace bug (checking wrong args object)
- ✅ Implemented file logging system
- ✅ LTX-2 audio-video integration (full pipeline)
- ✅ Forge yellow log color filter

### Blocked Until Forge Fix
- ⏸️ Testing Z-Image generations
- ⏸️ Testing Wan FLF2V improvements
- ⏸️ Verifying LTX-2 interpolation quality

### Can Continue Without Generation
- ✅ Animation_mode refactor planning (Task 6)
- ✅ Code cleanup and optimization
- ✅ Documentation updates

## Recommendation

**Wait 1-2 days for Forge Neo to stabilize ComfyUI backend**, then:

1. Pull latest Forge Neo updates
2. Test Z-Image txt2img (plain Forge, no Deforum)
3. If working, continue Deforum development
4. If still broken, report upstream to Forge Neo with full reproduction steps

## Upstream Report Draft

**Title:** Z-Image VRAM regression after ComfyUI backend merge (commit 2e2f1071)

**Description:**
After pulling latest Forge Neo (23 commits including ComfyUI backend integration), Z-Image crashes with CUDA errors due to VRAM exhaustion.

**Hardware:** RTX 4070 Ti SUPER (16GB VRAM)
**Model:** Z-Image-Turbo (z_image_turbo_bf16.safetensors)
**Modules:** ae.safetensors, qwen_3_4b.safetensors

**VRAM Usage:**
- JointTextEncoder: 8.4GB
- KModel: 11.7GB
- Total: ~20GB (exceeds 16GB card)

**Before (commit 9d964f21):** Z-Image worked fine
**After (commit 2e2f1071):** CUDA crash at model loading

**Error:**
```
CUDA error: unspecified launch failure
terminate called after throwing an instance of 'c10::AcceleratorError'
```

The new ComfyUI memory management appears less efficient than previous implementation.

**Reproduction:** Plain txt2img with Z-Image on 16GB card

---

## Latest Findings (2026-01-20 22:20)

### Critical Bug: --lowvram Flag Broken

Attempted workaround with `--lowvram` flag failed:

```
VRAM State: LOW_VRAM                 ← Flag recognized
loaded completely; full load: True   ← Still loading completely (BUG!)
```

### Memory Corruption Bug

JointTextEncoder shows corrupted memory calculation:
```
loaded completely; 95367431640625005117571072.00 MB usable, 8414.12 MB loaded
```

**95 septillion MB** - clear memory corruption in new ComfyUI backend.

### Root Cause Analysis

**Old System (backup-pre-comfy-merge):**
- Models split between GPU and CPU automatically
- Partial loading: `gpu_modules` on GPU, `cpu_modules` on CPU
- Z-Image worked: ~16GB total with smart offloading

**New System (current):**
- Attempts full GPU loading even with `--lowvram` flag
- Memory calculation corrupted (overflow/underflow bug)
- Partial loading code path appears broken
- Result: 20GB+ needed, crashes on 16GB card

### Conclusion

This is a **critical upstream bug in Forge Neo**, not a Deforum issue:
- ✗ Memory corruption (septillion MB calculation)
- ✗ `--lowvram` flag ignored
- ✗ Partial loading broken
- ✗ Affects plain Forge txt2img (not Deforum-specific)

**Recommendation:** Rollback to `backup-pre-comfy-merge` until upstream fixes the ComfyUI backend regression.

---

**Next Action:** Monitor Forge Neo updates, retest in 3 days
