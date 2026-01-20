# 3-Day Checklist - Deforum Development Status
**Date Created:** 2026-01-20
**Next Review:** 2026-01-23 (3 days)

## ✅ Completed Today (70 commits pushed)

### Critical Fixes
- [x] **Render_mode persistence bug** - Fixed "Keyframes+Interpolation" falling back to "New 3D"
  - Location: `deforum/orchestration/run_deforum.py:71`
  - Impact: Prevented 3D infrastructure from loading unnecessarily

- [x] **Depth namespace bug** - Fixed depth.py checking wrong args object
  - Location: `deforum/rendering/helpers/depth.py:35`
  - Changed: `args.render_mode` → `anim_args.render_mode`

- [x] **File logging system** - Survives CUDA crashes
  - Location: `deforum/utils/system/logging/logger.py`
  - Files: `output/deforum_TIMESTAMP.log`

- [x] **Forge yellow log filter** - Maintains slopcore theme
  - Location: `deforum/utils/system/output_filter.py`
  - Applied: `preload.py` (early init)

### Major Features
- [x] **LTX-2 audio-video integration** - Full pipeline
  - Location: `deforum/integrations/ltx2/`
  - Features: Audio-guided conditioning, auto-download
  - Methods: Wan FLF2V, FILM, LTX-2

- [x] **Gaussian-splat merge** - 67 commits from feature branch
  - Two-Pass Refinement system
  - DA3-3DGS Tuning Lab (4 modes)
  - Resume feature, delta schedules

### Documentation
- [x] `FORGE_NEO_REGRESSION.md` - Detailed regression analysis
- [x] Commit messages with full context
- [x] 70 commits pushed to `origin/dev`

---

## ⏸️ Blocked by Forge Neo Regression

### Issue Summary
**Forge Neo commit:** `2e2f1071` (gguf)
**Problem:** ComfyUI backend memory corruption + VRAM exhaustion
**Impact:** Cannot test ANY generation (txt2img, img2img, Deforum)

### Evidence
```
VRAM State: LOW_VRAM
loaded completely; 95367431640625005117571072.00 MB usable  ← CORRUPTION
full load: True  ← --lowvram flag ignored
CUDA error: unspecified launch failure
```

### Blocked Tasks
- ⏸️ Testing render_mode persistence fix
- ⏸️ Testing depth namespace fix
- ⏸️ Testing LTX-2 interpolation quality
- ⏸️ Testing Wan FLF2V improvements
- ⏸️ Verifying Z-Image generations work
- ⏸️ All Task 2 (Wan improvements) testing

---

## 📋 3-Day Checklist (Recheck 2026-01-23)

### Step 1: Check Forge Neo Updates

```bash
cd /home/zirteq/workspace/forge-neo
git fetch origin
git log --oneline HEAD..origin/neo
```

**Look for commits mentioning:**
- [ ] "memory" / "vram" / "lowvram"
- [ ] "load" / "offload" / "partial"
- [ ] "z-image" / "lumina" / "cuda"
- [ ] "fix" / "regression" / "crash"

**If 5+ new commits:** Likely still mid-work, wait another day
**If 0-2 new commits:** May be abandoned, consider reporting bug upstream
**If commit mentions memory/vram:** Pull and test!

### Step 2: Pull and Test (If Promising Commits)

```bash
# Pull latest Forge Neo
git pull origin neo

# Restart WebUI
python webui.py

# Test PLAIN Forge txt2img (NOT Deforum yet)
# Model: z_image_turbo_bf16.safetensors
# Resolution: 720p
# Steps: 20
```

**Expected output (if fixed):**
```
VRAM State: NORMAL_VRAM (or LOW_VRAM)
loaded completely; 13xxx MB usable  ← Normal number
full load: False  ← Partial loading enabled
Moving model(s) has taken X.XX seconds
```

**Result:**
- [ ] ✅ Generation succeeds → Forge is fixed, proceed to Step 3
- [ ] ✗ Still crashes → Forge still broken, wait another 2-3 days

### Step 3: Test Deforum Fixes (Only if Step 2 Passes)

#### Test 1: Render Mode Persistence
```
1. Select: "Keyframes + Interpolation" mode
2. Generate with Deforum
3. Check log: Should show "Keyframes + Interpolation", NOT "New 3D"
4. Verify: No depth model loading messages
```

**Expected:**
```log
INFO: VERIFICATION: anim_args.render_mode = 'Keyframes + Interpolation'
DEBUG: → Skipping depth model (Keyframes + Interpolation mode doesn't use depth warping)
```

**Result:**
- [ ] ✅ Correct mode detected
- [ ] ✅ Depth model skipped
- [ ] ✗ Fallback to New 3D (bug still present)

#### Test 2: File Logging
```
1. Start generation
2. Force crash (Ctrl+C or wait for CUDA error)
3. Check: ls output/Deforum_*/deforum_*.log
4. Read log file
```

**Result:**
- [ ] ✅ Log file exists
- [ ] ✅ Contains full debug output
- [ ] ✗ Log file missing/incomplete

#### Test 3: LTX-2 Interpolation (Optional)
```
1. Mode: Keyframes + Interpolation
2. Interpolation method: LTX-2
3. Generate short animation (17 keyframes)
4. Check: Audio sync accuracy
```

**Result:**
- [ ] ✅ Interpolation works
- [ ] ✅ Audio sync <1 frame error
- [ ] ⏸️ Skipped (not priority)

### Step 4: Decision Tree

```
IF Forge is fixed AND all tests pass:
  ✅ Mark FORGE_NEO_REGRESSION.md as resolved
  ✅ Close GitHub issue (if created)
  ✅ Continue with Task 2 (Wan FLF2V improvements)
  ✅ Continue with Task 6 (animation_mode refactor)

ELSE IF Forge is fixed BUT tests fail:
  ⚠️ Our fixes have bugs, debug Deforum code
  ⚠️ Create new GitHub issues for Deforum bugs

ELSE IF Forge still broken after 3 days:
  ❌ Report to Forge Neo upstream with FORGE_NEO_REGRESSION.md
  ❌ Consider temporary rollback to backup-pre-comfy-merge
  ❌ Or wait another week for upstream fix
```

---

## 🎯 Next Tasks (When Forge is Stable)

### High Priority - Task 2: Wan FLF2V Improvements

**Location:** `deforum/rendering/keyframe_interp.py:233-298`

**Improvements to implement:**

1. **Adaptive Keyframe Strength** (Motion-based)
   - [ ] Implement `calculate_prompt_similarity()` using CLIP
   - [ ] Implement `adaptive_keyframe_strength()` function
   - [ ] Add UI controls: Min/Max strength sliders
   - [ ] Test: Dramatic changes ("city" → "forest") use LOW strength
   - [ ] Test: Similar prompts ("city day" → "city night") use HIGH strength

2. **Motion-Aware FLF2V Prompts**
   - [ ] Implement `analyze_movement_pattern()` from schedules
   - [ ] Implement `construct_motion_prompt()` for FLF2V
   - [ ] Implement `adaptive_flf2v_guidance()` (3.0-5.5 range)
   - [ ] Test: Verify motion descriptions are accurate

3. **Qwen Integration for FLF2V**
   - [ ] Wire Qwen prompt expander into FLF2V pipeline
   - [ ] Add checkbox: "Enable Qwen for FLF2V" (default: ON)
   - [ ] Test: Compare with/without Qwen enhancement

**Success Criteria:**
- [ ] SSIM scores improve over fixed strength baseline
- [ ] Motion descriptions match actual camera movement
- [ ] Qwen enhancement produces coherent interpolations

### Medium Priority - Task 6: Animation_Mode Refactor

**Problem:** Dual `render_mode`/`animation_mode` system causes bugs

**Plan:**
1. **Phase 1: Migration Layer**
   - [ ] Create `RenderModeAdapter` class
   - [ ] Map `render_mode` → internal representation
   - [ ] Deprecate `animation_mode` parameter

2. **Phase 2: Backend Updates**
   - [ ] Replace all `animation_mode` checks with `render_mode`
   - [ ] Update 3D rendering logic
   - [ ] Update keyframe distribution logic

3. **Phase 3: Cleanup**
   - [ ] Remove legacy `animation_mode` code
   - [ ] Update documentation
   - [ ] Migration guide for custom scripts

**Files to update:**
- [ ] `deforum/rendering/data/anim/animation_mode.py`
- [ ] `deforum/rendering/core.py`
- [ ] `deforum/rendering/flux_interp.py`
- [ ] `deforum/orchestration/run_deforum.py`

### Low Priority - Cleanup

- [ ] Review open GitHub issues
- [ ] Update CLAUDE.md with latest architecture
- [ ] Clean up dead code (vulture scan)
- [ ] Add missing type hints
- [ ] Run complexity analysis (radon cc)

---

## 📊 Current Status Summary

**Branch:** `dev` (70 commits ahead of `origin/dev` - PUSHED ✅)
**Forge Neo:** Broken (commit `2e2f1071`)
**Blocking Issue:** Memory corruption in ComfyUI backend
**ETA for Testing:** 3-7 days (waiting for upstream fix)

**Code Status:**
- ✅ All fixes implemented and committed
- ✅ LTX-2 integration complete
- ✅ Gaussian-splat merge complete
- ⏸️ Testing blocked by Forge regression

**Recommendation:**
- Wait 3 days, check Forge Neo updates
- If still broken, report upstream with full evidence
- If fixed, test all changes and continue Task 2

---

## 🔗 Quick Reference

**Key Files:**
- Regression doc: `FORGE_NEO_REGRESSION.md`
- This checklist: `TODO_3_DAY_CHECKLIST.md`
- Render mode fix: `deforum/orchestration/run_deforum.py:71`
- Depth namespace fix: `deforum/rendering/helpers/depth.py:35`
- File logging: `deforum/utils/system/logging/logger.py`
- LTX-2 pipeline: `deforum/integrations/ltx2/ltx2_pipeline.py`

**Forge Neo:**
- Repo: https://github.com/Haoming02/sd-webui-forge-classic/tree/neo
- Current commit: `2e2f1071` (gguf)
- Backup commit: `9d964f21` (pre-ComfyUI merge)

**Commands:**
```bash
# Check Forge updates
cd /home/zirteq/workspace/forge-neo && git fetch origin && git log --oneline HEAD..origin/neo

# Pull latest Forge
git pull origin neo

# Start with lowvram (when testing)
python webui.py --lowvram

# View latest Deforum log
ls -lt output/Deforum_*/deforum_*.log | head -1 | awk '{print $NF}' | xargs cat
```

---

**See you in 3 days! 🚀**
