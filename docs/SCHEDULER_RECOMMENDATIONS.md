# Scheduler Recommendations for Deforum Render Modes

## Overview

Forge offers 14 different noise schedulers that determine **where timesteps are placed** during diffusion sampling. Even with discrete step counts, choosing the right scheduler can significantly impact quality, consistency, and temporal coherence.

## NEW: Fractional Strength Interpolation (1% Precision)

**Status:** ✅ Implemented and enabled by default

Deforum now includes **fractional strength interpolation** that provides 1% strength granularity instead of discrete levels:

| Mode | Steps | Without Fractional | With Fractional |
|------|-------|-------------------|-----------------|
| Flux Schnell | 4 | 0.25 (25%) | **0.01 (1%)** |
| Flux Dev | 20 | 0.05 (5%) | **0.01 (1%)** |
| Lumina 2.0 | 30 | 0.033 (3.3%) | **Disabled** (already fine) |

**How it works:**
- Uses Forge's `Prediction.sigma()` log-linear interpolation
- Generates fractional timesteps (e.g., 667.3 instead of rounding to 667 or 668)
- Automatically enabled for img2img at steps < 30
- Transparent: no user configuration needed
- Can be disabled: Settings → Deforum → "Enable fractional strength interpolation"

**Impact:**
- **Flux Schnell (4 steps):** Massive improvement - 25x finer control (25% → 1%)
- **Flux Dev (20 steps):** Significant improvement - 5x finer control (5% → 1%)
- **I2V chaining:** More precise continuity tuning
- **No performance penalty:** < 1% overhead

See `deforum/pipeline/fractional_strength.py` for implementation details.

## Available Schedulers

| Scheduler | Type | Characteristics | Best For |
|-----------|------|-----------------|----------|
| **Automatic** | Model-based | Uses model's native scheduler | Default behavior |
| **Karras** | K-Diffusion | Standard, well-balanced (rho=7.0) | Most workflows |
| **Exponential** | K-Diffusion | Smoother sigma progression | Long schedules |
| **Polyexponential** | K-Diffusion | Adjustable via rho parameter | Flexible tuning |
| **Normal** | Custom | Linear in timestep space | Stable generation |
| **Simple** | Custom | Samples from pre-computed sigmas | Fast sampling |
| **Uniform** | Custom | Even distribution | General use |
| **SGM Uniform** | Custom | Uniform in timestep space | SD tuning |
| **Linear Quadratic** | Custom | Linear → quadratic phases | **4-6 steps optimized** |
| **KL Optimal** | Information theory | KL divergence-based | Optimal coverage |
| **DDIM** | Classic | DDIM-style discrete sampling | Fast generation |
| **Align Your Steps** | NVIDIA Research | Model-aware optimal placement | Quality focus |
| **Beta** | Statistical | Beta distribution-based | Advanced tuning |
| **Turbo** | Fast | Speed-optimized | Ultra-fast |
| **Bong Tangent** | Parametric | Highly configurable curve | Custom control |

## Render Mode Recommendations

### Classic 3D

**Characteristics:**
- Fixed low cadence (default: 2)
- 24 FPS
- 20 steps (Flux Dev)
- Frequent diffusions every 2 frames
- Single strength schedule (normal strength)

**Recommended Schedulers:**

1. **Karras** (Default) ⭐
   - Well-balanced for frequent sampling
   - Proven stability across many frames
   - Good temporal coherence

2. **Normal**
   - Linear progression in timestep space
   - Very predictable behavior
   - Good for debugging

3. **Simple**
   - Fast and consistent
   - Minimal overhead
   - Good for long animations

**Avoid:**
- Linear Quadratic (designed for low steps, not needed at 20)
- Turbo (not needed for 20 steps)

---

### New 3D (Default)

**Characteristics:**
- Keyframes at prompt boundaries
- Cadence frames distributed between keyframes
- 60 FPS
- 20 steps (Flux Dev)
- Dual strength schedules:
  - Keyframes: LOW strength (0.15) → 17/20 steps denoising
  - Cadence: HIGH strength (0.85) → 3/20 steps denoising

**Recommended Schedulers:**

1. **Karras** (Default) ⭐
   - Handles both low and high strength well
   - Proven for mixed-strength workflows
   - Excellent temporal coherence

2. **Exponential**
   - Smoother transitions between keyframes
   - Good for high-motion sequences
   - Handles cadence frames well

3. **Normal**
   - Predictable behavior
   - Good balance
   - Simpler than Karras but still effective

**Avoid:**
- Linear Quadratic (not needed at 20 steps)
- KL Optimal (may not handle dual-strength as well)

---

### Keyframes Only

**Characteristics:**
- Only diffuse at keyframes
- All tweens interpolated via depth warping
- 60 FPS
- 20 steps (Flux Dev)
- Single strength schedule (keyframe strength, typically LOW for dramatic changes)

**Recommended Schedulers:**

1. **Karras** (Default) ⭐
   - Excellent keyframe quality
   - Well-tested
   - Reliable

2. **KL Optimal**
   - Information-theoretic optimal step placement
   - Maximizes quality per keyframe
   - Good for sparse diffusions

3. **Align Your Steps**
   - NVIDIA research-backed optimal placement
   - Pre-computed for quality
   - Good when keyframe quality is critical

**Avoid:**
- Linear Quadratic (not needed at 20 steps)
- Simple/Uniform (keyframes need highest quality)

---

### Flux + Interpolation

**Characteristics:**
- Phase 1: Flux/Lumina keyframes at prompt boundaries
- Phase 2: Interpolation (Wan FLF2V / RIFE / FILM)
- Phase 3: Stitch video
- 24 FPS (default)
- Variable steps: 4 (Schnell), 20 (Dev), 30 (Lumina)
- Single strength schedule (keyframe strength for Wan I2V chaining)

**Recommendations BY STEP COUNT:**

#### Flux Schnell (4 steps)

**Strength Resolution:** 0.25 (25% granularity)

1. **Linear Quadratic** ⭐⭐⭐
   - **SPECIFICALLY DESIGNED FOR 4-6 STEPS**
   - Optimal timestep placement at low counts
   - Smooth noise reduction
   - **HIGHLY RECOMMENDED**

2. **KL Optimal**
   - Information-theoretic optimal placement
   - Good at low step counts
   - Alternative to Linear Quadratic

3. **Align Your Steps**
   - Pre-computed optimal timesteps
   - Good quality at 4 steps
   - Second choice after Linear Quadratic

**Avoid at 4 steps:**
- Karras (not optimized for 4 steps)
- Exponential (needs more steps)
- Normal (mediocre at 4 steps)

#### Flux Dev (20 steps)

**Strength Resolution:** 0.05 (5% granularity)

1. **Karras** (Default) ⭐
   - Excellent all-around
   - Well-balanced
   - Proven quality

2. **Normal**
   - Predictable
   - Good keyframe quality
   - Simple and effective

3. **Exponential**
   - Smoother progression
   - Good for complex prompts
   - Alternative to Karras

**Avoid:**
- Linear Quadratic (not needed at 20 steps)

#### Lumina 2.0 (30 steps)

**Strength Resolution:** 0.033 (3.3% granularity)

**Note:** Lumina uses different scheduler internally (`linear_quadratic` is built-in).

1. **Karras** (Default) ⭐
   - Safe choice
   - Well-tested

2. **Exponential**
   - Smoother long schedules
   - Good for 30 steps

3. **Normal**
   - Reliable fallback

**Special Note:** Lumina's sampler may override scheduler selection. Test to verify.

---

## Step Count Impact on Strength Resolution

**Critical Understanding:**

```
Strength Resolution = 1 / steps
Actual Denoising Steps = int(denoising_strength * total_steps)
```

| Model | Steps | Resolution | Discrete Levels | Tuning Difficulty |
|-------|-------|------------|-----------------|-------------------|
| Flux Schnell | 4 | 0.25 | 4 | Hard |
| Flux Schnell | 6 | 0.167 | 6 | Moderate |
| Flux Dev | 20 | 0.05 | 20 | Easy |
| Lumina 2.0 | 30 | 0.033 | 30 | Very Easy |

**Example at 4 steps:**
- Strength 0.25 → 1 step denoising
- Strength 0.50 → 2 steps denoising
- Strength 0.75 → 3 steps denoising
- Strength 1.00 → 4 steps denoising

**No intermediate values!** Scheduler can only optimize WHERE those 1/2/3/4 steps sample.

---

## General Guidelines

### When to use Linear Quadratic
✓ **4-6 steps only** (Flux Schnell, custom low-step workflows)
✓ Speed-critical workflows with acceptable quality trade-offs
✓ Testing quick iterations

### When to use Karras
✓ **Default choice for 20+ steps**
✓ Proven temporal coherence
✓ Mixed-strength workflows (New 3D)
✓ Long animations requiring stability

### When to use KL Optimal
✓ Keyframes Only mode (sparse diffusions)
✓ Maximum quality per diffusion
✓ Low step counts (4-8) as alternative to Linear Quadratic

### When to use Align Your Steps
✓ Keyframe quality critical
✓ Willing to experiment
✓ Have VRAM for potential overhead

### When to use Exponential
✓ Long schedules (30+ steps)
✓ Smooth motion between keyframes
✓ Complex prompts with many changes

### When to use Normal/Simple/Uniform
✓ Debugging
✓ Predictable behavior needed
✓ Baseline comparisons

---

## Experimental Features

### Extra Noise Parameter
Even with discrete steps, you can add variation:
- **Location:** Settings → img2img → Extra noise
- **Range:** 0.0 - 1.0
- **Effect:** Adds noise at denoising start point
- **Use Case:** Compensate for coarse strength resolution at low steps

### Sigma Min/Max Override
Adjust noise schedule range:
- **Location:** Settings → Samplers
- **Effect:** "Zoom in" on portion of noise schedule
- **Use Case:** Fine-tune behavior at specific strength ranges

---

## Quick Reference Table

| Render Mode | Steps | Best Scheduler | Second Choice | Third Choice |
|-------------|-------|----------------|---------------|--------------|
| Classic 3D | 20 | Karras | Normal | Simple |
| New 3D | 20 | Karras | Exponential | Normal |
| Keyframes Only | 20 | Karras | KL Optimal | Align Your Steps |
| Flux + Interp (Schnell) | 4 | **Linear Quadratic** | KL Optimal | Align Your Steps |
| Flux + Interp (Dev) | 20 | Karras | Normal | Exponential |
| Flux + Interp (Lumina) | 30 | Karras | Exponential | Normal |

---

## Implementation Status

- ✅ All schedulers available in Forge backend
- ✅ Scheduler selection exposed in UI
- ⚠️ No per-mode default scheduler (uses Forge default)
- ⚠️ No UI hints about optimal scheduler per mode

**Future Enhancement:**
Consider adding per-mode scheduler recommendations in UI tooltips or auto-selecting optimal scheduler when mode changes.

---

## Testing Recommendations

When comparing schedulers:

1. **Use same seed** across tests
2. **Keep strength schedule constant**
3. **Test on 50-100 frame sequence** (not single frame)
4. **Evaluate temporal coherence** (not just single-frame quality)
5. **Check for jitter/flicker** in motion
6. **Verify prompt adherence** across keyframes

---

## References

- Forge scheduler implementations: `/home/zirteq/workspace/forge-neo/modules/sd_schedulers.py`
- Deforum render modes: `deforum/rendering/data/render_mode.py`
- Strength investigation: `/tmp/deforum_fractional_interpolation_investigation.md`
- Forge backend analysis: `/tmp/forge_strength_analysis.md`
