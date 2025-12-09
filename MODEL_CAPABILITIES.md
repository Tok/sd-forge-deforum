# Model Capabilities Reference

Comprehensive reference for model-specific features, parameters, and behaviors in Deforum.

## Quick Reference Table

| Model | Negative Prompts | Traditional CFG | Distilled CFG / Shift | Recommended Steps | CFG/Shift Range | Notes |
|-------|-----------------|-----------------|----------------------|-------------------|-----------------|-------|
| **Flux.1 Dev** | ⚠️ Via true_cfg | ⚠️ Via true_cfg | ✅ Dist CFG (3.5) | 50 (28 min) | dist: 3.5 default (>1 to enable) | Best quality, slower |
| **Flux.1 Schnell** | ❌ No | ❌ No (must be 0) | ❌ Must be 0 | 4 | N/A (guidance=0) | Fast, 1-8 steps max |
| **Lumina 2.0** | ✅ Yes | ✅ Yes (4.0) | ❌ Ignored | 30 | CFG: 4.0-5.5 | Anime-optimized, requires linear_quadratic |
| **Z-Image-Turbo** | ❌ No | ⚠️ Yes (1.0 only) | ✅ Shift (3.0) | 9 (8 passes) | shift: 1.0-5.0 | CFG=1.0 required, FlowMatch scheduler, Beta sigmas recommended |
| **SDXL** | ✅ Yes | ✅ Yes (7.5) | ❌ Ignored | 25 | CFG: 4.0-15.0 | Standard diffusion |
| **SD 1.5** | ✅ Yes | ✅ Yes (7.5) | ❌ Ignored | 25 | CFG: 4.0-15.0 | Classic SD |

## Optimal Settings Summary

| Model | Steps | Sampler | Scheduler | CFG | Dist CFG / Shift | Strength (Normal/KF) | Resolution |
|-------|-------|---------|-----------|-----|------------------|---------------------|------------|
| **Flux Dev** | 50 (28 min) | Euler | Simple | true_cfg:1.0 | dist:3.5 | 0.85 / 0.20 | 1024x1024 |
| **Flux Schnell** | 4 | Euler | Simple | 0.0 | 0.0 | 0.85 / 0.20 | 1024x1024 |
| **Lumina 2.0** | 30 | Euler | linear_quadratic | 4.0 | - | 0.85 / 0.20 | 1024x1024 |
| **Z-Image** | 9 (8 passes) | Euler | Beta | 1.0 | shift:3.0 | 0.85 / 0.20 | 1024x1024 |
| **SDXL** | 25 | DPM++ 2M | Normal | 7.5 | - | 0.85 / 0.20 | 1024x1024 |
| **SD 1.5** | 25 | DPM++ 2M | Normal | 7.5 | - | 0.85 / 0.20 | 512x512 |

**Note:** Strength values shown are for cadence frames (Normal) and keyframes (KF). Deforum uses inverted strength semantics (higher = more preservation).

## Resolution Recommendations

### Common 16:9 Resolutions (Landscape/Portrait)

All dimensions divisible by 8 (VAE requirement for most models):

| Aspect Ratio | Width × Height | Megapixels | Best For |
|--------------|---------------|-----------|----------|
| **16:9 Landscape** | 1280 × 720 | 0.92 MP | HD, standard deforum |
| **16:9 Landscape** | 1536 × 864 | 1.33 MP | Higher quality 16:9 |
| **16:9 Landscape** | 1920 × 1080 | 2.07 MP | Full HD (Flux max) |
| **9:16 Portrait** | 720 × 1280 | 0.92 MP | Mobile/vertical |
| **9:16 Portrait** | 864 × 1536 | 1.33 MP | Higher quality vertical |
| **9:16 Portrait** | 1080 × 1920 | 2.07 MP | Full HD vertical |

### Model-Specific Resolutions

**Flux.1 (Dev/Schnell):**
- **Native:** 1024 × 1024 (1:1)
- **Max:** 2.0 MP (e.g., 1920 × 1080 or 1408 × 1408)
- **16:9:** 1344 × 768 (1.0 MP), 1920 × 1088 (2.0 MP)
- **Divisibility:** 32 or 64 recommended (Flux architecture)
- **Trained range:** 0.2 to 2.0 megapixels, various aspect ratios

**Lumina 2.0:**
- **Native:** 1024 × 1024 (1:1)
- **Supported:** 768 × 1532, 968 × 1322, ≥ 1024
- **16:9:** 1216 × 684, 1536 × 864
- **Divisibility:** 8 (VAE requirement)

**Z-Image-Turbo:**
- **Native:** 1024 × 1024 (1:1, official recommendation)
- **Max:** 2048 × 2048
- **16:9:** 1216 × 684, 1536 × 864, 1280 × 720
- **Portrait:** 684 × 1216, 864 × 1536, 720 × 1280
- **Divisibility:** 8 (VAE requirement)

**SDXL:**
- **Native:** 1024 × 1024 (1:1)
- **16:9:** 1280 × 720, 1536 × 864
- **Divisibility:** 8 (VAE requirement)

**SD 1.5:**
- **Native:** 512 × 512 (1:1)
- **16:9:** 768 × 432, 640 × 360
- **Max recommended:** 768 × 768 (quality degrades beyond training resolution)
- **Divisibility:** 8 (VAE requirement)

**Technical Note:** Most models require dimensions divisible by 8 due to VAE architecture. Flux recommends divisibility by 32/64 for optimal results.

---

## Optimal Settings Presets

Deforum includes pre-configured preset files with optimal settings for each model. These presets are battle-tested configurations that work well for most use cases.

**Preset Locations:**
- `deforum/config/defaults/new_3d/flux_dev.json` - Flux Dev 3D animation
- `deforum/config/defaults/new_3d/flux_schnell.json` - Flux Schnell 3D animation
- `deforum/config/defaults/new_3d/lumina.json` - Lumina 2.0 3D animation
- `deforum/config/defaults/flux_interpolation/flux_dev.json` - Flux Dev + Interpolation mode

**Loading Presets:**
Load these via the Deforum UI or manually in your settings file to get optimal starting points for each model.

**Automatic Validation:**
Deforum validates your settings against each model's capabilities and displays warnings if:
- Steps are outside recommended range (e.g., using 20 steps with Flux Schnell when 4 is optimal)
- CFG scale is out of range or being used with models that ignore it
- Scheduler is incompatible (e.g., using 'normal' with Lumina when 'linear_quadratic' is required)
- Sampler is not in the compatible list for the model

Example validation output:
```
⚠️ Steps too high for Flux.1 Schnell: 20 > 8 (recommended: 4). This wastes computation without improving quality.
⚠️ Scheduler 'normal' may not work optimally with Lumina 2.0. Recommended: linear_quadratic
⚠️ Z-Image-Turbo does NOT support CFG at all (neither traditional nor distilled). Set cfg_scale=0.0. Uses in-prompt constraints instead of negative prompts.
```

**Validation Implementation:**
- **Location:** `deforum/config/model_configs.py:274` - `validate_settings()`
- **Called:** Automatically during `run_deforum()` before generation starts
- **Logged:** Warnings printed to console at start of generation

---

## Detailed Model Specifications

### Flux.1 Dev

**Type:** Flux (guidance-distilled diffusion)
**Display Name:** Flux.1 Dev
**Developer:** Black Forest Labs

**Parameters:**
- **Steps:** 50 recommended (range: 28-50, absolute min: 8)
- **Distilled CFG (guidance_scale):** 3.5 default (enable by setting > 1, higher = more prompt alignment but lower quality)
- **True CFG (true_cfg_scale):** 1.0 default (enable > 1 for negative prompts)
- **Scheduler:** `simple` (compatible: simple, normal, karras, exponential)
- **Sampler:** `euler` (compatible: euler, dpmpp_2m)
- **Resolution:** 1024x1024 (native)
- **Max Sequence Length:** 512 tokens

**Capabilities:**
- ⚠️ **Negative prompts:** Supported ONLY via `true_cfg_scale > 1` (advanced feature)
- ✅ **Distilled CFG guidance** (primary guidance mechanism)
- ✅ **True CFG guidance** (optional, for negative prompts)
- ✅ High quality output
- ✅ Strong text understanding
- ✅ Full img2img support

**Guidance System (Dual Mechanism):**

Flux Dev has TWO guidance modes:

1. **Embedded Guidance (`guidance_scale`)** - Default mode
   - Default: 3.5 (official Black Forest Labs default)
   - Enable by setting > 1
   - Higher values = more prompt alignment, lower quality
   - Guidance baked into distilled model
   - No negative prompts needed
   - Faster inference
   - Recommended for most use cases

2. **True CFG (`true_cfg_scale`)** - Advanced mode
   - Default: 1.0 (disabled)
   - Enable by setting > 1 (e.g., 3.0)
   - **Requires** `negative_prompt` parameter
   - Uses traditional classifier-free guidance
   - Slower (2x forward passes)
   - More precise control when you have good negative prompts

**Notes:**
- **Deforum implementation:** Uses embedded guidance (`guidance_scale=3.5`)
- **Negative prompts:** Currently not supported in Deforum (would require `true_cfg_scale` implementation)
- Official recommendation: 50 steps for best quality (28 minimum)
- Higher quality than Schnell but slower generation
- Best for final renders where quality matters

**UI Visibility:**
- Negative prompt field: Hidden (Deforum uses embedded guidance mode)
- Distilled CFG scale: Shown
- True CFG: Not implemented in Deforum (future enhancement)

**Optimal Settings (from preset):**
```json
{
  "steps": 50,
  "sampler": "Euler",
  "scheduler": "Simple",
  "cfg_scale_schedule": "0: (1.0)",  // Not used
  "distilled_cfg_scale_schedule": "0: (3.5)",
  "strength_schedule": "0: (0.85)",
  "keyframe_strength_schedule": "0: (0.20)",
  "W": 1024,
  "H": 1024
}
```

**References:**
- [Flux Pipeline - HuggingFace Diffusers](https://huggingface.co/docs/diffusers/en/api/pipelines/flux)
- [FLUX.1-dev Model Card](https://huggingface.co/black-forest-labs/FLUX.1-dev)

---

### Flux.1 Schnell

**Type:** Flux (timestep-distilled diffusion)
**Display Name:** Flux.1 Schnell
**Developer:** Black Forest Labs

**Parameters:**
- **Steps:** 4 recommended (range: 1-4, absolute max: 8)
- **Guidance Scale:** **MUST be 0.0** (timestep-distilled, no guidance)
- **True CFG:** Not supported (timestep-distilled model)
- **Scheduler:** `simple` (only compatible scheduler)
- **Sampler:** `euler` (only compatible sampler)
- **Resolution:** 1024x1024 (native)
- **Max Sequence Length:** 256 tokens (half of Dev)

**Capabilities:**
- ❌ **Negative prompts:** NOT supported (timestep-distilled)
- ❌ **Distilled CFG:** NOT supported (must be 0.0)
- ❌ **True CFG:** NOT supported
- ✅ Very fast generation (1-4 steps)
- ✅ Good quality at low steps
- ✅ Full img2img support

**Guidance System:**

Schnell is **timestep-distilled**, meaning:
- ALL guidance is baked into the model
- `guidance_scale` MUST be 0.0
- No negative prompts possible
- No CFG adjustments possible
- Trade-off: speed for flexibility

**Notes:**
- **Optimized for 1-4 steps** - More steps waste computation without improving quality
- Much faster than Dev but slightly lower quality
- Official documentation: "guidance_scale=0" (no guidance parameter)
- Best for previews, rapid iteration, or when speed is critical
- Not suitable for workflows requiring negative prompts or CFG tuning

**Strength Resolution:**
- 4 steps = 0.25 resolution (coarse control)
- Harder to tune for I2V chaining
- Consider using Flux Dev for strength-heavy workflows

**UI Visibility:**
- Negative prompt field: Hidden (not supported)
- Distilled CFG scale: Hidden/disabled (must be 0)
- True CFG: Not available

**Optimal Settings (from preset):**
```json
{
  "steps": 4,
  "sampler": "Euler",
  "scheduler": "Simple",
  "cfg_scale_schedule": "0: (0.0)",  // Must be 0
  "distilled_cfg_scale_schedule": "0: (0.0)",  // Must be 0
  "strength_schedule": "0: (0.85)",
  "keyframe_strength_schedule": "0: (0.20)",
  "W": 1024,
  "H": 1024,
  "max_sequence_length": 256
}
```

**Note:** With only 4 steps, strength resolution is 0.25 (coarse). Consider Flux Dev for workflows requiring fine strength control.

**References:**
- [Flux Pipeline - HuggingFace Diffusers](https://huggingface.co/docs/diffusers/en/api/pipelines/flux)
- [FLUX.1-schnell Model Card](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

---

### Lumina 2.0

**Type:** DiT (Diffusion Transformer)
**Display Name:** Lumina 2.0
**Developer:** Neta-Art (anime-optimized)

**Parameters:**
- **Steps:** 30 recommended (range: 20-50)
- **Traditional CFG:** 4.0 default (range: 4.0-5.5)
- **Distilled CFG:** Ignored
- **Scheduler:** `linear_quadratic` (REQUIRED, compatible: linear_quadratic, normal, karras)
- **Sampler:** `euler` (compatible: euler, dpmpp_2m, res_multistep)
- **Resolution:** 1024x1024 (native)

**Capabilities:**
- ✅ Negative prompts (fully supported via `negative_prompt` parameter)
- ✅ Traditional CFG guidance (classifier-free guidance)
- ❌ Distilled CFG (ignored)
- ✅ Anime-style optimization
- ✅ Full img2img support
- ✅ 1024x1024 native resolution
- ✅ Advanced CFG features (normalization, truncation)

**Notes:**
- REQUIRES `linear_quadratic` scheduler (compatibility patch applied automatically)
- Optimized for anime/illustration styles (Neta-Art fork)
- Uses traditional CFG with lower range (4.0-5.5) compared to SD (7.0-12.0)
- Official default: `guidance_scale=4.0` (HuggingFace Diffusers)
- Advanced features: `cfg_normalization=True`, `cfg_trunc_ratio=0.25`
- May produce suboptimal results for photorealistic content
- 2B parameters (Gemma-2B text encoder), efficient VRAM usage (≥8GB minimum)

**Known Issues:**
- `KeyError: 'num_tokens'` - Fixed via automatic compatibility patch in `deforum/integrations/lumina/compat_patch.py`
- Patch ensures `dynamic_args["num_tokens"]` is populated before sampling

**UI Visibility:**
- Negative prompt field: Shown (fully functional)
- Traditional CFG scale: Shown (4.0-5.5 range)
- Distilled CFG: Hidden (ignored)

**References:**
- [Lumina2 Pipeline - HuggingFace Diffusers](https://huggingface.co/docs/diffusers/en/api/pipelines/lumina2)
- [Neta-Lumina Model Card](https://huggingface.co/neta-art/Neta-Lumina)
- [Lumina-Image 2.0 GitHub](https://github.com/Alpha-VLLM/Lumina-Image-2.0)

---

### Z-Image-Turbo

**Type:** Distilled diffusion model (few-step, FlowMatch-based)
**Display Name:** Z-Image-Turbo
**Developer:** Alibaba Tongyi Lab

**Parameters:**
- **Steps:** 9 recommended (actual forward passes: 8, range: 4-20)
  - Speed mode: 4-6 steps (lower quality, faster)
  - Quality mode: 15-20 steps (higher quality, slower, diminishing returns)
- **Traditional CFG:** 1.0 (REQUIRED for prompt adherence - empirically confirmed)
  - **CRITICAL:** Despite documentation claiming CFG=0.0, CFG=1.0 is REQUIRED for prompts to work
  - Setting CFG=0.0 causes prompts to be completely ignored
  - Only CFG=1.0 works correctly in practice
- **Shift Parameter:** 3.0 (FlowMatch timestep schedule shift)
  - Controlled via `distilled_cfg_scale` in Forge (repurposed parameter)
  - Range: 1.0-5.0 (higher = more variation, lower = more consistent)
  - Purpose: Resolution-dependent noise scaling in FlowMatchEulerDiscreteScheduler
- **Scheduler:** `beta` recommended (Euler Beta - warmer colors, sharper micro-contrast)
  - Compatible: beta, simple, normal
  - Technical: FlowMatchEulerDiscreteScheduler with `use_beta_sigmas=True`
- **Sampler:** `euler` (compatible: euler, euler_a, dpmpp_2m, dpmpp_sde)
- **Resolution:** 1024x1024 (native, max 2048x2048)

**Capabilities:**
- ❌ Negative prompts (NOT supported - distilled model)
- ⚠️ CFG guidance (MUST be set to 1.0, despite documentation claiming 0.0 required)
- ✅ Shift parameter (FlowMatch timestep schedule tuning)
- ✅ Fast generation (few-step distilled via Decoupled-DMD)
- ✅ Full img2img support
- ✅ Good quality at low steps (8 forward passes)
- ✅ Strong instruction-following
- ✅ Sub-second latency on H800 GPUs, works on 16GB VRAM

**Scheduler System (FlowMatchEulerDiscreteScheduler):**

Z-Image uses a flow-matching scheduler with these parameters:

1. **Shift (`shift`)** - Default: 3.0
   - Controls timestep schedule scaling
   - Higher values = more noise variation/stylization
   - Lower values = more stable/consistent output
   - Forge implementation: Exposed via `distilled_cfg_scale` parameter

2. **Dynamic Shifting (`use_dynamic_shifting`)** - Default: False
   - When enabled: Adjusts shift based on image resolution on-the-fly
   - Z-Image keeps this disabled for consistent Turbo performance

3. **Base Shift (`base_shift`)** - Default: 0.5
   - Stabilizes image generation baseline
   - Increasing reduces variation and improves consistency

4. **Max Shift (`max_shift`)** - Default: 1.15
   - Maximum change allowed to latent vectors
   - Increasing encourages more variation/stylization

5. **Beta Sigmas (`use_beta_sigmas`)** - Recommended: True
   - Enables Euler Beta scheduler variant
   - Results in warmer colors and slightly sharper micro-contrast
   - Popular community preference for Z-Image

**Notes:**
- Distilled model using Decoupled-DMD
- **CRITICAL CFG REQUIREMENT:** Must set `cfg_scale=1.0` (empirically confirmed, contradicts documentation)
  - Official documentation claims CFG=0.0 required, but this is INCORRECT
  - CFG=0.0 causes prompts to be completely ignored
  - Only CFG=1.0 provides proper prompt adherence
  - This discrepancy is a known issue with Z-Image-Turbo documentation
- Use in-prompt constraints instead of negative prompts (e.g., "no watermark", "plain background")
- Works best with long, detailed prompts (recommended: LLM enhancement before generation)
- 1024x1024 native resolution (max 2048x2048, must be divisible by 8)
- Official steps: 9 (results in 8 actual DiT forward passes due to scheduler implementation)
- Recommended data type: `torch.bfloat16` for optimal performance

**Forge-Specific Implementation:**
- `distilled_cfg_scale` parameter repurposed to control `shift` value
- Retrieved as: `shift = getattr(prompt, 'distilled_cfg_scale', 3.0)`
- This allows runtime adjustment while maintaining backward compatibility

**UI Visibility:**
- Negative prompt field: Hidden/grayed (not supported)
- Traditional CFG scale: Shown (MUST be set to 1.0)
- Distilled CFG scale: Shown as "Shift" (controls FlowMatch shift parameter, NOT CFG!)

**References:**
- [Z-Image-Turbo Official Model Card](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- [FlowMatchEulerDiscreteScheduler Documentation](https://huggingface.co/docs/diffusers/api/schedulers/flow_match_euler_discrete)
- [ComfyUI Z-Image-Turbo Configuration](https://github.com/erosDiffusion/ComfyUI-EulerDiscreteScheduler/blob/master/Z-IMAGE-TURBO.md)
- [Forge Neo Z-Image Implementation](https://github.com/Haoming02/sd-webui-forge-classic/commit/93fd1ab26b3b376687a91ca99cf7862255ace064)
- [HuggingFace Discussion: Z-Image-Turbo does not use negative prompts](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/discussions/8)
- [Official Prompting Guide](https://gist.github.com/illuminatianon/c42f8e57f1e3ebf037dd58043da9de32)

---

### SDXL

**Type:** Stable Diffusion XL
**Display Name:** SDXL
**Developer:** Stability AI

**Parameters:**
- **Steps:** 25 recommended (range: 15-50)
- **Traditional CFG:** 7.5 (range: 4.0-15.0)
- **Distilled CFG:** Ignored
- **Scheduler:** `normal` (compatible: normal, karras, exponential, simple)
- **Sampler:** `dpmpp_2m` (compatible: euler_a, dpmpp_2m, dpmpp_2m_sde, ddim)

**Capabilities:**
- ✅ Negative prompts (fully supported)
- ✅ Traditional CFG guidance
- ❌ Distilled CFG (ignored)
- ✅ High resolution (1024x1024 native)
- ✅ Full img2img support
- ✅ Excellent quality

**Notes:**
- Standard Stable Diffusion architecture
- Uses traditional CFG in 4.0-15.0 range
- 1024x1024 native resolution
- Widely compatible with LoRAs and extensions

**UI Visibility:**
- Negative prompt field: Shown
- Traditional CFG scale: Shown
- Distilled CFG: Hidden (ignored)

---

### SD 1.5

**Type:** Stable Diffusion 1.5
**Display Name:** SD 1.5
**Developer:** Stability AI

**Parameters:**
- **Steps:** 25 recommended (range: 15-50)
- **Traditional CFG:** 7.5 (range: 4.0-15.0)
- **Distilled CFG:** Ignored
- **Scheduler:** `normal` (compatible: normal, karras, exponential, simple)
- **Sampler:** `dpmpp_2m` (compatible: euler_a, dpmpp_2m, dpmpp_2m_sde, ddim)

**Capabilities:**
- ✅ Negative prompts (fully supported)
- ✅ Traditional CFG guidance
- ❌ Distilled CFG (ignored)
- ✅ 512x512 native resolution
- ✅ Full img2img support
- ✅ Massive ecosystem of LoRAs/embeddings

**Notes:**
- Classic Stable Diffusion model
- Same parameters as SDXL but 512x512 resolution
- Huge community ecosystem
- Best compatibility with existing workflows

**UI Visibility:**
- Negative prompt field: Shown
- Traditional CFG scale: Shown
- Distilled CFG: Hidden (ignored)

---

## Feature Support Matrix

### Negative Prompts

**Supported:**
- ✅ Lumina 2.0
- ✅ SDXL
- ✅ SD 1.5

**NOT Supported:**
- ❌ Flux.1 Dev (distilled architecture, no CFG)
- ❌ Flux.1 Schnell (distilled architecture, no CFG)
- ❌ Z-Image-Turbo (distilled few-step model, no CFG)

**Why distilled models don't support negative prompts:**
Flux and Z-Image-Turbo use distilled diffusion processes that don't have a separate unconditional path for negative guidance. The model architectures fundamentally don't support CFG-style negative prompting. For Z-Image, use in-prompt constraints like "no watermark", "plain background" instead.

---

### CFG Guidance Systems

**Traditional CFG (Classifier-Free Guidance):**
- ✅ Lumina 2.0 (4.0-5.5)
- ⚠️ Z-Image-Turbo (MUST be 1.0 - empirically required despite docs saying 0.0)
- ✅ SDXL (4.0-15.0)
- ✅ SD 1.5 (4.0-15.0)

**Distilled CFG (Flux-specific):**
- ✅ Flux.1 Dev (1.0-10.0, default 3.5)
- ✅ Flux.1 Schnell (1.0-10.0, default 3.5)

**How Distilled CFG differs:**
- Uses single forward pass instead of conditional + unconditional
- Guidance is "baked into" the model during distillation
- Faster than traditional CFG (no dual forward pass)
- Different scale range and meaning

---

### Scheduler Requirements

**Flexible (most schedulers work):**
- Flux.1 Dev: simple, normal, karras, exponential
- SDXL: normal, karras, exponential, simple
- SD 1.5: normal, karras, exponential, simple

**Restricted (limited scheduler compatibility):**
- Flux.1 Schnell: `simple` only
- Z-Image-Turbo: simple, normal

**Specific Requirement:**
- Lumina 2.0: `linear_quadratic` REQUIRED (compatibility patch handles this)

---

### Strength Resolution (Steps Impact)

**High Resolution (fine control):**
- Flux.1 Dev: 20 steps = 1/20 = 0.05 resolution
- SDXL/SD 1.5: 25 steps = 1/25 = 0.04 resolution
- Lumina 2.0: 30 steps = 1/30 = 0.033 resolution

**Medium Resolution:**
- Z-Image-Turbo: 9 steps = 1/9 = 0.11 resolution

**Low Resolution (coarse control):**
- Flux.1 Schnell: 4 steps = 1/4 = 0.25 resolution

**Note:** With fractional strength patches (always enabled), all models get 0.01 (1%) precision regardless of steps.

---

## Implementation Details

### Model Detection

**Location:** `deforum/utils/model_detection.py`

**Functions:**
- `is_flux_model()` - Detects both Flux Dev and Schnell
- `is_lumina_model()` - Detects Lumina 2.0
- `is_zimage_model()` - Detects Z-Image-Turbo
- `is_sdxl_model()` - Detects SDXL
- `get_current_model_name()` - Returns friendly model name

**Detection Methods:**
1. Model class name (most reliable)
2. Checkpoint filename patterns
3. Full path patterns
4. Fallback to "Unknown"

---

### Model Configs

**Location:** `deforum/config/model_configs.py`

**Structure:**
```python
ModelConfig(
    model_type: str
    display_name: str
    recommended_steps: int
    min_steps: int
    max_steps: int
    uses_cfg: bool  # Traditional CFG
    cfg_scale_default: float
    cfg_scale_min: float
    cfg_scale_max: float
    uses_distilled_cfg: bool  # Flux distilled CFG
    distilled_cfg_scale_default: float
    distilled_cfg_scale_min: float
    distilled_cfg_scale_max: float
    recommended_scheduler: str
    compatible_schedulers: List[str]
    recommended_sampler: str
    compatible_samplers: List[str]
    notes: str
)
```

**Functions:**
- `get_model_config(model_name)` - Get config for current model
- `validate_settings(args, model_name)` - Validate settings against config
- `log_model_config(model_name)` - Log detected config to console

---

### UI Adaptation

**Negative Prompt Visibility:**
```python
# deforum/orchestration/generate.py:323
model_ignores_negative = is_flux_model() or is_lumina_model()

# Only print if model uses negatives and prompt is not empty
if not model_ignores_negative:
    if neg_prompt and neg_prompt.strip():
        logger.info(f"Neg Prompt: {neg_prompt}")
```

**Distilled CFG Column:**
```python
# deforum/orchestration/generate.py:382-385
# Only show Distilled CFG for Flux models
if is_flux_model():
    columns.append("Dist. CFG")
    values.append(str(p.distilled_cfg_scale))
```

---

## Testing Model-Specific Behavior

### Flux Models
```python
# Should show distilled CFG, hide traditional CFG
assert is_flux_model() == True
assert config.uses_distilled_cfg == True
assert config.uses_cfg == False
```

### Z-Image/Lumina
```python
# Should show traditional CFG, hide distilled CFG
assert config.uses_cfg == True
assert config.uses_distilled_cfg == False
```

### Negative Prompts
```python
# Flux should ignore negative prompts
if is_flux_model():
    assert model_ignores_negative == True
    # Negative prompt field should not be shown/logged
```

---

## Adding New Models

To add support for a new model:

1. **Add detection in `deforum/utils/model_detection.py`:**
   ```python
   def is_new_model() -> bool:
       """Detect if NewModel is loaded."""
       class_name = _get_model_class_name(model)
       return class_name == 'NewModelClass'
   ```

2. **Add config in `deforum/config/model_configs.py`:**
   ```python
   "new_model": ModelConfig(
       model_type="new_model",
       display_name="New Model Name",
       # ... all parameters
   )
   ```

3. **Update detection in `detect_model_type_extended()`:**
   ```python
   if is_new_model():
       return "new_model"
   ```

4. **Update UI logic if needed:**
   - Negative prompt visibility
   - CFG column display
   - Any model-specific UI elements

5. **Update this document** with new model's capabilities!

---

## Changelog

- **2025-01-08:** Initial documentation
  - Documented all 6 supported models
  - Feature support matrix
  - Implementation details
  - Testing guidelines
