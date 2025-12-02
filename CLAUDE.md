# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

**sd-forge-deforum** is an experimental fork of the Deforum extension for Stable Diffusion WebUI Forge that generates frame-precise animated videos using keyframe scheduling. This fork adds:
- **Flux.1 support** for state-of-the-art image generation
- **Lumina 2.0 support** (Forge Neo only) - 2B parameter model with 1024x1024 native resolution
- **Wan 2.1 AI Video Generation** (Alibaba's text-to-video model) with Deforum scheduling integration
- **Parseq keyframe redistribution** for intelligent frame placement
- **Camera Shakify integration** for realistic camera shake effects from Blender data
- **QwenPromptExpander** for AI-powered prompt enhancement and movement analysis
- **Reverse Generation** for stable forward-motion clips (POV, dash-cam, body-cam footage)

The extension operates as a Forge extension that hooks into WebUI's script system to provide animation capabilities across 3 modes: 3D (default), Flux + Interpolation, and Interpolation.

## Running the Extension

### Prerequisites

Must be installed in a working Stable Diffusion WebUI Forge installation. See parent repository CLAUDE.md for Forge setup.

### Testing the Extension

**Quick test with default settings:**
```bash
# From Forge webui directory, launch normally
python launch.py
# Or if environment already prepared:
python webui.py
```

Then in the UI:
1. Navigate to Deforum → Init → Quick Test
2. Click "Generate Test" to create a 5-second test video with synthetic audio and simple camera movement

**Run with Deforum API enabled:**
```bash
python webui.py --deforum-api
```

**Run tests:**
```bash
# From extension directory
pytest tests/ --start-server
# Or if server already running:
pytest tests/
```

**Install/update dependencies:**
```bash
cd extensions/sd-forge-deforum
./setup.sh --prepare  # First-time setup (PyTorch + SageAttention + Deforum deps)
# OR
./setup.sh --install  # Just install Deforum requirements.txt
```

### Setup Scripts

**Location:** Root directory and `shell_scripts/`

**First-Time Setup:**
```bash
./setup.sh --prepare
```
This automated script:
1. Installs PyTorch via Forge (using `launch.py --exit`)
2. Attempts to install SageAttention (requires CUDA toolkit, optional)
3. Installs all Deforum dependencies

**Manual SageAttention Installation:**
```bash
./shell_scripts/install-cuda-toolkit.sh    # Install CUDA toolkit (~3GB)
source ~/.bashrc                           # Reload environment
./shell_scripts/install-sageattention.sh   # Compile SageAttention
```

**SageAttention Details:**
- Requires CUDA toolkit (nvcc compiler) to build from source
- `setup.sh --prepare` attempts installation but continues if it fails
- Manual installation scripts handle full CUDA toolkit setup + compilation
- Enables `--sage` optimization flag for RTX 30/40/50 GPUs

**Model Downloads:**
```bash
./shell_scripts/download-all-models.sh     # Interactive: Flux, Lumina, Z-Image, Wan, etc.
```

**Script Organization:**
- **Root:** `setup.sh/bat`, `start-forge.sh/bat` (quick access)
- **`shell_scripts/`:** Download, install, test, and launch scripts
- **`scripts/`:** Python development tools only

## Architecture

### Entry Points and Flow

1. **Extension Loading** (`preload.py:18`)
   - Registers CLI arguments: `--deforum-api`, `--deforum-simple-api`, `--deforum-run-now`, `--deforum-terminate-after-run-now`
   - Called by Forge before main initialization

2. **Extension Initialization** (`scripts/deforum.py:24`)
   - Extends Python path with `deforum_sys_extend()`
   - Applies diffusers compatibility patches for Forge integration
   - Creates `Models/Deforum/` directory for model downloads
   - Registers UI tabs via `script_callbacks.on_ui_tabs(on_ui_tabs)`
   - Registers settings via `script_callbacks.on_ui_settings(on_ui_settings)`

3. **UI Creation** (`scripts/deforum_helpers/ui_elements.py`, `scripts/deforum_helpers/ui_left.py`)
   - Builds Gradio interface with main tabs: Run, Keyframes, Distribution, Prompts (with AI Enhancement accordion), Shakify, 3D Depth, Init, Wan Models, Output
   - Keyframes tab has flattened single-level navigation (Motion, Strength, CFG, Seed, etc.)
   - Distribution tab promoted to main level for render mode selection
   - Shakify and 3D Depth promoted from buried subtabs to dedicated main tabs
   - Returns tuple: `(deforum_interface, "Deforum", "deforum")`

4. **Main Orchestrator** (`scripts/deforum_helpers/run_deforum.py:43`)
   - `run_deforum(*args)` is the primary entry point when user clicks generate
   - Parses component arguments from UI into structured objects
   - Detects Wan Video mode to skip SD model loading if not needed
   - Processes arguments via `process_args()` into `args`, `anim_args`, `video_args`, `parseq_args`, etc.
   - Routes to appropriate rendering pipeline based on animation mode

5. **Rendering Pipelines**
   - **Standard mode (3D/Interpolation):** `deforum/rendering/core.py:22` - `render_animation()`
   - **Flux + Interpolation mode:** `deforum/rendering/flux_interp.py` - `render_wan_flux()` - Flux keyframes + choice of interpolation (Wan/RIFE/FILM)

### Core Systems

**1. Keyframe Distribution System** (`scripts/deforum_helpers/rendering/data/frame/`)
- **Purpose:** Intelligently places keyframes across animation timeline without cadence
- **Key Files:**
  - `key_frame_distribution.py` - Distribution algorithms
  - `diffusion_frame.py` - Frame metadata and state
  - `diffusion_frame_data.py` - Collection of all frames
  - `tween_frame.py` - Interpolated frames between keyframes
- **Integration:** Works with or without Parseq for precise timing

**2. Animation Pipeline** (`deforum/rendering/core.py`)
- Central render loop that:
  1. Creates `RenderData` object (central state container)
  2. Generates subtitle .srt file asynchronously
  3. Iterates through frames calling `process_frame()`
  4. Applies transformations (2D/3D movement, depth warping)
  5. Handles masks, noise schedules, and color coherence
  6. Stitches final video with ffmpeg
- **Note:** Legacy/stable core has been removed - render core is now the only render pipeline

**3. Wan Video Pipeline** (`scripts/deforum_helpers/wan/`, `scripts/deforum_helpers/rendering/`)
- **wan_simple_integration.py** - Wan FLF2V wrapper and utilities
  - Auto-discovers models from `models/Deforum/wan/` directory
  - Handles T2V (text-to-video), I2V (image-to-video), and FLF2V (first-last-frame-to-video)
  - Calculates frame counts as 4n+1 per Wan requirements
  - Integrates Deforum prompt scheduling, FPS, seed, and strength
- **flux_interp.py** - Flux + Interpolation mode pipeline with multi-method support
  - Phase 1: Generate ALL keyframes with Flux
  - Phase 2: Interpolate tweens with selected method (Wan FLF2V / FILM)
  - Phase 3: Stitch final video
  - Supports two interpolation methods via `flux_flf2v_interpolation_method` parameter:
    - **Wan FLF2V (default):** AI-generated video with semantic understanding
    - **FILM:** Google's Frame Interpolation for Large Motion (handles dramatic changes)
  - **Note:** RIFE is NOT available here (defaults to single frames on dramatic changes). RIFE is available for post-processing smooth videos only.
- **qwen_prompt_expander.py** - AI prompt enhancement (integrated into Prompts tab)
  - Auto-selects Qwen model (3B/7B/14B) based on VRAM
  - Analyzes Deforum movement schedules, translates to English
  - Lazy-loads models only when "Enhance Prompts" clicked
  - Auto-cleanup before video generation to free VRAM

**4. Central State** (`scripts/deforum_helpers/rendering/data/render_data.py:42`)
- `RenderData` class holds all state during rendering:
  - Frame metadata, animation keys, schedules
  - Depth models, masks, images
  - Parseq integration data
  - Camera shake patterns
  - Progress tracking

**5. Depth Estimation** (`scripts/deforum_helpers/depth*.py`)
- Uses Depth-Anything V2 for depth estimation (legacy depth models removed: MiDaS, AdaBins, LeReS, ZoeDepth)
- Used for 3D mode to warp frames based on estimated depth
- Models auto-download to `models/Deforum/` on first use

**6. Camera Shakify** (`scripts/deforum_helpers/rendering/data/shakify/`)
- Pre-recorded camera shake patterns from Blender
- Patterns: EARTHQUAKE, FILM_GRAIN, GENTLE_HANDHELD, INVESTIGATION, etc.
- Applied on top of scheduled movement transforms
- Data sourced from EatTheFuture's Camera Shakify Blender plugin (CC0 license)

**7. Parseq Integration** (`scripts/deforum_helpers/parseq_adapter.py`)
- Adapter pattern to integrate Parseq keyframe data
- Translates between Parseq JSON format and Deforum's internal structures
- Enables complex scheduling with GUI-based keyframe editor

### Directory Structure

```
scripts/
├── deforum.py                          # Main extension script (init)
├── deforum_api.py                      # REST API endpoints
├── deforum_api_models.py               # API data models
├── deforum_controlnet.py               # ControlNet integration
├── default_settings.txt                # Default configuration template
└── deforum_helpers/                    # Core implementation
    ├── run_deforum.py                  # Main orchestrator
    ├── args.py                         # Argument parsing
    ├── defaults.py                     # Default values
    ├── ui_right.py                     # UI construction
    ├── ui_settings.py                  # Settings tab
    ├── prompt.py                       # Prompt scheduling
    ├── animation.py                    # Animation calculations
    ├── depth*.py                       # Depth estimation backends
    ├── parseq_adapter.py               # Parseq integration
    ├── wan/                            # Wan video generation
    │   ├── wan_simple_integration.py   # Main Wan pipeline
    │   ├── qwen_prompt_expander.py     # AI prompt enhancement
    │   └── ...
    ├── rendering/                      # Rendering pipeline
    │   ├── core.py        # Main render loop
    │   ├── data/                       # Data structures
    │   │   ├── render_data.py          # Central state
    │   │   ├── frame/                  # Frame systems
    │   │   ├── shakify/                # Camera shake data
    │   │   └── subtitle/               # Subtitle generation
    │   └── util/                       # Rendering utilities
    └── src/                            # Third-party code
        ├── adabins/                    # AdaBins depth model
        └── clipseg/                    # CLIPSeg segmentation

tests/
├── conftest.py                         # Pytest configuration
├── deforum_test.py                     # Main test suite
└── utils.py                            # Test utilities

preload.py                              # CLI argument registration
requirements.txt                        # Python dependencies
pytest.ini                              # Pytest settings
```

### Key Concepts

**Render Modes (New 4-Mode System):**
The extension now uses a unified `RenderMode` system that replaces the old animation_mode + keyframe_distribution combinations. Each mode has specific characteristics and default settings:

1. **Classic 3D** - Traditional Deforum with fixed low cadence
   - Keyframe distribution: OFF (uniform cadence)
   - Strength schedules: Single (normal strength only)
   - Default: 24 FPS, cadence=2, 20 steps
   - Best for: RAFT, ControlNet, maximum stability
   - Shows: 3D tabs (Depth, Shakify), real cadence slider

2. **New 3D** (Default) - Modern keyframe redistribution
   - Keyframe distribution: REDISTRIBUTED
   - Strength schedules: Dual (normal + keyframe strength)
   - Default: 60 FPS, cadence=5, 20 steps
   - Best for: Balanced quality, speed, and stability
   - Shows: 3D tabs (Depth, Shakify), real cadence slider

3. **Keyframes Only** - Pure keyframe diffusion with depth tweening
   - Keyframe distribution: KEYFRAMES_ONLY
   - Strength schedules: Single (keyframe strength only)
   - Default: 60 FPS, pseudo-cadence display, 20 steps
   - Best for: Fastest rendering, slow movements, pure depth transforms
   - Shows: 3D tabs (Depth, Shakify), pseudo-cadence display (read-only)
   - Not compatible with: RAFT, ControlNet (too many non-diffused frames)

4. **Flux + Interpolation** - Flux + Multi-Method Interpolation workflow
   - Keyframe distribution: None (separate Flux + Interpolation pipeline)
   - Strength schedules: Single (keyframe strength for I2V chaining with Wan)
   - Default: 24 FPS, pseudo-cadence display, 20 steps
   - Best for: Dramatic changes, choice of interpolation method
   - Shows: Interpolation tab (if using Wan), Flux Interpolation Settings, pseudo-cadence display (read-only)
   - Hides: 3D tabs (Depth, Shakify, RAFT, ControlNet)
   - Phase 1: Generate ALL keyframes with Flux at prompt boundaries
   - Phase 2: Interpolate tweens with selected method (Wan FLF2V / FILM)
   - Phase 3: Stitch final video
   - Integrated Qwen prompt enhancement
   - Two interpolation methods available:
     - **Wan FLF2V (default):** AI video generation with semantic understanding (guidance_scale=3.5)
     - **FILM:** Google's Frame Interpolation for Large Motion (handles dramatic changes)
   - **Note:** RIFE is NOT available here (defaults to single frames on dramatic changes). RIFE is available for post-processing smooth videos only.

**Mode Selection Impact:**
- Top-level UI controls adapt based on selected mode
- FPS, steps, and cadence/pseudo-cadence visibility auto-adjust
- Strength sliders (1 or 2) show/hide based on mode requirements
- Tab visibility (3D vs Wan) changes automatically

### Strength System

**Deforum uses INVERTED strength semantics** (opposite of standard img2img):

| Deforum Strength | Meaning | Forge Conversion | Actual Steps (20 total) |
|-----------------|---------|------------------|------------------------|
| **0.0** | No preservation, full regeneration | `1.0 - 0.0 = 1.0` | 20/20 steps |
| **0.20** | Low preservation, dramatic changes | `1.0 - 0.2 = 0.8` | 16/20 steps |
| **0.85** | High preservation, stability | `1.0 - 0.85 = 0.15` | 3/20 steps |
| **1.0** | Maximum preservation | `1.0 - 1.0 = 0.0` | 0/20 steps |

**Conversion:** `deforum/pipeline/webui_sd_pipeline.py:50`
```python
p.denoising_strength = 1 - args.strength
```

**Why Inverted?**
- Deforum strength = "how much to preserve from previous frame"
- Standard img2img = "how much to change"
- Higher Deforum value = MORE preservation = LESS diffusion work

**Typical Values:**
- **Keyframes (0.15-0.30):** Low preservation → 14-17/20 steps → Dramatic changes
- **Cadence (0.80-0.90):** High preservation → 2-4/20 steps → Smooth stability
- **Default:** Normal=0.85, Keyframe=0.20

**Fractional Strength Precision (Always Enabled):**
Fractional strength interpolation is now **always enabled** via automatic monkey patches applied at extension init, providing 1% precision for all strength values:
- **Without**: Resolution = 1/steps (e.g., 0.05 at 20 steps, 0.25 at 4 steps) - coarse tuning
- **With**: Resolution = 0.01 (1%) regardless of step count - fine-grained control
- **Why it matters**: Enables precise I2V chaining tuning, especially critical for Flux Schnell (4 steps)
- **Implementation**:
  - Monkey patches: `deforum/pipeline/fractional_img2img_patch.py`, `fractional_sigma_slicer_patch.py`
  - Core logic: `deforum/pipeline/fractional_strength.py`
  - Applied automatically at extension load (no user action needed)
- **Tuning impact**: All automated tuning tests use 0.01 step size to leverage fractional precision
  - See `TUNING.md` for comprehensive parameter sweep configurations
  - Enables meaningful testing of Flux Schnell viability at 4 steps

**Keyframe Distribution:**
- Replaces traditional cadence-based rendering
- Intelligently places diffusion keyframes at prompt boundaries
- Interpolates (tweens) non-keyframes from nearest keyframes
- Reduces diffusion steps while maintaining quality

**Render Core:**
- Render core is now the **only** render core (legacy/stable core has been removed)
- Now integrated into the 4 render modes (see above) instead of separate distribution selection
- Keyframe distribution modes (internal): OFF, KEYFRAMES_ONLY, REDISTRIBUTED (ADDITIVE removed)
- Incompatible with some features (Kohya HR Fix, FreeU, ControlNet)
- Provides better synchronization and less jitter at high/no cadence

**Wan Integration Points:**
- Prompts from "Prompts" tab with frame numbers
- AI Prompt Enhancement in "Prompts" tab → "AI Prompt Enhancement" accordion (Qwen)
- FPS from "Output" tab
- Optional seed scheduling from "Keyframes → Seed" tab
- Optional strength scheduling from "Keyframes → Strength" tab for I2V chaining
- Model selection and configuration in "Wan Models" tab
- FLF2V settings in "Distribution" tab (for 2D/3D modes) or "Wan Models" tab (for Wan Only/Flux modes)

**I2V Chaining (Wan):**
- Uses last frame of previous clip as initialization for next clip
- VACE models (Video Adaptive Conditional Enhancement) recommended for best continuity
- Strength schedule controls how much previous frame influences next clip
- Automatically handles 4n+1 frame requirements per Wan spec

**Strength Resolution (Steps Dependency):**
- **Critical Relationship:** Strength resolution = 1/steps
- The number of sampling steps determines the precision of strength tuning
- **Flux Dev (20 steps):** 1/20 = 0.05 resolution (fine control, easier to tune)
- **Flux Schnell (4 steps):** 1/4 = 0.25 resolution (coarse control, harder to tune)
- **Strength Values:**
  - **0.0** - Cut (no previous frame feed, complete regeneration)
  - **1.0** - Fully feed previous image (maximum continuity/restriction)
  - **Too High (>0.8)** - Risk of artifacts, overly constrained generation
  - **Too Low (<0.2)** - Risk of discontinuity, jitter, poor frame coherence
- **Best Practice:** Use Flux Dev (20 steps) for strength-based workflows (I2V chaining, keyframe interpolation) to get finer control
- Displayed in UI as "Strength Resolution" component showing current 1/steps calculation

**FLF2V (First-Last-Frame-to-Video):**
- AI-powered interpolation between two keyframes using Wan's FLF2V pipeline
- **Critical Parameter: guidance_scale**
  - **3.5** (Default) - Smooth visual morphing, prioritizes seamless transitions
  - **5.5** - Official example value, balanced prompt adherence
  - **3.0-7.0** - Safe range for smooth transitions
  - **7.0+** - Risk of jitter, flicker, temporal inconsistency
  - **0.0** - ⚠️ **NEVER USE** - Breaks interpolation completely (ignores all conditioning including last_image)
- **Why 0.0 Breaks FLF2V:** When guidance_scale=0.0, the model ignores ALL conditioning inputs, including the `last_image` parameter, causing it to just extend the first frame instead of interpolating
- **Prompt Modes:**
  - **"none"** - No text prompting (recommended for smooth interpolation)
  - **"keyframe"** - Use keyframe prompts during interpolation
  - **"interpolated"** - Blend prompts between keyframes
- Used in:
  - 2D/3D modes with "Enable Wan FLF2V for Tweens" (Distribution tab)
  - Wan Only mode Phase 2 (ALL tween interpolation)
  - Wan Flux mode Phase 2 (Flux→Wan→Flux interpolation)

**Reverse Generation:**
- **Purpose:** Solve the forward zoom-in problem by generating frames in reverse order
- **The Problem:** Forward zoom-in is extremely difficult because the model must "imagine" what exists outside the current frame boundaries
- **The Solution:** Generate zoom-OUT (easy - model just fills visible areas naturally), then play the video backward to create perfect zoom-IN
- **Implementation:**
  - Checkbox location: Top-level UI, between FPS/Steps and Cadence
  - Frame processing order: Last→First (333→1)
  - Frame saving: Uses original frame numbers (frame.frame_idx), so video reassembles correctly
  - Tween handling: Automatic reassignment to ensure correct dependencies
- **Technical Details:**
  - When enabled, `run_render_animation()` reverses the frame list
  - Each frame's tweens are moved to the NEXT frame in generation order (PREVIOUS in timeline)
  - This ensures tweens are emitted AFTER their source keyframe exists
  - Example: Frame 20's tweens (11-19) are reassigned to Frame 10, emitted after Frame 20 is generated
- **Use Cases:**
  - POV camera movement (dash-cam, body-cam, FPV drone footage)
  - Forward zoom-in with stable backgrounds
  - Any scenario where you need to generate "what's outside the frame"
- **Compatible With:** All img2img workflows, 3D mode, depth warping, tween interpolation
- **First-Person Perspective AI Mode:** Qwen can generate POV camera prompts optimized for reverse generation
- **Implementation Files:**
  - `deforum/config/args.py` - Parameter definition
  - `deforum/ui/ui_left.py` - UI checkbox
  - `deforum/rendering/core.py:72-98` - Tween reassignment logic
  - `deforum/ui/handlers/audio_prompt_generator.py:159-189` - First-person perspective generation mode

**Orbital Camera Rotation Factor (Empirically Validated):**
- **Purpose:** Optimal counter-rotation for depth warping orbital camera paths
- **Empirical Optimal:** rotation_factor = **-8.0**
- **Discovery:** Comprehensive testing (99 configurations, 19,800 frames) across -50 to -1 range
- **Performance:**
  - Optimal value: -8.0 (125/200 frames = 62.5% stability)
  - Optimal range: -6.0 to -8.5 (all within 2% of best)
  - Stability ceiling: 62.5% over 200 iterations (limited by cumulative depth drift)
- **Theory vs Practice:**
  - Theoretical perfect orbit: -1.0 (360° translation → 360° counter-rotation)
  - Empirical optimal: -8.0 (8× stronger counter-rotation required)
  - **Why the difference:** Depth warping approximations don't preserve geometric perfection
- **Performance by Regime:**
  - Under-rotation (-50 to -10): Insufficient counter-rotation, sphere drifts ~57% stability
  - Optimal range (-6 to -9): Balanced, sphere stays centered ~62% stability
  - Over-rotation (-5 to -1): Excessive counter-rotation, faster drift
- **Technical Details:**
  - Schedules must use frame-to-frame DELTAS, not absolute positions
  - Both translation (x/y) AND rotation must be converted to deltas
  - Movement scale impacts stability: 5.0 = moderate, 2.0 = gentle
- **Default Values:**
  - Camera path presets: rotation_factor = -8.0
  - Tuning Lab UI: Min=-10.0, Max=-6.0 (sweep optimal range)
- **Implementation Files:**
  - `tests/integration/test_depth_warping_orbit_tuning.py` - Schedule generation (deltas!)
  - `tests/integration/test_camera_path_presets.py` - Default -8.0
  - `deforum/ui/ui_tuning.py` - Tuning Lab UI defaults
  - `ORBIT_TESTS.md` - Complete empirical results documentation
- **Related Testing:** See `ORBIT_TESTS.md` for comprehensive sweep results and RAFT integration

## Common Development Tasks

### Adding a New Animation Mode

1. Add enum to `scripts/deforum_helpers/rendering/data/anim/animation_mode.py`
2. Add UI option in `scripts/deforum_helpers/ui_right.py`
3. Implement pipeline in new file under `scripts/deforum_helpers/rendering/`
4. Route from `run_deforum.py` based on `anim_args.animation_mode`

### Adding a New Depth Model

1. Create `scripts/deforum_helpers/depth_<model>.py` following existing pattern
2. Implement `predict()` function returning depth map
3. Register in `scripts/deforum_helpers/depth.py`
4. Add UI option for model selection

### Modifying Keyframe Distribution

Edit `scripts/deforum_helpers/rendering/data/frame/key_frame_distribution.py`
- `distribute_keyframes()` - Main distribution algorithm
- `calculate_tween_weights()` - Interpolation between keyframes

### Adding Wan Features

- Model handling: `scripts/deforum_helpers/wan/wan_model_manager.py`
- Generation logic: `scripts/deforum_helpers/wan/wan_simple_integration.py`
- UI components: `scripts/deforum_helpers/wan/wan_ui_components.py`
- Qwen enhancement: `scripts/deforum_helpers/wan/qwen_prompt_expander.py`

### Extending UI

All UI code in `scripts/deforum_helpers/ui_right.py:28` in `on_ui_tabs()`
- Uses Gradio 4 components
- Returns `(interface, "Deforum", "deforum")` tuple
- Settings tab in `ui_settings.py`

## Important Patterns

**Lazy Logger Pattern:**
Module-level logger initialization is safe thanks to lazy proxy pattern:
```python
from deforum.utils.system.logging import get_logger

# Safe at module level - returns lazy proxy
logger = get_logger()

# Logger only initializes when first method is called
logger.info("This triggers initialization")  # First call initializes
logger.debug("Subsequent calls reuse instance")  # No re-initialization
```

**How it works:**
- `get_logger()` returns `_LazyLogger` proxy (not real `DeforumLogger`)
- Proxy defers initialization until first method call (`.info()`, `.error()`, etc.)
- Handles `opts` timing issues gracefully - falls back to defaults if opts unavailable
- Eliminates need for per-file boilerplate wrapper functions
- All 69+ files with `logger = get_logger()` at module level work correctly

**Why this matters:**
Before lazy pattern, `logger = get_logger()` at module level would fail if `modules.shared.opts` wasn't initialized yet, causing import failures and preventing functions from being defined. The lazy proxy solves this timing issue.

**Implementation:**
- `deforum/utils/system/logging/logger.py:307` - `_LazyLogger` proxy class
- Tests: `tests/unit/test_lazy_logger.py` - 13 comprehensive tests

**Argument Structure:**
Arguments flow as: Raw args → `process_args()` → Structured namespaces
- `args` - General settings (seed, sampler, steps, cfg_scale, etc.)
- `anim_args` - Animation settings (animation_mode, max_frames, etc.)
- `video_args` - Video output settings (fps, codec, etc.)
- `parseq_args` - Parseq integration settings
- `wan_args` - Wan video settings

**Frame Processing:**
Each frame goes through: Prompt → Seed → Denoise strength → Transformation → Depth → Output
- Keyframes: Full diffusion generation
- Tween frames: Interpolated from neighboring keyframes (render core only)

**Model Loading:**
- Wan mode skips SD/Flux model loading entirely (check in `run_deforum.py:52`)
- Qwen models lazy-load only when enhancement requested
- Depth models auto-download on first use to `models/Deforum/`

**Error Handling:**
- Use `JobStatusTracker().update_phase()` for progress
- Use `JobStatusTracker().fail_job()` for errors
- Rich console output with color codes from `rendering/util/log_utils.py`

## Testing

**Run all tests:**
```bash
pytest tests/ --start-server
```

**Run specific test:**
```bash
pytest tests/deforum_test.py::test_name -v
```

**Test configuration:**
- `tests/conftest.py` - Server startup fixture
- `pytest.ini` - Pytest settings (filters deprecation warnings)
- Tests use Deforum API endpoints at `http://localhost:7860/deforum_api/`

**Manual testing workflow:**
1. Launch Forge with `python webui.py`
2. Navigate to Deforum tab
3. Load `deforum/config/default_settings.txt`
4. Modify settings as needed
5. Click generate
6. Check console output for errors
7. Review generated video in output directory

## File References

When discussing code, use `file_path:line_number` format:
- Extension init: `scripts/deforum.py:24`
- Main orchestrator: `scripts/deforum_helpers/run_deforum.py:43`
- Standard render pipeline (3D/Interpolation): `deforum/rendering/core.py:22`
- Flux + Interpolation pipeline: `deforum/rendering/flux_interp.py:1`
- Central state: `scripts/deforum_helpers/rendering/data/render_data.py:42`
- Wan FLF2V wrapper: `scripts/deforum_helpers/wan/wan_simple_integration.py:1`
- Qwen enhancement (in Prompts tab): `scripts/deforum_helpers/wan/qwen_prompt_expander.py:1`
- UI elements: `scripts/deforum_helpers/ui_elements.py:1`
- UI left panel: `scripts/deforum_helpers/ui_left.py:1`
- Keyframe distribution: `scripts/deforum_helpers/rendering/data/frame/key_frame_distribution.py:1`

## Dependencies

Core dependencies (from `requirements.txt`):
- `numexpr`, `matplotlib`, `pandas` - Math and plotting
- `av`, `pims`, `imageio_ffmpeg` - Video processing
- `rich` - Console output formatting
- `gdown` - Google Drive downloads
- `easydict` - Dictionary utilities
- `diffusers` (git main) - Hugging Face diffusers library
- `transformers>=4.36.0,<4.46.0` - For Wan and Qwen models
- `accelerate>=0.25.0,<0.31.0` - For model acceleration

**Model Requirements and Directory Structure:**

**Shared Components (used by multiple models):**
- **FLUX VAE** (`models/VAE/ae.safetensors`, ~320MB)
  - Shared by: Flux, Lumina, Z-Image
  - Repository: `black-forest-labs/FLUX.1-dev`

**Flux Models:**
- **Checkpoint:** `models/Stable-diffusion/Flux/flux1-dev-bnb-nf4-v2.safetensors` (~12GB, 4-bit quantized)
- **Text Encoders:**
  - `models/text_encoder/clip_l.safetensors` (~235MB)
  - `models/text_encoder/t5xxl_fp16.safetensors` (~9.2GB)
- **Repository:** `lllyasviel/flux1-dev-bnb-nf4`, `comfyanonymous/flux_text_encoders`

**Lumina 2.0 (Anime-Optimized):**
- **UNet:** `models/Stable-diffusion/Lumina/Unet/neta-lumina-v1.0.safetensors` (~4.9GB)
- **Text Encoder:** `models/Stable-diffusion/Lumina/Text Encoder/gemma_2_2b_fp16.safetensors` (~4.9GB)
- **VAE:** `models/Stable-diffusion/Lumina/VAE/ae.safetensors` (symlinked to shared FLUX VAE)
- **Repository:** `neta-art/Neta-Lumina`
- **Full img2img support** - works with all Deforum render modes

**Z-Image-Turbo:**
- **DiT Model:** `models/Stable-diffusion/Z-Image/diffusion_pytorch_model.safetensors`
- **Text Encoder:** `models/text_encoder/qwen_3_4b.safetensors` (~7.5GB)
- **VAE:** Shares `models/VAE/ae.safetensors` (FLUX VAE)
- **Repository:** `stabilityai/stable-diffusion-3-medium`

**Wan AI Video:**
- **FLF2V:** `models/Deforum/wan/Wan2.1-FLF2V-14B/` (~14GB)
- **TI2V:** `models/Deforum/wan/Wan2.2-TI2V-5B/` (~5GB)
- **Repository:** `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers`, `Wan-AI/Wan2.2-TI2V-5B-Diffusers`

**Auto-Downloaded Models:**
- **Qwen:** `models/Deforum/qwen/` (3B/7B/14B variants, lazy-loaded)
- **Depth:** `models/Deforum/` (Depth-Anything V2, auto-downloaded on first use)
- **FILM:** `models/Deforum/film_interpolation/film_net_fp16.pt`

**Download Script:**
```bash
./shell_scripts/download-all-models.sh  # Interactive download with dependency checks
```

## Known Limitations

Known compatibility issues:
- **Kohya HR Fix** - May need to be disabled
- **FreeU** - May need to be disabled
- **Flux Schnell** - Limited precision with only 4 steps
- **Lumina 2.0** - Experimental support with automatic compatibility patch:
  - `KeyError: 'num_tokens'` fixed via automatic patch in `deforum/integrations/lumina/compat_patch.py`
  - Patch ensures `dynamic_args["num_tokens"]` is populated before sampling
  - Applied automatically when Lumina model detected (both txt2img and img2img)
  - Requires different parameters than Flux (CFG 4.0-5.5, Steps 30, scheduler linear_quadratic)
  - Anime-optimized, may produce suboptimal results for other styles

**Removed Features:**
- **Legacy/Stable Core** - Removed in favor of render core only
- **2D Animation Mode** - Removed (3D mode is now default and only depth-based mode)
- **Wan Only Mode** - Removed (superseded by Flux + Interpolation mode)
- **Hybrid Video Mode** - Removed completely
- **Legacy Depth Models** - MiDaS, AdaBins, LeReS, ZoeDepth removed (Depth-Anything V2 only)
- **Legacy ControlNet** - Removed (use Flux ControlNet V2 instead, available in 3D Depth tab)

## Troubleshooting

**Import errors after installation:**
Restart WebUI completely: `Ctrl+C` then `python launch.py`

**Settings not loading correctly:**
Download latest `deforum/config/default_settings.txt` from repo and load in UI

**Wan models not found:**
```bash
huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir models/Deforum/wan
```

**Qwen enhancement fails:**
- Check VRAM available
- Use "Cleanup Qwen Cache" button in UI
- Select smaller model (3B instead of 7B/14B)
- Check console for error messages

**Generation fails with render core:**
- Disable Kohya HR Fix
- Disable FreeU
- Ensure keyframes align with prompt frame numbers

**Out of memory:**
- Reduce resolution
- Use quantized Flux model (bnb-nf4)
- Reduce max_frames or frame count
- For Wan: Use 1.3B instead of 14B model

**FLF2V interpolation not working:**
- Check guidance_scale is NOT 0.0 (default should be 3.5)
- Try guidance_scale values between 3.0-7.0
- Use prompt_mode="none" for smoothest interpolation
- Verify both first and last keyframes are properly generated
- Check console for FLF2V DEBUG logs in Wan Only/Flux modes

## Coding Standards

When writing or modifying code in this repository, follow **STRICT** functional programming principles and Python best practices. See `CODING_GUIDE.md` for complete details.

### Critical Rules (Must Follow)

1. **Complexity Limit:** All functions MUST have McCabe complexity ≤ 10
2. **Type Hints:** Complete type annotations required on ALL functions  
3. **Error Handling:** Comprehensive try-catch blocks with graceful fallbacks
4. **Documentation:** Clear docstrings with Google-style parameter descriptions
5. **Code Style:** Black formatting (100 char line length), flake8 linting must pass

### Functional Programming Principles

- **Small pure functions:** Max 20 lines, single responsibility, side-effect free
- **Prefer expressions over statements:** Use ternary operators and comprehensions
- **No magic numbers:** Extract all constants to module top or `constants.py`
- **Immutable by default:** Return new objects, don't modify inputs
- **Function composition:** Design functions that can be chained/composed
- **Explicit dependencies:** Function params show what data is needed
- **Separate pure logic from side effects:** Pure functions in `utils/`, side effects in orchestrators

### Key Architecture Components

1. **Experimental Core** (`deforum/rendering/core.py:22`)
   - This is the ONLY render core (legacy core removed)
   - Main render loop at `render_animation()`
   - Generates subtitle .srt file asynchronously
   - Iterates through frames calling `generate_inner()`
   - Applies transformations (2D/3D movement, depth warping)
   - Handles masks and noise schedules
   - Stitches final video with ffmpeg

2. **IMG2IMG Pipelines** (Forge backend)
   - Located in `backend/` (cherry-picked from ComfyUI)
   - `backend/nn/unet.py` - UNet with patching system
   - `backend/patcher/` - Model patching (LoRA, ControlNet, etc.)
   - `backend/sampling/` - Sampling implementations
   - Deforum hooks into Forge's processing pipeline via `modules/processing.py`

3. **Keyframe Distribution System** (`scripts/deforum_helpers/rendering/data/frame/`)
   - `key_frame_distribution.py` - Distribution algorithms
   - `diffusion_frame.py` - Frame metadata and state
   - `tween_frame.py` - Interpolated frames between keyframes
   - Central to render core operation

### Tools and Commands

```bash
# Check complexity (must be ≤ 10 for all functions)
pip install radon
radon cc scripts/deforum_helpers/ -a -nc

# Format code
pip install black flake8
black scripts/deforum_helpers/ tests/ --line-length 100
flake8 scripts/deforum_helpers/ tests/ --max-line-length 100

# Type checking
pip install mypy
mypy scripts/deforum_helpers/ --strict

# Run tests with coverage
pytest tests/unit/ -v --cov-report=html

# Find dead code
pip install vulture
vulture scripts/deforum_helpers/
```

### Example: Good Refactoring

**Before:**
```python
def process(frames, settings):
    results = []
    for i in range(len(frames)):
        if frames[i] is not None:
            if settings['mode'] == '3D':
                if frames[i]['width'] > 1920:
                    results.append({'data': frames[i]['data'] * 0.5, 'id': i})
                else:
                    results.append({'data': frames[i]['data'], 'id': i})
    return results
```

**After:**
```python
from typing import NamedTuple
from dataclasses import dataclass

MAX_WIDTH = 1920
SCALE_FACTOR = 0.5

@dataclass(frozen=True)
class Frame:
    data: np.ndarray
    width: int

@dataclass(frozen=True) 
class ProcessedFrame:
    data: np.ndarray
    id: int

def should_scale(width: int) -> bool:
    """Determine if frame needs scaling based on width."""
    return width > MAX_WIDTH

def process_frame(frame: Frame, frame_id: int) -> ProcessedFrame:
    """Process single frame with appropriate scaling.
    
    Args:
        frame: Input frame data
        frame_id: Frame index in sequence
        
    Returns:
        Processed frame with scaling applied if needed
    """
    scale = SCALE_FACTOR if should_scale(frame.width) else 1.0
    return ProcessedFrame(data=frame.data * scale, id=frame_id)

def process_frames(frames: list[Frame | None], mode: str) -> list[ProcessedFrame]:
    """Process frames in 3D mode with appropriate scaling.
    
    Args:
        frames: List of frames (None entries are skipped)
        mode: Rendering mode ('2D' or '3D')
        
    Returns:
        List of processed frames
    """
    if mode != '3D':
        return []
        
    return [
        process_frame(frame, idx)
        for idx, frame in enumerate(frames)
        if frame is not None
    ]
```

See `CODING_GUIDE.md` for comprehensive guidelines and anti-patterns to avoid.
