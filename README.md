# Zirteq's Fluxabled Fork of the Deforum Extension for Forge Neo Fork of Forge WebUI Fork of Automatic1111

[![Unit Tests](https://github.com/Tok/sd-forge-deforum/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/Tok/sd-forge-deforum/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/Tok/sd-forge-deforum/branch/main/graph/badge.svg)](https://codecov.io/gh/Tok/sd-forge-deforum)

⚠️ **COMPATIBILITY WARNING**: This fork is **100% incompatible** with older Deforum versions and original `deforum_settings.txt` files. You must use the new settings format from this repository.

**Primary Target:** [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) - Fully tested and supported
**Other Forge versions:** May work but remain untested

Experimental fork of the [Deforum extension](https://github.com/deforum-art/sd-forge-deforum),
completely refactored and modernized to work with **Flux.1/2**, **Lumina 2.0**, **Z-Image-Turbo**, **Wan 2.1/2.2 AI Video Generation**, and advanced workflow automation.

## 📖 Table of Contents

- [⚡ Major New Features](#-major-new-features)
- [🚀 Quick Start](#-quick-start)
- [💿 Installation](#-installation)
- [📥 Model Downloads](#-model-downloads)
- [🎛️ Render Modes](#️-render-modes)
- [🎨 Keyframe Scheduling](#-keyframe-scheduling)
- [🧪 Camera Path Tuning Lab](#-camera-path-tuning-lab)
- [🔧 Helper Scripts](#-helper-scripts)
- [🏛️ Architecture & Documentation](#️-architecture--documentation)
- [🤝 Contributing](#-contributing)

## ⚡ Major New Features

### 🎬 **Wan AI Video Generation** (Alibaba's state-of-the-art T2V/I2V)
- **FLF2V Integration** (Wan 2.1): First-Last-Frame-to-Video interpolation with guidance_scale=3.5
- **TI2V Models** (Wan 2.2): Unified Text-to-Video and Image-to-Video generation
- **I2V Chaining**: Seamless clip transitions using last frame as init for next clip
- **Frame-Perfect Timing**: Full integration with Deforum's prompt scheduling system
- **Auto-Discovery**: Automatic model detection from `models/Deforum/wan/` directory

### ⚙️ **Fractional Strength Precision** (Always Enabled)
- **1% Precision**: Fine-grained strength control (0.01 resolution) regardless of step count
- **Auto-Enabled**: Applied via monkey patches at extension load (no configuration needed)
- **Critical for I2V**: Enables precise tuning for Flux Schnell (4 steps) and all I2V chaining workflows
- **Tuning-Ready**: Integrated tuning platform uses fractional precision for empirical parameter optimization
- **See**: `TUNING.md` for comprehensive parameter sweep configurations (18 test cases)

### 🤖 **Qwen AI Prompt Enhancement & Generation**
- **5 Model Options**: From 3B (low VRAM) to 14B (maximum quality)
- **Auto-Selection**: Intelligent model choice based on available VRAM (4GB-28GB)
- **Movement Analysis**: Translates Deforum schedules to natural language descriptions
- **Frame-Specific Analysis**: Unique camera movement descriptions per keyframe
- **Lazy Loading**: Models only load when needed, auto-cleanup before generation
- **Bilingual**: English and Chinese prompt enhancement support

### 📺 **Terminal Dashboard with ASCII Art Preview**
- **Fixed-Position Display**: Real-time dashboard with progress bars, VRAM monitoring, and scrolling log
- **16:9 ASCII Art Preview**: Live colored preview of last generated frame (32x18 pixels, 2-space blocks)
- **True Color Support**: 24-bit ANSI background colors for accurate frame representation
- **Theme Integration**: Respects slopcore/classic/simple theme settings
- **Aspect Ratio Preservation**: Automatic aspect ratio handling for all resolutions
- **Optional Log Spam**: Write ASCII art to scrolling log on each frame (off by default)
- **Memory Monitoring**: Parse and display Forge VRAM stats with visual progress bar
- **Parallel Progress Tracking**: 5 themed tqdm bars (Tweens, Total Frames, Steps, Total Steps, Diffusion Frames)
- **Zero Terminal Clutter**: Suppresses redundant Forge output, shows only essential messages
- **Settings**: `Settings → Deforum → Console & UI Output Settings`
  - Enable/disable dashboard (on by default)
  - ASCII preview in header (on by default)
  - ASCII to scrolling log (off by default)

### 💾 **Privacy-Focused Video Metadata Embedding**
- **Full Reproducibility**: Automatically embeds ALL generation settings into video files
- **Dual Embedding Approach**:
  - **Base64-Encoded Comment**: Complete settings (all args, movements, prompts) for machine parsing
  - **Human-Readable Fields**: Key info directly accessible (`deforum_resolution`, `deforum_model`, `deforum_seed`, etc.)
- **No User-Identifying Info**: Only technical generation parameters, no branding or personal data
- **Fork-ception Identity**: Includes full fork name and GitHub URL for version tracking
- **Git Commit Tracking**: Exact commit ID embedded for perfect reproducibility
- **ComfyUI-Inspired**: Similar workflow to ComfyUI's image metadata embedding
- **Extract & Restore**: Load settings from any Deforum-generated video (drag-and-drop planned)
- **Comprehensive Data**: Includes:
  - Core settings: model, sampler, scheduler, steps, cfg_scale, distilled_cfg_scale
  - Resolution: width x height
  - Animation: fps, max_frames, render_mode, animation_mode
  - Movement schedules: All camera movement parameters
  - Prompt schedule: Frame-specific prompts (truncated if >500 chars)
  - Advanced: Parseq data, ControlNet settings, Wan parameters
- **FFmpeg Integration**: Embedded using standard MP4/MOV metadata fields
- **Always Enabled**: Automatic embedding with every video generation (toggle coming soon)

### 🎵 **Automatic Audio Event Detection & Synchronization**
- **Librosa Integration**: Professional audio analysis with onset/beat/bass detection
- **Real-Time Preview**: Adjustable sensitivity with ±5% buttons
- **Multi-Method Detection**: Onsets, beats, bass energy, and combined events
- **Frame-Perfect Sync**: Automatic keyframe placement at detected audio events
- **Prompt Distribution**: AI-generated prompts synchronized to music beats
- **Generation Modes**: Escalating, cyclical, thematic, narrative, and custom modes

### 🎞️ **FILM Smearcore Integration** (Google's frame interpolation)
- **High-Quality Interpolation**: Google Research's FILM model for cinematic motion
- **Smearcore Aesthetic**: Configurable motion blur and temporal blending
- **Multi-Method Support**: Choose between Wan FLF2V or FILM for Flux interpolation; RIFE v4.26 available for post-processing
- **Post-Processing Pipeline**: Apply FILM interpolation after initial render
 
### 🔄 **Resurrected & Upgraded RIFE v4.26**
- **Post-Processing Only**: Latest RIFE model (Sept 2024) for smoothing finished videos
- **Best For**: Framerate doubling on smooth motion (defaults to single frames on dramatic changes)
- **Not For**: Interpolating between dramatically different keyframes (use FILM instead)
- **Multi-GPU Support**: Efficient memory management for interpolation tasks

### ⏪ **Reverse Generation** (Essential for stable forward-motion clips)
- **Reverse Frame Order**: Generate frames from last→first (333→1), reassembled correctly for final video
- **Natural Zoom-Out**: Model naturally fills new areas when zooming OUT during generation
- **Perfect Zoom-In**: Final video plays forward (1→333) showing smooth zoom-in effect
- **POV Camera Movement**: Ideal for dash-cam, body-cam, FPV drone footage with stable backgrounds
- **Intelligent Tween Handling**: Automatic tween reassignment ensures correct dependency order
- **Works with All img2img Workflows**: Compatible with 3D mode, depth warping, and all standard features
- **First-Person Perspective AI Mode**: Qwen can generate POV camera prompts optimized for reverse generation
- **Why It Matters**: Forward zoom-in is notoriously difficult (model struggles with "imagining" what's outside the frame). Reverse generation solves this by generating zoom-out (easy - just fill visible areas), then playing backward for perfect zoom-in.

### 🏗️ **Total Codebase Refactor**
- **1000+ Unit Tests**: Comprehensive test coverage with pytest
- **Type-Safe**: Complete type hints and mypy strict mode compliance
- **Functional Patterns**: Pure functions, immutable data, composition over inheritance
- **Clean Architecture**:
  - `deforum/core/` - Business logic (keyframes, prompts, seeds)
  - `deforum/utils/` - Pure utility functions
  - `deforum/rendering/` - Rendering pipeline
  - `deforum/integrations/` - External integrations (Wan, Parseq, Flux ControlNet)
- **Developer Friendly**: Clear documentation, CLAUDE.md guidance, CODING_GUIDE.md standards

### 🔌 **Revived Deforum API with OpenAPI Interface**
- **RESTful Endpoints**: Full programmatic control over Deforum
- **OpenAPI Schema**: Auto-generated documentation and client SDKs
- **Integration Tests**: Automated E2E testing via API
- **Batch Operations**: Queue multiple renders programmatically
- **Status Monitoring**: Real-time progress tracking and error reporting

### 🌐 **Model Context Protocol (MCP) Integration**
- **Claude Desktop Integration**: Control Deforum directly from Claude Desktop
- **Standardized Tools**: 8 MCP tools for job management, status, settings, and generation
- **Async Job Execution**: Non-blocking renders with real-time progress updates
- **Type-Safe Protocol**: Full Pydantic validation and error handling
- **One-Click Setup**: Example config at `config/claude_desktop_config.example.json`
- **See**: `docs/MCP_INTEGRATION.md` for complete setup guide

### 🎨 **Reworked Workflow-Centric UI**
- **4 Render Modes**: Classic 3D, New 3D, Keyframes Only, Flux + Interpolation
- **Flattened Navigation**: Single-level tab structure for faster access
- **Promoted Tabs**: Distribution, Shakify, 3D Depth elevated to main level
- **Context-Aware Controls**: UI adapts based on selected render mode
- **Dual Strength Schedules**: Normal + keyframe strength for advanced control

### 🕳️ **Depth-Anything V2** (Only depth model - faster & more accurate)
- **State-of-Art**: Latest depth estimation from DepthAnything team
- **Unified Model**: Single model replaces 5 legacy options (MiDaS, AdaBins, LeReS, ZoeDepth, DPT-Large)
- **Auto-Download**: First use downloads to `models/Deforum/`
- **GPU Accelerated**: Optimized for modern hardware

### 📹 **Camera Shakify Integration** (EatTheFuture's Blender patterns)
- **Pre-Recorded Patterns**: EARTHQUAKE, FILM_GRAIN, GENTLE_HANDHELD, INVESTIGATION, SMOOTH_DOLLY
- **CC0 Licensed**: Creative Commons public domain shake data

### 🎥 **3D Camera Path Spline Generation**
- **8 Preset Movements**: rotate-around, figure-eight, forward-zoom, orbit-up, spiral, street, dashcam, bodycam
- **Smooth Interpolation**: Bezier and Catmull-Rom splines for cinematic camera motion
- **Closed Loop Support**: Seamless looping animations for perfect transitions
- **Real-Time Preview**: 3D visualization before populating schedules
- **Automatic Schedule Population**: Generates translation_x/y/z and rotation_3d_x/y/z schedules
- **Randomization**: Add controlled chaos to camera paths for organic movement
- **Look-At Logic**: Camera automatically follows curve tangents

### 🎼 **AI Audio Generation** (Meta MusicGen Integration)
- **MusicGen Small**: Meta's 341M parameter music generation model
- **Stable Audio Open**: Alternative audio generation backend
- **Theme-Based Generation**: Describe the vibe, get matching audio loops
- **BPM Control**: Adjustable tempo for beat-synchronized animations
- **Mono Output**: Optimized for video soundtracks (1D array format)
- **Fallback Patterns**: Amen break and other classic samples when models unavailable
- **Zero-HITL Integration**: Automatic audio generation with intentional chaos (20% tempo manipulation, 15% micro-loops, 10% vinyl crackle)
- **Dedicated Tab**: Easy access to all patterns and intensity controls
- **Realistic Motion**: Add cinematic camera shake on top of scheduled movement

## ❌ Removed Legacy Features

### **Complete Removals** (100% gone)
- **❌ Hybrid Video Mode**: Completely removed from codebase
- **❌ Legacy/Stable Render Core**: Only experimental (now called "render core") remains
- **❌ 2D Animation Mode**: 3D mode is now the only depth-based option
- **❌ Wan Only Mode**: Superseded by Flux + Interpolation hybrid workflow
- **❌ Original Deforum Render Core**: Total replacement with new architecture

### **Deprecated Depth Models** (replaced by Depth-Anything V2)
- **❌ MiDaS**: Removed (slow, inaccurate)
- **❌ AdaBins**: Removed (memory hungry, unstable)
- **❌ LeReS**: Removed (poor quality)
- **❌ ZoeDepth**: Removed (compatibility issues)
- **❌ DPT-Large**: Removed (redundant with Depth-Anything V2)

### **Settings Incompatibility**
- **❌ Old deforum_settings.txt**: Will NOT work - download new format from repo
- **❌ Legacy Parameters**: Many renamed or restructured for clarity
- **❌ Backward Compatibility**: None - this is a complete rewrite

## Current Status

This fork is **actively maintained** and **production-ready** for Flux.1 workflows.

⚠️ **Compatibility Notes**:
- ✅ **Flux Models**: Fully tested and working
- ⚠️ **Flux Schnell**: Limited (only 4 steps makes fine-tuning difficult)
- ⚠️ **SD 1.5/XL**: Untested and not supported in this fork
- ❌ **Kohya HR Fix**: May need to be disabled
- ❌ **FreeU**: May need to be disabled
- ⚠️ **ControlNet**: Flux ControlNet V2 works, legacy ControlNet untested

## UI Structure

The Deforum UI is organized into workflow-centric tabs optimized for different generation modes:

### Main Tabs

**Top-Level Controls** (always visible):
- **Render Mode** - 4 workflow presets: Classic 3D, New 3D, Keyframes Only, Flux + Interpolation
- **FPS** - Frame rate (auto-adjusts: 24 for Flux, 60 for 3D modes)
- **Steps** - Sampling steps (mode-specific tooltips explain what it controls)
- **Cadence/Pseudo-Cadence** - Real cadence slider or calculated display
- **Strength Schedules** - 1 or 2 sliders (normal + keyframe for New 3D mode)

**Tab Navigation:**
1. **Run** - Main generation controls and status
2. **Keyframes** - Motion, CFG, Seed, Step, Sampler, Scheduler, Checkpoint, Noise, Coherence, Anti-Blur
   - Single-level flattened structure (no nested sub-tabs)
3. **Distribution** - Render mode selection + Wan FLF2V tween integration
   - Keyframe type scheduling (keyframe vs cadence diffusions)
4. **Prompts** - Text prompts with frame numbers
   - **🧠 AI Prompt Enhancement** accordion (Qwen integration)
5. **Shakify** - Camera shake patterns (3D modes only)
6. **3D Depth** - Depth warping + Flux ControlNet V2 (3D modes only)
7. **Init** - Audio Sync tab with event detection + Initialization settings
8. **Interpolation** - Multi-method interpolation config (Flux + Interpolation mode only)
9. **Output** - Video encoding, FPS, resolution, audio

### Render Modes

#### **1. Classic 3D**
Traditional Deforum with fixed cadence for maximum stability:
- Keyframe Distribution: OFF (uniform cadence)
- Strength: Single (normal only)
- Defaults: 24 FPS, cadence=2, 20 steps
- Best For: RAFT optical flow, ControlNet, legacy workflows

#### **2. New 3D** (Default) ⭐
Modern keyframe redistribution with dual strength:
- Keyframe Distribution: REDISTRIBUTED
- Strength: Dual (normal + keyframe)
- Defaults: 60 FPS, cadence=5, 20 steps
- Best For: Balanced quality, speed, stability

#### **3. Keyframes Only** ⚡
Pure keyframe diffusion with depth tweening:
- Keyframe Distribution: KEYFRAMES_ONLY
- Strength: Single (keyframe only)
- Defaults: 60 FPS, pseudo-cadence, 20 steps
- Best For: Maximum speed, slow movements
- Note: Not compatible with RAFT/ControlNet

#### **4. Flux + Interpolation** 🎬
Hybrid Flux keyframes + multi-method interpolation:
- Interpolation: Wan FLF2V or pure FILM for smearcore aesthetics
- Strength: Single (I2V chaining)
- Defaults: 24 FPS, pseudo-cadence, 20 steps
- Best For: Dramatic changes, cinematic quality
- Features: Qwen prompt enhancement, movement analysis

## Requirements

### Get SD WebUI Forge
Install the 'one-click installation package' of
[Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge):
* Python 3.10.6
* CUDA 12.1
* Pytorch 2.3.1

### Run Flux on Forge

Get `flux1-dev-bnb-nf4-v2.safetensors` from huggingface:
```bash
# Download Flux checkpoint to models/Stable-diffusion/Flux/
wget https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/resolve/main/flux1-dev-bnb-nf4-v2.safetensors
```

Get VAE and text encoders:
```bash
# Download to models/VAE/
wget https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors
wget https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors
wget https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors
```

Restart Forge, set mode to "flux", select the flux checkpoint and all 3 VAEs in "VAE / Text Encoder".

### Run Lumina 2.0 on Forge Neo (Alternative to Flux)

**⚠️ Forge Neo Only** - Lumina 2.0 is only available in [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo), not classic Forge.

Lumina 2.0 is a 2B parameter model (vs Flux's 12B) with **full img2img support**, making it perfect for:
- **Lower VRAM systems** (2GB model vs 12GB Flux)
- **Faster generation** (fewer parameters = faster inference)
- **1024x1024 native resolution** (optimal quality at this size)
- **All Deforum render modes** (Classic 3D, New 3D, Keyframes Only, and can replace Flux in "Flux + Interpolation" mode)

**Installation:**
```bash
# Option 1: Use Forge's model downloader UI (if available)
# Go to Forge's model list, search for "Lumina"

# Option 2: Manual download via huggingface-cli (recommended)
cd /path/to/forge-neo
hf download neta-art/Neta-Lumina --local-dir models/Stable-diffusion/Lumina

# This downloads:
# - neta-lumina-v1.0-all-in-one.safetensors (9.9GB) - complete bundled model
# - Text Encoder/gemma_2_2b_fp16.safetensors (4.9GB)
# - Unet/neta-lumina-v1.0.safetensors (4.9GB)
# - VAE/ae.safetensors (320MB) - FLUX-VAE
```

**Usage:**
1. Restart Forge Neo
2. In main UI, select "Lumina" mode (instead of "flux")
3. Select a Lumina checkpoint from the model dropdown
4. Lumina shares the same FLUX-VAE as Flux models (ae.safetensors)
5. Uses Gemma-2-2B text encoder (auto-loaded)

**Deforum Compatibility:**
- ⚠️ **EXPERIMENTAL** - Lumina may require parameter tuning
- ✅ Classic 3D - Fixed cadence img2img rendering
- ✅ New 3D - Keyframe redistribution with dual strength
- ✅ Keyframes Only - Pure keyframes + depth tweening
- ✅ Flux + Interpolation - Use Lumina for keyframes, then interpolate with Wan/RIFE/FILM

**⚠️ Known Issues (With Automatic Fix):**
- **num_tokens error**: Fixed via automatic compatibility patch
  - Deforum now automatically detects Lumina and ensures `num_tokens` is populated
  - Patch applied before sampling in both txt2img and img2img modes
  - If errors still occur, please report with console logs
- **Output quality**: Lumina is anime-optimized and may need different CFG/steps than Flux:
  - CFG Scale: 4.0-5.5 (vs Flux's 1.0-3.5)
  - Steps: 30 recommended (vs Flux's 20)
  - Scheduler: linear_quadratic preferred (vs simple)
  - Sampler: res_multistep or euler_ancestral work best

**Technical Details:**
- Architecture: Flow-based diffusion transformer (like Flux)
- Text Encoder: Gemma-2-2B (vs Flux's T5-XXL)
- VAE: FLUX-VAE-16CH (shared with Flux)
- License: Apache-2.0 (fully open source)

### Run Flux 2 on Forge Neo

**⚠️ EXPERIMENTAL** - Flux 2 is not currently working with Deforum on Forge Neo.

Flux 2 is Black Forest Labs' latest model with improved efficiency:
- 8 double-stream blocks + 48 single-stream blocks (vs Flux 1's 19/38)
- Single text encoder: Mistral Small 3.1
- Same quality, faster inference

**Current Status:**
- ❌ Not working with current GGUF models (architecture mismatch)
- ✅ Flux 1 GGUF works great with compatibility patch
- 🔧 Infrastructure in place for future Flux 2 support

**What Works:**
- Model loads successfully
- Architecture auto-detected
- Diagnostic logging available

**What Doesn't:**
- Generation fails with dimension mismatch
- See [FLUX2_STATUS.md](docs/FLUX2_STATUS.md) for technical details

The compatibility work done for Flux 2 improves Flux 1 GGUF support and prepares Deforum for when proper Flux 2 compatibility becomes available

### Run Z-Image-Turbo on Forge Neo (Ultra-Fast)

**⚠️ Forge Neo Only** - Z-Image-Turbo is only supported in [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo).

Z-Image-Turbo is Tongyi MAI's ultra-fast image generation model with:
- **3.8B parameters** (smaller than Flux, larger than Lumina)
- **BFloat16 precision** - fast inference with good quality
- **Optimized for speed** - designed for rapid generation

**Installation:**
```bash
# Download from Hugging Face (see download guide)
# https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models
cd /path/to/forge-neo/models/Stable-diffusion
# Follow Forge Neo's model download guide for Z-Image-Turbo
```

**Deforum Compatibility:**
- ✅ Works with current dev branch
- ✅ All render modes supported
- ⚠️ **Tuning needed** - optimal parameters still being discovered
- ⚠️ Different architecture may require adjusted CFG/steps

**Resources:**
- Model: [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo)
- Download Guide: [Forge Neo Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models)

### 🔥 **Zero-HITL Slopcore Generator** (Qwen-Spec'd, Zero Human-In-The-Loop)
*Specifications and review by Qwen3-Next-80B-A3B. Implementation was practically zero-HITL too (copy-paste driven development).*

⚠️ **EXPERIMENTAL - CURRENTLY MISTUNED**: The Qwen AI Creative Director component is functional but produces suboptimal results. The chaos parameters (audio sabotage, RGB inversion, glitch effects) work as intended, but prompt generation needs retuning. Expect interesting failures.

One-click AI video generation with **intentional chaos and glitches**. Click **"🔥 SLOP IT! 🔥"** and walk away.

**Complete Pipeline:**
1. **Intentionally Sabotaged Audio** (Stable Audio Open Small 341M)
   - 20% tempo chaos (0.5x/2x speed with pitch shift)
   - 15% micro-loop glitch (1s loops with jump cuts)
   - 10% vinyl crackle at 120% volume
   - 5% corrupted WAV header (white noise intro)
   - Special: "random" theme → malfunctioning microwave sounds

2. **Curated Chaos Parameters**
   - Randomized render settings with slopcore aesthetics
   - 20% RGB inversion, 25% style combinations
   - 10% seed=0 (glitch art mode)
   - 7 blue/purple gradient shades + neon pink palette

3. **Qwen AI Creative Director** (Camera-Aware)
   - Makes ALL artistic decisions
   - 25% contradiction prompts ("serene beach with neon tornado")
   - Responds to camera movement (spin → vortex prompts)
   - Structured JSON creative direction

4. **Camera Chaos**
   - 30% camera jitter (±5-20px random shifts)
   - 10% 360° spin during calm scenes
   - Informs Qwen's prompt generation

5. **VHS Scan Lines** (Always Applied)
   - Dual approach: Per-frame (1-3px) + FFmpeg post-processing
   - Randomized parameters: degrade, chroma, noise, jitter
   - Left-side degradation (realistic VHS tape wear)
   - Mandatory for authentic glitch aesthetic

6. **Accidental Masterpiece Mode** (Always Enabled)
   - Detects "too good" outputs via broken heuristics
   - Applies 2-5 glitch effects: RGB invert, datamosh, scanline overdrive, color shift
   - **Healing Glitches**: If too broken (3+ effects), applies motion blur for "accidental beauty"

7. **Slopcore Confidence Score**
   - Calculated from triggered chaos events
   - 90-100% = GLITCHED (desirable)
   - 0-50% = TOO GOOD (rejects, needs more chaos)

**Features:**
- Zero configuration (all decisions automated)
- Full transparency (detailed slop log with emoji storytelling)
- Settings export (JSON of all generated parameters)
- Isolated to dedicated tab (normal Deforum unaffected)

**See:** `docs/ZERO_HITL_DESIGN.md` for Qwen's complete specifications


## Installation

### Directly in Forge (recommended)

Go to Extensions → Install from URL:
```
https://github.com/Tok/sd-forge-deforum.git
```

**Note on Dependencies:** This extension automatically manages `huggingface-hub` compatibility. On first startup, it may detect an incompatible version and fix it automatically. If you see import errors, simply restart Forge once and the extension will resolve them.

### From Command Line

```bash
cd <forge_install_dir>/extensions
git clone https://github.com/Tok/sd-forge-deforum
cd sd-forge-deforum
pip install -r requirements.txt
```

**After Updates:** If you update the extension via `git pull`, clear Python bytecode cache to ensure changes take effect:
```bash
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
```

**Forge Neo Compatibility:** This extension is fully compatible with [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo).

**For Forge Setup & Base Models:** See the [Forge Neo README](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo) and [Model Download Wiki](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Download-Models) for setting up Forge WebUI itself and downloading SD/SDXL/Flux base models.

## Helper Scripts

This extension includes comprehensive helper scripts for common tasks. All scripts have both Linux/Mac (`.sh`) and Windows (`.bat`) versions.

### Script Organization

**Extension Root** (quick access):
- `setup.sh` / `setup.bat` - Dependency installation and venv management
- `start-forge.sh` / `start-forge.bat` - Launch Forge with optimizations

**`shell_scripts/`** (organized utilities):
- Model downloads (`download-all-models.sh`, `download-forge-models.sh`)
- Installation scripts (`install-cuda-toolkit.sh`, `install-sageattention.sh`)
- Test runners (`run-unit-tests.sh`, `run-api-tests.sh`)
- Tuning lab launcher (`run-tuning-lab.sh`)

**`scripts/`** (development tools):
- Python-only development utilities (migrate_prints_to_logger.py, etc.)

### Setup & Dependencies

Located in the **extension root**:

**`setup.sh` / `setup.bat`** - Unified setup and migration tool
```bash
./setup.sh              # Interactive menu
./setup.sh --check      # Check Python version, dependencies, optimizations
./setup.sh --prepare    # First-time setup (PyTorch + SageAttention + Deforum deps)
./setup.sh --install    # Install Deforum requirements.txt only
./setup.sh --migrate    # Migrate venv to Python 3.11.9 (Linux only)
```

**First-Time Setup (Recommended):**
```bash
./setup.sh --prepare
```
This runs a complete first-time setup:
1. Launches Forge once with `--exit` to install PyTorch
2. Installs SageAttention (now that torch is available)
3. Installs all Deforum dependencies
4. Ready to launch with full optimizations

**Features:**
- Python version check (3.11.9 recommended, 3.12 supported)
- Dependency verification (pandas, rich, librosa, etc.)
- Optimization check (SageAttention, FlashAttention)
- First-time setup mode handles PyTorch → SageAttention dependency chain
- Full venv migration with backup (Linux only)

**Manual SageAttention Installation (if `--prepare` fails):**

Located in `shell_scripts/`:

**`install-cuda-toolkit.sh` / `.bat`** - Install CUDA Toolkit (required for SageAttention)
```bash
./scripts/install-cuda-toolkit.sh    # Linux: Install CUDA 12.6 via apt
./scripts/install-cuda-toolkit.bat   # Windows: Installation instructions
```

**`install-sageattention.sh` / `.bat`** - Install SageAttention optimization
```bash
./scripts/install-sageattention.sh   # Compile SageAttention from source
```

**Why separate scripts?** SageAttention requires the CUDA toolkit (nvcc compiler) to compile from source. Most users only have PyTorch with CUDA support, not the full CUDA development toolkit. These scripts:
1. Install CUDA toolkit (~3GB)
2. Set CUDA_HOME environment variable
3. Compile SageAttention with access to torch during build

**Note:** The `--prepare` mode attempts SageAttention installation but continues if it fails (CUDA toolkit missing). Use these scripts if you want to install CUDA toolkit and SageAttention manually.

### Launching Forge

Located in the extension root:

**`start-forge.sh` / `start-forge.bat`** - Launch Forge with optimizations
```bash
./start-forge.sh           # Start with --sage --fast-fp16 --cuda-malloc --cuda-stream
./start-forge.sh --no-opt  # Start without optimizations
./start-forge.sh --listen  # Add custom flags (keeps optimizations)
```

**Optimization Flags (all enabled by default):**
- `--sage`: SageAttention (RTX 30/40/50 GPUs) - Run `./setup.sh --prepare` first if this fails
- `--fast-fp16`: Fast FP16 accumulation (requires PyTorch 2.7+)
- `--cuda-malloc`: CUDA malloc optimization
- `--cuda-stream`: CUDA stream optimization

**Note:** If you get a SageAttention build error on first launch, run `./setup.sh --prepare` to properly install PyTorch and SageAttention in the correct order.

### Model Downloads

Located in `shell_scripts/`:

**`download-all-models.sh` / `.bat`** - Download Deforum-specific models
```bash
./scripts/download-all-models.sh
```

**Downloads:**
1. Flux.1 Dev BNB NF4 v2 (~10GB, 4-bit quantized)
2. Flux VAE and text encoders (CLIP-L, T5-XXL)
3. Flux ControlNet V2 (Canny, Depth) - optional
4. FILM interpolation model
5. Wan AI Video models (FLF2V-14B, TI2V-5B, TI2V-A14B) - optional
6. Qwen prompt enhancement models (3B/7B/14B) - optional

**`download-forge-models.sh` / `.bat`** - Download base Forge models
```bash
./scripts/download-forge-models.sh
```

**Downloads:**
- SD1 VAE (vae-ft-mse-840000)
- SDXL VAE (sdxl-vae-fp16-fix)
- Links to CivitAI for SD1.5/SDXL checkpoints

### Testing

Located in `shell_scripts/`:

**`run-unit-tests.sh` / `.bat`** - Run unit tests
```bash
./scripts/run-unit-tests.sh        # Run all unit tests
./scripts/run-unit-tests.sh -v     # Verbose output
```

**`run-api-tests.sh` / `.bat`** - Run API integration tests
```bash
./scripts/run-api-tests.sh         # Run against existing server
./scripts/run-api-tests.sh --start # Start server, run tests, stop server
```

### Tuning Lab

Located in `shell_scripts/`:

**`run-tuning-lab.sh` / `.bat`** - Launch Forge with tuning tab
```bash
./scripts/run-tuning-lab.sh           # Start with optimizations + tuning tab
./scripts/run-tuning-lab.sh --no-opt  # Start without optimizations
```

The tuning tab provides empirical parameter optimization tools for I2V chaining, orbital camera paths, and more. See `docs/TUNING.md` for comprehensive tuning documentation.

## Update Deforum Settings

⚠️ **CRITICAL**: Old settings files will NOT work. Download the new format:

```bash
# Download to webui root directory
wget https://raw.githubusercontent.com/Tok/sd-forge-deforum/main/deforum/config/default_settings.txt
# Rename to deforum_settings.txt
mv default_settings.txt deforum_settings.txt
```

Then in Deforum UI: Settings File field → Load All Settings

**Recommendation**: Use Forge's Settings → Defaults to save your custom presets.

## Wan AI Video Generation ✨

Full integration guide: [docs/wan/README.md](docs/wan/README.md)

### Quick Setup

```bash
# Wan 2.2 TI2V Models (Text-to-Video & Image-to-Video)
# Recommended: TI2V-5B (16GB VRAM with CPU offload) - Unified T2V+I2V
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir models/Deforum/wan/Wan2.2-TI2V-5B

# High Quality: TI2V-A14B (24GB+ VRAM) - MoE, highest quality
huggingface-cli download Wan-AI/Wan2.2-TI2V-A14B-Diffusers --local-dir models/Deforum/wan/Wan2.2-TI2V-A14B

# Note: Wan 2.1 FLF2V models are used internally for interpolation and auto-discovered

# Optional: Qwen for AI prompt enhancement (auto-downloads on first use)
# Stored in: webui/models/Deforum/qwen/
# Auto-selected: 3B (4GB), 7B (8GB), 14B (16GB+)
```

### Deforum Integration Features
- **Prompt Scheduling**: Frame-perfect timing from Prompts tab
- **FPS Sync**: Single FPS setting controls both Deforum and Wan
- **Seed Scheduling**: Optional per-clip seed control
- **I2V Chaining**: Last frame → next clip init for seamless transitions
- **Qwen Enhancement**: AI prompt expansion with movement analysis

### FLF2V Configuration (Critical)

**Guidance Scale Settings:**
- **3.5** (Default) - Smooth visual morphing ⭐
- **5.5** - Balanced (official example)
- **7.0+** - Risk of jitter/flicker ⚠️
- **0.0** - ❌ **NEVER USE** - Breaks interpolation completely!

**Why 0.0 Breaks**: Disables ALL conditioning including `last_image`, causing first-frame extension instead of interpolation.

**Prompt Modes:**
- `"none"` - No text (recommended for smooth transitions)
- `"keyframe"` - Use keyframe prompts
- `"interpolated"` - Blend between keyframes

## Audio Event Detection 🎵

### Automatic Synchronization

The Audio Sync feature automatically detects musical events and places keyframes:

1. **Upload/Configure Audio**:
   - Init tab → Audio Sync sub-tab
   - Upload file or enter path/URL

2. **Detect Events**:
   - Click "🎵 Synchronize Audio to Keyframe Prompts"
   - Adjust sensitivity with ±5% buttons
   - Preview detected events

3. **Generate Prompts** (optional):
   - Prompts tab → AI Prompt Enhancement accordion
   - Set count, mode (escalating/cyclical/thematic), intensity
   - Click "💡 Generate Prompts with local Qwen"
   - Edit generated prompts before rendering

4. **Render**:
   - Events automatically become keyframes
   - Prompts sync to detected beats/onsets

### Detection Methods
- **Onsets**: Note attacks and transients
- **Beats**: Rhythmic pulse tracking
- **Bass**: Low-frequency energy (kick drums)
- **Combined**: Multi-method fusion

## Default Bunny Test

After installation, test with the default bunny animation:

1. Set Distribution → Keyframes Only
2. Set Animation Mode → 3D
3. Click Generate

**What happens:**
- Downloads Depth-Anything V2 on first run
- Generates 333 frames at 720p, 60 FPS
- Only 19 frames diffused (keyframes at prompt boundaries)
- Synced to amen break beat (enable sound in settings)

https://github.com/user-attachments/assets/5f637a04-104f-4d87-8439-15a386685a5e

## Troubleshooting

### Wan 2.1 Issues
* **No models found**: Download using commands above to `models/Deforum/wan/`
* **Generation fails**: Try 1.3B model, check VRAM
* **Flash attention errors**: Automatic fallback should work
* **Audio sync problems**: Verify prompt frame numbers

### AI Enhancement Issues
* **Model download fails**: Check internet, models go to `webui/models/Deforum/qwen/`
* **Out of VRAM**: Use "Cleanup Qwen Cache" or select 3B model
* **Slow enhancement**: Use smaller model (3B/7B instead of 14B)

### Audio Sync Issues
* **No events detected**: Increase sensitivity or try different detection method
* **Too many events**: Decrease sensitivity with -5% button
* **Wrong timing**: Verify FPS matches your target frame rate

### Settings File
Old `deforum_settings.txt` files **will not work**. Download new format:
https://github.com/Tok/sd-forge-deforum/blob/main/deforum/config/default_settings.txt

### General Issues
* **Import errors**: Restart WebUI completely (Ctrl+C → relaunch)
* **Missing dependencies**: `pip install -r requirements.txt`
* **Performance issues**: Check VRAM, reduce resolution/frame count

## Testing

Comprehensive test suite with unit and integration tests:

```bash
cd extensions/sd-forge-deforum

# Unit tests (fast, no server required)
pytest tests/unit/ -v

# Integration tests (requires Forge server)
pytest tests/integration/ --start-server
```

See [tests/README.md](tests/README.md) for detailed testing documentation.

## Documentation

- **[CLAUDE.md](CLAUDE.md)** - Developer guide for AI assistants
- **[CODING_GUIDE.md](CODING_GUIDE.md)** - Code standards and patterns
- **[Wan User Guide](docs/wan/README.md)** - Complete Wan setup
- **[Technical Reference](docs/wan/TECHNICAL.md)** - Developer docs

## Credits

- **Original Deforum**: [deforum-art/sd-forge-deforum](https://github.com/deforum-art/sd-forge-deforum)
- **Wan 2.1**: Alibaba Tongyi Vision Intelligence Lab
- **Qwen**: Alibaba Cloud
- **FILM**: Google Research
- **RIFE**: Megvii Research
- **Depth-Anything V2**: DepthAnything Team
- **Camera Shakify**: EatTheFuture (CC0 license)

## License

AGPL-3.0 (same as original Deforum)

Integrated third-party components retain their original licenses (see respective directories).
