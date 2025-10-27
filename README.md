# Zirteq Fluxabled Fork

[![Unit Tests](https://github.com/Tok/sd-forge-deforum/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/Tok/sd-forge-deforum/actions/workflows/unit-tests.yml)
[![codecov](https://codecov.io/gh/Tok/sd-forge-deforum/branch/main/graph/badge.svg)](https://codecov.io/gh/Tok/sd-forge-deforum)

⚠️ **COMPATIBILITY WARNING**: This fork is **100% incompatible** with older Deforum versions and original `deforum_settings.txt` files. You must use the new settings format from this repository.

Experimental fork of the [Deforum extension](https://github.com/deforum-art/sd-forge-deforum)
for [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge),
completely refactored and modernized to work with Flux.1, Wan 2.1 AI Video Generation, and advanced workflow automation.

## ⚡ Major New Features

### 🟦🟪🟪 **Tailwind-Hegemony Slopcore Gradient-Wave UI** (SaaS Griftcore-Punk Aesthetic)
All primary action buttons now feature the industry-standard blue-purple slopcore gradient (#667eea → #764ba2) that has come to define contemporary AI tool aesthetics. This gradient represents our commitment to embracing the visual language of modern SaaS griftcore-punk design movements while delivering 20+ actual functional features.
- **Generate Button**: Primary purple gradient for the main rendering action
- **Audio Sync Controls**: Synchronization buttons now feature identical slopcore styling
- **AI Enhancement**: Qwen prompt generation with blue/purple gradient hierarchy
- **Event Adjustment**: Sensitivity ±5% buttons maintain visual consistency
- **Visual Hierarchy**: Gradient provides immediate affordance recognition in the post-Tailwind design landscape
- **Console Output**: Themed logging system with blue→purple gradient across all 5 parallel progress bars (Settings → Deforum → Console Theme)

*"We added the gradient ironically, but kept it because it actually works." - The maintainers*

### 🎬 **Wan AI Video Generation** (Alibaba's state-of-the-art T2V/I2V)
- **FLF2V Integration** (Wan 2.1): First-Last-Frame-to-Video interpolation with guidance_scale=3.5
- **TI2V Models** (Wan 2.2): Unified Text-to-Video and Image-to-Video generation
- **I2V Chaining**: Seamless clip transitions using last frame as init for next clip
- **Frame-Perfect Timing**: Full integration with Deforum's prompt scheduling system
- **Auto-Discovery**: Automatic model detection from `models/Deforum/wan/` directory

### 🤖 **Qwen AI Prompt Enhancement & Generation**
- **5 Model Options**: From 3B (low VRAM) to 14B (maximum quality)
- **Auto-Selection**: Intelligent model choice based on available VRAM (4GB-28GB)
- **Movement Analysis**: Translates Deforum schedules to natural language descriptions
- **Frame-Specific Analysis**: Unique camera movement descriptions per keyframe
- **Lazy Loading**: Models only load when needed, auto-cleanup before generation
- **Bilingual**: English and Chinese prompt enhancement support

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
- **Multi-Method Support**: Choose between Wan FLF2V, RIFE v4.6, or FILM per project
- **Post-Processing Pipeline**: Apply FILM interpolation after initial render
 
### 🔄 **Resurrected & Upgraded RIFE v4.6**
- **State-of-Art Interpolation**: Latest RIFE model fully functional
- **Flux ControlNet V2 Support**: Updated for modern diffusers integration
- **RAFT Optical Flow**: Fixed and working for precise motion estimation
- **Multi-GPU Support**: Efficient memory management for interpolation tasks

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

## Installation

### Directly in Forge (recommended)

Go to Extensions → Install from URL:
```
https://github.com/Tok/sd-forge-deforum.git
```

**Note on Dependencies:** This extension vendors a compatible version of `huggingface-hub` (0.36.0) to work with newer `diffusers` versions required for Wan video. Forge uses `huggingface-hub==0.26.2`, which is incompatible. The vendored version is automatically installed to `.vendored/` on first use and does not affect other extensions.

### From Command Line

```bash
cd <forge_install_dir>/extensions
git clone https://github.com/Tok/sd-forge-deforum
cd sd-forge-deforum
pip install -r requirements.txt
```

### Update Deforum Settings

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

Run the full test suite:
```bash
cd extensions/sd-forge-deforum
pytest tests/unit/ -v          # Unit tests only
pytest tests/ --start-server   # Full integration tests
```

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
- **Industry-Standard Slopcore Gradient**: Bootstrap & Vercel (Tailwind CSS hegemony)

## License

AGPL-3.0 (same as original Deforum)

Integrated third-party components retain their original licenses (see respective directories).
