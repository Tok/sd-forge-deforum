# Fluxabled Fork of the Deforum Stable Diffusion Extension for Forge

Experimental fork of the [Deforum extension](https://github.com/deforum-art/sd-forge-deforum)
for [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge), 
fix'd up to work with Flux.1 and integrate Parseq keyframe redistribution logic.
Integrates dynamic camera shake effects with data sourced from EatTheFutures 'Camera Shakify' Blender plugin.

## Current status

This fork of the extension is _basically working_.

&#x26A0;&#xFE0F; Configurations that use the **experimental render core** by enabling the 
new keyframe distribution feature, may require that some unsupported features are being kept disabled.

## Wan 2.1 Video Integration

This fork includes comprehensive support for **Wan 2.1 video generation** with advanced Flow Matching architecture. 
WAN provides state-of-the-art text-to-video and image-to-video generation capabilities using a completely isolated 
environment to prevent conflicts with WebUI's diffusion systems.

### ⚠️ Important: Repository Setup Required

**WAN requires the official repository to be cloned as a submodule:**

```bash
cd <forge_install_dir>/extensions/sd-forge-deforum
git clone https://github.com/Wan-Video/Wan2.1.git wan_official_repo
```

The system will automatically handle this on first run, but manual setup ensures better reliability.

### 🔧 WAN System Architecture

**Isolated Environment Design:**
- **Complete Isolation**: WAN runs in a separate Python environment with its own dependencies
- **No Conflicts**: Zero interference with WebUI's diffusion pipelines or transformers
- **Flow Matching**: Uses WAN's native Flow Matching framework (NOT traditional diffusion)
- **Official Code**: Integrates with the actual WAN 2.1 repository for authentic inference
- **Auto-Setup**: Downloads missing HuggingFace components (tokenizer, text encoder) automatically

**Key Components:**
- **T5 Text Encoder**: Multilingual text understanding and conditioning
- **3D Causal VAE (Wan-VAE)**: Advanced video encoding/decoding with temporal consistency
- **Flow Matching Pipeline**: State-of-the-art generative framework for smooth video synthesis
- **Cross-Attention Mechanisms**: Precise text-to-video conditioning throughout generation

### 📋 WAN Setup Requirements

#### 1. **System Requirements**
- **GPU Memory**: 12GB+ VRAM (16GB recommended for 720p generation)
- **Storage**: ~10GB for models and repository
- **Python**: 3.10+ (included with Forge)
- **Git**: Required for repository management

#### 2. **Model Files Setup**
Place your WAN model files in: `<forge_install_dir>/models/wan/`

**Supported Model Formats:**
- Sharded SafeTensors: `diffusion_pytorch_model-00001-of-00007.safetensors` (etc.)
- Single SafeTensors: `diffusion_pytorch_model.safetensors`
- PyTorch Checkpoints: `model.bin`, `pytorch_model.bin`

#### 3. **Directory Structure (Auto-Created)**
```
<forge_install_dir>/extensions/sd-forge-deforum/
├── wan_official_repo/           # Official WAN 2.1 repository (auto-cloned)
│   ├── wan/                     # Core WAN modules
│   │   ├── text2video.py        # Text-to-video generation
│   │   ├── image2video.py       # Image-to-video generation
│   │   └── modules/             # WAN model components
│   ├── generate.py              # Main generation script
│   └── requirements.txt         # WAN dependencies
├── wan_isolated_env/            # Isolated Python environment
│   ├── models/                  # Prepared model structure
│   └── site-packages/           # Isolated WAN dependencies
└── scripts/deforum_helpers/
    ├── wan_integration.py       # Main WAN integration
    ├── wan_isolated_env.py      # Environment management
    └── wan_flow_matching.py     # Flow Matching pipeline
```

### 🎬 Using WAN Video Generation

#### 1. **Basic Setup**
1. Set **Animation Mode** to "Wan Video"
2. Configure **Model Path**: Point to your WAN model directory
3. Set **Resolution**: Choose from 720p, 480p, or portrait variants
4. Configure **Generation Settings**: FPS, duration, steps, guidance scale

#### 2. **Prompt Configuration**
Use **standard Deforum prompt format** in the Prompts tab:
```json
{
    "0": "A cute bunny hopping on grass, photorealistic",
    "12": "A bunny with sunglasses at a neon construction site", 
    "43": "A cyberpunk bunny with glowing eyes on a digital grid"
}
```

The system automatically calculates:
- **Clip 1**: Frames 0-11 (12 frames, 0.2s @ 60fps)
- **Clip 2**: Frames 12-42 (31 frames, 0.5s @ 60fps)
- **Clip 3**: Frames 43+ (remaining duration)

#### 3. **Generation Modes**
- **Text-to-Video**: First clip generation from text prompt
- **Image-to-Video**: Subsequent clips use the last frame from previous clip as init image
- **Automatic Transitioning**: Seamless clip-to-clip continuity

### ⚙️ WAN Configuration Options

#### **Basic Settings**
- **Clip Duration**: 2-8 seconds per clip (4s default)
- **FPS**: 15-60 fps (60fps recommended)
- **Resolution**: 
  - `1280x720` (16:9 landscape, best quality)
  - `720x1280` (9:16 portrait, social media)
  - `854x480` (16:9 landscape, faster)
  - `480x854` (9:16 portrait, fastest)

#### **Quality Settings**
- **Inference Steps**: 20-80 (50 default, lower = faster)
- **Guidance Scale**: 1.0-20.0 (7.5 default, higher = more prompt adherence)
- **Motion Strength**: 0.1-2.0 (1.0 default, higher = more movement)

#### **Advanced Options**
- **Frame Overlap**: Blend frames between clips for smoother transitions
- **Interpolation**: AI-enhanced frame interpolation for smoother motion
- **Seed Control**: Reproducible generation with fixed seeds

### 🚨 Troubleshooting

#### **Common Issues & Solutions**

**1. "WAN repository not properly set up"**
```bash
# Manually clone the repository:
cd <forge_install_dir>/extensions/sd-forge-deforum
rm -rf wan_official_repo
git clone https://github.com/Wan-Video/Wan2.1.git wan_official_repo
```

**2. "No WAN model class found"**
- Ensure the repository has been cloned correctly
- Check that `wan_official_repo/wan/text2video.py` exists
- Restart WebUI after repository setup

**3. "Out of GPU Memory"**
- Reduce resolution: Use 480p variants
- Lower inference steps: Try 20-30 steps
- Reduce clip duration: Use 2-3 second clips
- Close other GPU applications

**4. "Model loading failed"**
- Verify model files are in correct directory: `models/wan/`
- Check model file format (SafeTensors recommended)
- Ensure sufficient storage space (models can be 10GB+)

**5. "Import errors in isolated environment"**
- Delete isolation directory: `rm -rf wan_isolated_env`
- Restart generation (will rebuild environment)
- Check internet connection for HuggingFace downloads

#### **Performance Optimization**

**For Speed:**
- Use 480p resolution
- Lower inference steps (20-30)
- Shorter clip duration (2-3s)
- Disable frame interpolation

**For Quality:**
- Use 720p resolution
- Higher inference steps (50-80)
- Higher guidance scale (10-15)
- Enable frame overlap

**For Memory Efficiency:**
- Use portrait resolutions (less pixels)
- Generate shorter clips
- Close unnecessary applications

### 🔍 Environment Isolation Details

WAN uses a sophisticated isolation system to prevent conflicts:

**Isolated Dependencies:**
- Separate Python package installation in `wan_isolated_env/site-packages/`
- Version-specific transformers, accelerate, and torch packages
- No interference with WebUI's diffusion libraries

**Dynamic Import Management:**
- Temporary `sys.path` modification during WAN generation
- Context managers ensure clean environment switching
- Complete restoration of original state after generation

**Repository Integration:**
- Automatic discovery of WAN modules in official repository
- Dynamic import of `wan.text2video` and `wan.image2video`
- Graceful fallback if repository structure changes

**Fail-Fast Philosophy:**
- No placeholder or fallback generation
- Clear error messages with specific solutions
- Immediate termination if requirements not met
- No hidden dependencies or surprise behaviors

This architecture ensures that WAN can be seamlessly integrated into existing Deforum workflows without any 
impact on traditional diffusion-based generation modes.

## Requirements

### Get SD WebUI Forge
Install, update and run the 'one-click installation package' of
[Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge
as described. Includes:
* Python 3.10.6
* CUDA 12.1
* Pytorch 2.3.1

Other versions _may_ work with this extension, but have not been properly tested.

### Run Flux on Forge

Get `flux1-dev-bnb-nf4-v2.safetensors` from huggingface and put it into your `<forge_install_dir>/models/Stable-diffusion/Flux`:
https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/blob/main/flux1-dev-bnb-nf4-v2.safetensors

Get the following 3 files from huggingface and put them into `<forge_install_dir>/models/VAE`
* `ae.safetensors` https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main
* `clip_l.safetensors` https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main
* `t5xxl_fp16.safetensors` https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main

Restart Forge, set mode to "flux", select the flux checkpoint and all the 3 VAEs in "VAE / Text Encoder" and test with Txt2Img.

## Installation

### Directly in Forge (recommended)

Go to tab "Extensions" - "Install from URL" and use this: https://github.com/Tok/sd-forge-deforum.git

### From the commandline

Open commandline and run `<forge_install_dir>/venv/Scripts/activate.bat` 
to activate the virtual environment (venv) for Python used by Forge.

With the venv from Forge activated, do:
```
cd <forge_install_dir>/extensions
git clone https://github.com/Tok/sd-forge-deforum
cd sd-forge-deforum
pip install -r requirements.txt
```

### Update Deforum Settings

Get the latest default-settings.txt and place it directly into your 'webui' directory, then click "Load All Settings":
https://raw.githubusercontent.com/Tok/sd-forge-deforum/main/scripts/default_settings.txt
Rename it to `deforum_settings.txt` (or whatever matches the name of your settings file in the UI) and put it directly into your 'webui' directory.

&#x26A0;&#xFE0F; Some Settings are currently not properly loaded or are not persisted 
in `default_settings.txt` and may need to be set manually the first time:
* Tab "Prompts" - "Prompts negative" not resetting
  * Consider removing the defaults because they're not used with Flux.

Recommendation: **Use ForgeUIs "Settings" - "Defaults" to save your settings.**

## Default Bunny Testrun

After installation, you can test the setup by generating the default bunny with
"Distribution" set to "Keyframes Only" and "Animation Mode" set to "3D".
This also downloads the MiDaS model for depth warping when ran for the first time
and demonstrates prompt synchronization in a no-cadence setup.

The default bunnies contain 333 frames at 720p, but only 19 of them are actually diffused.
The diffused frames are placed in the clip according to the keyframes defined in the prompts.
The prompts themselves are aligned to be synchronized at 60 FPS with the beat of an 
amen break you can find linked in the settings (enable sound):

https://github.com/user-attachments/assets/5f637a04-104f-4d87-8439-15a386685a5e

If you used other versions of the Deforum plugin before, it may also be necessary
to update or adjust your Deforum settings. The latest example settings with for the default bunny can also be downloaded here:

https://github.com/Tok/sd-forge-deforum/blob/main/scripts/default_settings.txt



## What should work, what doesn't and what's untested

### Should work:
#### Keyframe Distribution
Causes the rendering to run on an experimental core that can rearrange keyframes,
which makes it possible to set up fast generations with less jitter at high or no cadence.

##### New sub-tab under "Keyframes"
* Can now be used **with- or without- Parseq**.
* Allows for precise sync at high cadence.
* Detailed info and recommendations on new tab.

#### Asynchronous Subtitle generation
All subtitles are now generated and written to an .srt file in advance.
Complex subtitle generations should work fine with Parseq but are currently limited with Deforum-only setups.
* New Deforum setting for skipping the prompt-part to a new line in .srt files.
* New Deforum setting for choosing simple (non-technical) subtitles that contain only the text from the prompt.
  * Complex subtitles should work fine when Parseq is used, but are otherwise limited to essential information only.
  * Recommendation: turn on for now if not using Parseq
* Removed emtpy "--neg" param from being written into the subtitles
  because negative prompts are ignored in Flux workflows.
* Improved padding of technical information so subtitles jitter less.

### Camera Shakify Effects

Add camera shake effects to your renders on top of your other movement.

##### New sub-sub-tab under "Keyframes"

This feature enhances the realism of your animations by simulating natural camera movements, adding a layer of depth
and engagement to your visuals. Perfect for creating action sequences or adding a sense of spontaneity, 
it allows for customizable shake parameters to fit your specific needs.

The shake data is available under Creative Commons CC0 1.0 Universal license and was sourced from the
['Camera Shakify' Blender plugin by EatTheFuture](https://github.com/EatTheFuture/camera_shakify).


### Perhaps working (untested)
* Flux schnell
  * There's not a lot of precision for fine-tuning strength values when only 4 steps are required.
* Control Net
* Hybrid Video
* Non-Flux workflows

### Currently not working with experimental core
* Kohya HR Fix
  * may need to be left disabled
* FreeU
  * may need to be left disabled
* Control Net

### Other Stuff
* Includes a new default setup to generate default bunny at 60 FPS in 720p with keyframes only.
* Non-essential emojis can be turned off with a checkbox under "Settings" - "Deforum".
* Seed and Subseed tabs unified.

## Troubleshooting

### Settings file
During active development, content and structure of the `deforum_settings.txt` file 
can change quickly been updated. Settings from older versions may not behave as expected.
If necessary, the latest deforum-settings.txt are available for download here:
https://github.com/Tok/sd-forge-deforum/blob/main/scripts/default_settings.txt