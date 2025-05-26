# Fluxabled Fork of the Deforum Stable Diffusion Extension for Forge

Experimental fork of the [Deforum extension](https://github.com/deforum-art/sd-forge-deforum)
for [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge), 
fix'd up to work with Flux.1, integrate Parseq keyframe redistribution logic, and support **WAN 2.1 AI Video Generation**.
Integrates dynamic camera shake effects with data sourced from EatTheFutures 'Camera Shakify' Blender plugin.

## Current status

This fork of the extension is _basically working_.

&#x26A0;&#xFE0F; Configurations that use the **experimental render core** by enabling the 
new keyframe distribution feature, may require that some unsupported features are being kept disabled.

## Requirements

### Get SD WebUI Forge
Install, update and run the 'one-click installation package' of
[Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge)
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

## WAN 2.1 AI Video Generation ✨

### **NEW: Precision Text-to-Video Generation**

The extension now includes **WAN 2.1** (Alibaba's state-of-the-art video generation model) as an additional function alongside traditional Deforum animation.

#### 🎯 **Key Features**
- **Frame-Perfect Timing**: Uses exact frame differences from prompt schedules for audio sync
- **Auto-Discovery**: Automatically finds WAN models without manual configuration
- **Flash Attention Compatible**: Works with or without flash-attn for maximum compatibility
- **Memory Optimized**: Efficient handling of large video generations

#### 🚀 **Quick Setup**

1. **Download WAN Models** (choose one):
   ```bash
   # Recommended: 1.3B model (8GB+ VRAM)
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan
   
   # High Quality: 14B model (16GB+ VRAM) 
   huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/wan
   ```

2. **That's it!** The extension auto-discovers models in:
   - `models/wan/`
   - `models/WAN/` 
   - HuggingFace cache
   - Downloads folder

#### 🎬 **Usage**
Access WAN video generation through the **WAN** tab in Deforum. Set up frame-based prompts for precise timing:

```json
{
  "0": "A serene mountain landscape at dawn",
  "30": "Morning mist rising from the valleys", 
  "60": "Golden sunlight breaking through clouds",
  "90": "Full daylight illuminating the peaks"
}
```

Creates exactly 30, 30, 30 frames respectively - perfect for audio synchronization!

#### 🛠️ **Compatibility Layer**
- **No Flash Attention Required**: Automatically falls back to PyTorch native attention
- **Original WAN Repo Untouched**: Uses compatibility layer without modifying official code
- **Robust Error Handling**: Clear error messages and troubleshooting guidance

#### 📊 **Model Comparison**
| Model | Size | VRAM | Speed | Quality | Best For |
|-------|------|------|--------|---------|----------|
| **T2V-1.3B** | ~17GB | 8GB+ | Fast | Good | Most Users ⭐ |
| **T2V-14B** | ~75GB | 16GB+ | Slow | Excellent | High-end Systems |

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

#### WAN 2.1 Video Generation
* **Text-to-Video**: High-quality AI video generation with precise frame timing
* **Auto-Discovery**: Automatic model detection and validation
* **Flash Attention Fallback**: Compatible with systems without flash-attn
* **Audio Synchronization**: Frame-perfect timing for music videos
* **Multiple Resolutions**: Support for various output sizes

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

### WAN 2.1 Issues
* **No models found**: Download WAN models using the commands above
* **Generation fails**: Try the 1.3B model if using 14B, check VRAM usage
* **Flash attention errors**: Compatibility layer should handle this automatically
* **Audio sync problems**: Verify frame numbers in prompt schedule match your timing needs

### Settings file
During active development, content and structure of the `deforum_settings.txt` file 
can change quickly been updated. Settings from older versions may not behave as expected.
If necessary, the latest deforum-settings.txt are available for download here:
https://github.com/Tok/sd-forge-deforum/blob/main/scripts/default_settings.txt

### General Issues
* **Import errors**: Restart WebUI after installation
* **Missing dependencies**: Run `pip install -r requirements.txt`
* **Performance issues**: Check VRAM usage and reduce settings
