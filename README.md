# Fluxabled Fork of the Deforum Stable Diffusion Extension for Forge

Experimental fork of the [Deforum extension](https://github.com/deforum-art/sd-forge-deforum)
for [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge), 
fix'd up to work with Flux.1 and integrate Parseq keyframe redistribution logic.
Integrates dynamic camera shake effects with data sourced from EatTheFutures 'Camera Shakify' Blender plugin.

## Current status

This fork of the extension is _basically working_.

&#x26A0;&#xFE0F; Configurations that use the **experimental render core** by enabling the 
new keyframe distribution feature, may require that some unsupported features are being kept disabled.


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

### ControlNet with FLUX

This version includes comprehensive patches to enable compatibility between FLUX and ControlNet for animation rendering.

#### FLUX ControlNet Models

For best results with FLUX animations, use specially designed FLUX ControlNet models:

* **Official Models**: Download from [XLabs-AI FLUX ControlNet Collection](https://huggingface.co/XLabs-AI/flux-controlnet-collections)
* **Installation**: Place models in your `models/ControlNet` directory
* **Recommended Models**:
  * `flux-canny-controlnet-v3.safetensors`
  * `flux-depth-controlnet-v3.safetensors` 
  * `flux-hed-controlnet-v3.safetensors`

#### Setup and Configuration

1. **Enable ControlNet** in WebUI Forge settings before using with Deforum
2. **In Deforum's ControlNet Tab**:
   * Enable "Optimize ControlNet execution" to prevent memory issues
   * Set processor resolution to 720px or 1024px for best results with FLUX
   * If you experience issues, try a different input preprocessing mode
   * **New**: Support for up to 5 ControlNet units (previously some configs had issues with unit 5)

3. **Video Input Processing**:
   * You can use video files as ControlNet inputs
   * Enable "Batch Input" and provide the video path
   * Input videos will be automatically processed frame-by-frame

#### ⚠️ EXPERIMENTAL: WebUI Forge Compatibility Patching ⚠️

The extension includes a comprehensive patching system in `scripts/deforum_helpers/forge_controlnet_patcher.py`, `flux_model_utils.py` and `preload.py` that modifies WebUI Forge's ControlNet implementation for FLUX compatibility. These are **highly experimental** modifications:

* **Advanced Monkey Patching**: The extension dynamically modifies WebUI Forge's code at runtime to:
  * Fix missing `resize_mode` attribute errors
  * Handle type conversion issues in image processing functions (HWC3 assertion errors)
  * Fix model loading and recognition failures
  * Ensure dictionary fields exist before access
  * Fix unit 5 attribute initialization
  * Implement fallbacks for failed operations
  * **New**: Multi-level HWC3 patching to prevent assertion errors with different image formats

* **Files Modified**:
  * `/extensions-builtin/sd_forge_controlnet/scripts/controlnet.py` may be directly patched
  * Core WebUI processing classes are modified with additional methods
  * HWC3 function patched to handle more image formats safely
  * **New**: Direct file editing of utils.py to fix problematic assertions
  * **New**: Direct patching of ControlNet to fix 'NoneType' has no attribute 'strength' errors
  * **New**: Improved preprocessor handling to prevent 'process_after_every_sampling' errors

* **Recent Improvements**:
  * Fixed missing attributes for ControlNet unit 5
  * Added multiple levels of HWC3 patching to prevent assertion errors
  * Runtime monkey patching via `builtins` for global access to safe functions
  * Implemented import hooks to catch and patch dynamically loaded modules
  * **New**: Added early initialization system for more reliable patching
  * **New**: Direct editing of ControlNet script files with safety checks
  * **New**: Fixes for VAE initialization errors with WebUI Forge

* **Potential Impacts**:
  * These patches may interfere with other extensions that use ControlNet
  * You may see error messages in the console that are actually handled by the patches
  * Memory usage patterns might be affected
  * Future WebUI Forge updates could break compatibility

* **Debug Information**:
  * Enable extra logging with environment variable: `DEFORUM_DEBUG_CONTROLNET=1`
  * Key patch files: `forge_controlnet_patcher.py`, `flux_model_utils.py`, and `preload.py`

**If you encounter issues with other extensions**, try disabling either this extension or the conflicting one to isolate the problem.

### Perhaps working (untested)
* Flux schnell
  * There's not a lot of precision for fine-tuning strength values when only 4 steps are required.
* Hybrid Video
* Non-Flux workflows

### Currently not working with experimental core
* Kohya HR Fix
  * may need to be left disabled
* FreeU
  * may need to be left disabled
* Some advanced ControlNet features

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

### ControlNet issues
If you encounter problems with ControlNet and FLUX:

1. **Model Loading Errors**:
   * Error message: `Recognizing Control Model failed`
   * **Solutions**: 
     * Ensure model files are in the correct location (`models/ControlNet` directory)
     * Try using the basename of the model without extension (e.g., `flux-canny-controlnet-v3` instead of full path)
     * Restart WebUI Forge after adding new models
     * If FLUX models fail, the extension will attempt to use standard ControlNet models as fallbacks

2. **Memory Issues**:
   * Error messages: `CUDA out of memory` or `RuntimeError: CUDA error`
   * **Solutions**:
     * Enable "Optimize ControlNet execution" in the Deforum ControlNet tab
     * Reduce the number of active ControlNet units 
     * Lower the preprocessing resolution
     * Enable "Low VRAM" mode in advanced settings

3. **Image Processing Errors**:
   * Error messages: `HWC3 assertion error` or `Image format not supported`
   * **Solution**: The extension includes multiple patches to handle these errors automatically
   * These errors typically show in the console but are handled by the patching system

4. **Missing Attributes Errors**:
   * Error message: `'StableDiffusionProcessingTxt2Img' object has no attribute 'resize_mode'`
   * **Solution**: The extension adds this attribute dynamically - error messages are intercepted
   * You may see these in logs but they shouldn't affect rendering

5. **Pydantic Schema Generation Errors**:
   * Error messages: `PydanticSchemaGenerationError: Unable to generate pydantic-core schema for <class 'inspect._empty'>`
   * **Solution**: The extension includes a patch that automatically configures Pydantic models with `arbitrary_types_allowed=True` 
   * The patch is applied to the WebUI Forge API models during extension startup
   * This fixes UI errors related to script loading and model configuration

6. **ControlNet Script Not Found Errors**:
   * Error message: `Exception: Script not found: ControlNet`
   * **Solution**: The extension implements an improved script finding algorithm that:
     * Searches for ControlNet using multiple naming conventions
     * Looks through module hierarchy to find by partial name matching 
     * Works with WebUI Forge's unique extension layout
   * This ensures Deforum can properly locate and use ControlNet in WebUI Forge

7. **IndentationError in ControlNet Script**:
   * Error message: `IndentationError: expected an indented block after 'if' statement on line 414`
   * **Solution**: The extension includes multiple safeguards:
     * Sets `IGNORE_TORCH_INDENT_WARNING=1` environment variable
     * Adds proper indentation when patching the ControlNet script
     * Maintains consistent code style when modifying the script file
   * These changes prevent syntax errors when patching the assertion checks

8. **Debugging ControlNet Issues**:
   * Set environment variable `DEFORUM_DEBUG_CONTROLNET=1` before starting WebUI
   * Look for messages with prefix `[Deforum ControlNet Patcher]` or `[Deforum FLUX Utils]`
   * Check if models are properly registered at startup

## Technical Implementation

Deforum addresses various technical challenges when working with ControlNet and other modules:

1. A Forge ControlNet patching system (`forge_controlnet_patcher.py`) provides compatibility fixes for WebUI Forge, including:
   * HWC3 assertion failures in image processing
   * Dictionary KeyErrors in ControlNet scripts
   * Missing ControlNet script indices
   * Indentation errors in built-in scripts
   * Model loading failures with FLUX models
   * Pydantic schema generation errors

2. The patching system applies multiple levels of protection:
   * Environment variable configuration for WebUI Forge
   * Direct script modifications to fix syntax issues
   * Runtime monkey patching for dynamic fixes
   * Fallback mechanisms when models can't be loaded

3. FLUX model support through dedicated utilities:
   * Automatic registration of FLUX models at startup
   * Alternative model finding when exact models aren't available
   * Special handling for different model naming conventions
