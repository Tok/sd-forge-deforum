
# Fluxabled Fork of the Deforum Stable Diffusion Extension for Forge

Experimental fork of the [Deforum extension](https://github.com/deforum-art/sd-forge-deforum)
for [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge), 
fix'd up to work with Flux.1 and integrate Parseq keyframe redistribution logic.

## Current status

This fork of the extension is _basically working_.

&#x26A0;&#xFE0F; Configurations that use the **experimental render core** by enabling the 
new keyframe distribution feature, may require that some unsupported features are being kept disabled.

## Installation

### Requirements

#### Python 3.10.6
Forge currently recommends Python version 3.10.6: https://www.python.org/downloads/release/python-3106/

In case you insist on trying to use a newer version (not recommended), see comments below about `basicsr`.

#### SD WebUI Forge
Install the latest version of [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) and delete any old or alternative versions of this plugin
you may have had installed before (=remove this directory if it already exists: `<forge_install_dir>/extensions/sd-forge-deforum`).

### Install this Extension

#### Directly in Forge

Go to tab "Extensions" - "Install from URL" and use this URL: https://github.com/Tok/sd-forge-deforum.git

#### Or from the commandline

Open commandline and run `<forge_install_dir>/venv/Scripts/activate.bat` 
to activate the virtual environment (venv) for Python used by Forge.

With the venv from Forge activated, do:

```
cd <forge_install_dir>/extensions
git clone https://github.com/Tok/sd-forge-deforum
cd sd-forge-deforum
pip install -r requirements.txt
```


## What should work, what doesn't and what's untested

### Should work:

#### Flux.1 Dev
  * get `flux1-dev-bnb-nf4-v2.safetensors` from huggingface and put it into your `<forge_install_dir>/models/Stable-diffusion/Flux`
    * https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/blob/main/flux1-dev-bnb-nf4-v2.safetensors
  * get the following 3 files from huggingface and put them into `<forge_install_dir>/models/VAE`
    * `ae.safetensors`
      * https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main
    * `clip_l.safetensors`
      * https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main
    * `t5xxl_fp16.safetensors`
      * https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main
  * Switch UI to "flux", select the flux checkpoint and set all the 3 VAEs in "VAE / Text Encoder".
  * 'CFG' (or "Scale" if you use Parseq) is expected to be at around 1.0
  * 'Distilled CFG' at 3.5
  * Negative prompts are ignored.

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

### Perhaps working (untested)
* Flux schnell
  * There's not a lot of precision for fine-tuning strength values when only 4 steps are required.
* Control Net
* Hybrid Video
* Non-Flux workflows

### Currently not working with experimental core
* Seed distributions
  * Setting is ignored in favor of random seeds.
  * But should be fine when seeds are provided by parseq
  * To be fixed in future version..
* Kohya HR Fix
  * may need to be left disabled
* FreeU
  * may need to be left disabled

### Other Stuff
* Includes a new default setup.
  * optimized for generating clips at 60 FPS in 720p, using Flux.1 with the new render core.
  * is using `keyframes only` keyframe distribution.
  * backup and replace your existing  with the example provided here.
  * partially hardcoded, but also available as updated `deforum_settings.txt`
* Non-essential emojis can be turned off with a checkbox under "Settings" - "Deforum".

##### Example Clip (enable sound):
https://github.com/user-attachments/assets/9d7e511c-d109-4b8b-a5e4-e1540dfc5f17

## Troubleshooting

### Potential problems with different Python-, Torch- and 'basicsr' versions

Forge currently recommends Python version 3.10.6 and the torch versions that it runs.

Newer versions of Python or different torch combinations may refuse to install the 'basicsr'
dependency which is currently required by this extension.

In case you get any error related to 'basicsr' during the installation, and you don't want to set up a new 
Python 3.10.6 venv (as would be recommended), you can try the installation from commandline as described above, 
but replacing the `basicsr` version with version `1.4.2` in `requirements.txt` before installing the requirements.

If the extension was installed successfully, but there is an error related to `basicsr` while starting Forge,
try the same directly with `pip uninstall basicsr` and then `pip install basicsr==1.4.2` (while venv is active).

*TL;DR:* This extension should be able to run with either version `1.3.5` or `1.4.2` of `basicsr`, 
but not all versions may install or run properly with all combos of Python and Torch.

### ModuleNotFoundError: No module named 'torchvision.transforms.functional_tensor'
May happen when there is a version conflict with `basicsr` and `torchvision`.
A workaround can be found here: https://github.com/Tok/sd-forge-deforum/issues/1#issuecomment-2318333572

### Other problems with torch

In case of other problems related to Torch, try to reinstall it as recommended on the pytorch website:
https://pytorch.org/get-started/locally/

### Settings file
The content of the `deforum_settings.txt` file has been updated.
Settings from older versions may not behave as expected.
