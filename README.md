
# Fluxabled Fork of the Deforum Stable Diffusion Extension for Forge

Experimental fork of the [Deforum extension](https://github.com/deforum-art/sd-forge-deforum)
for [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge), 
fix'd up to work with Flux and integrate Parseq keyframe redistribution logic.

## Current status

&#x26A0;&#xFE0F; This fork of the extension is _basically working_ but still **extremely experimental**.
It contains temporary shortcuts, evil workarounds and dirty quick-fixes.

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

#### Flux Dev
  * get `flux1-dev-bnb-nf4-v2.safetensors` from huggingface and put it into your `<forge_install_dir>/models/Stable-diffusion/Flux`
    * https://huggingface.co/lllyasviel/flux1-dev-bnb-nf4/blob/main/flux1-dev-bnb-nf4-v2.safetensors
  * get the following 3 files from huggingface and put them into `<forge_install_dir>/models/VAE`
    * `ae.safetensors`
      * https://huggingface.co/black-forest-labs/FLUX.1-schnell/tree/main
    * `clip_l.safetensors`
      * https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main
    * `t5xxl_fp16.safetensors`
      * https://huggingface.co/comfyanonymous/flux_text_encoders/tree/main
  * switch UI to "flux", select the flux checkpoint and set all the 3 VAEs in "VAE / Text Encoder" in the same order as listed above (happens to be alphabetical).
  * make sure CFG (or "Scale" if you use Parseq) is at around 1.0
  * Make sure to remove the negative prompts!
    * &#x26A0;&#xFE0F; Warning: since Flux doesn't support negative prompts, "--neg" is currently ignored and everything after that is interpreted positively. 
      It may cause the opposite of their original intention.
  * Disable "Kohya HR Fix" and "FreeU".

#### Parseq Keyframe Redistribution
  * May be activated in the Parseq tab
  * Causes the rendering to run on an (even more) experimental core that can rearrange keyframes, which allows for precise Parseq sync at high cadence, making it possible to set up really fast generations with less jitter.
  * Additional details on how to use it can be found on the Parseq tab.

### Currently not working and may need to be disabled
* Kohya HR Fix
* FreeU

### Perhaps working (untested)
* Control Net
* Hybrid Video
* Non-Flux workflows

## Struggle shooting

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

### Other problems with torch

In case of other problems related to Torch, try to reinstall it as recommended on the pytorch website:
https://pytorch.org/get-started/locally/
