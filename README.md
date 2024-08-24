
# Fluxabled Fork of the Deforum Stable Diffusion Extension for Forge

Experimental fork of the Deforum extension for the [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge), 
fix'd up to work with Flux and integrate Parseq keyframe redistribution logic.

## Current status

This fork of the extension is &#x26A0;&#xFE0F;**extremely experimental**&#x26A0;&#xFE0F;.
It contains temporary shortcuts, evil workarounds and dirty quick-fixes.

## Installation

First install the latest version of [Stable Diffusion WebUI Forge](https://github.com/lllyasviel/stable-diffusion-webui-forge) and delete any old or alternative versions of this plugin
you may have had installed before (=remove this directory if it already exists: `<forge_install_dir>/extensions/sd-forge-deforum`).

### Activate Python venv
Depending on your setup, you may need to make sure that the virtual environment for Python that's used Forge is active by running `<forge_install_dir>/venv/Scripts/activate.bat`.
It should be visible on your commandline when Forges venv is active.

### Install the Extension
With Forges venv active do:

    cd <forge_install_dir>/extensions
    git clone https://github.com/Tok/sd-forge-deforum
    cd sd-forge-deforum
    pip install -r requirements.txt

You can also try to install it directly in Forge on tab "Extensions" - "Install from URL", after making sure that any old versions were removed properly: https://github.com/Tok/sd-forge-deforum.git


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
  * &#x26A0;&#xFE0F; Make sure to do a single generation in the `Txt2img` tab before starting your Deforum generation. It's currently required to be done like this, so the models init properly.

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
