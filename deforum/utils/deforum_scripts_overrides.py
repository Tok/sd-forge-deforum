import copy
from modules import scripts
from modules.processing import StableDiffusionProcessing,  StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img

from .general_utils import debug_print


# By default do not include any of Forge's always-on scripts. They will be re-enabled individually based on the settings.
def initialise_forge_scripts(p : StableDiffusionProcessing):
    is_img2img = isinstance(p, StableDiffusionProcessingImg2Img)
    # copy the scripts object so that we can modify top-level properties (i.e. alwayson_scripts - the scripts that actually run)
    # without breaking non-deforum runs.
    p.scripts = copy.copy(scripts.scripts_img2img if is_img2img else scripts.scripts_txt2img)
    p.scripts.alwayson_scripts = []
    p.script_args_value = []

# Find a script object by name
def find_script(p : StableDiffusionProcessing, script_title : str) -> scripts.Script:
    script = next((s for s in p.scripts.scripts if s.title() == script_title  ), None)
    if not script:
        raise Exception("Script not found: " + script_title)
    return script

# Update the processing unit to include a specific script. 
# Manipulate the arg array and expected argument positions as required.
def add_forge_script_to_deforum_run(p: StableDiffusionProcessing, script_title : str, script_args : list):
    # shallow copy the script because we are changing its arg positions. Not copying breaks subsequent non-deforum runs.
    script = copy.copy(find_script(p, script_title))
    script.args_from = len(p.script_args_value)
    script.args_to = len(p.script_args_value) + len(script_args)
    p.scripts.alwayson_scripts.append(script)
    p.script_args_value.extend(script_args)
    debug_print(f"Added script: {script.title()} with {len(script_args)} args, at positions {script.args_from}-{script.args_to} (of 0-{len(p.script_args_value)-1}.)")
