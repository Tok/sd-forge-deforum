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
    # Initialize script search
    script = None
    
    # Make sure p has the scripts attribute
    if not hasattr(p, 'scripts') or p.scripts is None:
        debug_print("Processing object does not have scripts attribute")
        p.scripts = copy.copy(scripts.scripts_txt2img if not isinstance(p, StableDiffusionProcessingImg2Img) else scripts.scripts_img2img)
    
    # Make sure p.scripts.scripts exists
    if not hasattr(p.scripts, 'scripts') or p.scripts.scripts is None:
        debug_print("Initializing empty scripts list")
        p.scripts.scripts = []
    
    # First try exact title match
    script = next((s for s in p.scripts.scripts if s.title() == script_title), None)
    
    # If not found and it's ControlNet, try alternative names
    if not script and script_title.lower() == "controlnet":
        # Try different casing and formats that might be used in WebUI Forge
        controlnet_alternatives = [
            "ControlNet", "controlnet", "Control Net", "control net", 
            "SD Forge ControlNet", "sd forge controlnet", "sd_forge_controlnet",
            "ControlNet Forge", "controlnet forge"
        ]
        
        # Try case-insensitive partial match
        for s in p.scripts.scripts:
            script_name = s.title().lower()
            if any(alt.lower() in script_name for alt in controlnet_alternatives):
                debug_print(f"Found ControlNet script with title: {s.title()}")
                script = s
                break
                
        # Try by module name if still not found
        if not script:
            for s in p.scripts.scripts:
                module_name = getattr(s, '__module__', '')
                if 'controlnet' in module_name.lower():
                    debug_print(f"Found ControlNet script through module name: {module_name}")
                    script = s
                    break
        
        # If we still can't find it, try to import from global scripts
        if not script:
            try:
                # Try to find ControlNet in global scripts
                debug_print("Checking global scripts list for ControlNet")
                from modules import scripts
                
                # Check in txt2img scripts
                if hasattr(scripts, 'scripts_txt2img') and scripts.scripts_txt2img:
                    for s in scripts.scripts_txt2img.scripts:
                        if s.title().lower() == "controlnet" or any(alt.lower() in s.title().lower() for alt in controlnet_alternatives):
                            debug_print(f"Found ControlNet in global txt2img scripts: {s.title()}")
                            # Add it to our current scripts
                            if s not in p.scripts.scripts:
                                p.scripts.scripts.append(s)
                            script = s
                            break
                
                # Check in img2img scripts if not found in txt2img
                if not script and hasattr(scripts, 'scripts_img2img') and scripts.scripts_img2img:
                    for s in scripts.scripts_img2img.scripts:
                        if s.title().lower() == "controlnet" or any(alt.lower() in s.title().lower() for alt in controlnet_alternatives):
                            debug_print(f"Found ControlNet in global img2img scripts: {s.title()}")
                            # Add it to our current scripts
                            if s not in p.scripts.scripts:
                                p.scripts.scripts.append(s)
                            script = s
                            break
                
                # Try to import the ControlNet module directly
                if not script:
                    try:
                        import importlib
                        from pathlib import Path
                        
                        # Try potential module paths
                        webui_path = Path(__file__).resolve().parent.parent.parent.parent.parent
                        controlnet_paths = [
                            webui_path / "extensions-builtin" / "sd_forge_controlnet" / "scripts" / "controlnet.py",
                            webui_path / "extensions-builtin" / "sd-webui-controlnet" / "scripts" / "controlnet.py",
                            webui_path / "extensions" / "sd_forge_controlnet" / "scripts" / "controlnet.py",
                            webui_path / "extensions" / "sd-webui-controlnet" / "scripts" / "controlnet.py"
                        ]
                        
                        for path in controlnet_paths:
                            if path.exists():
                                debug_print(f"Found ControlNet script at {path}")
                                try:
                                    # Try to load the module
                                    spec = importlib.util.spec_from_file_location("controlnet_module", path)
                                    if spec and spec.loader:
                                        module = importlib.util.module_from_spec(spec)
                                        spec.loader.exec_module(module)
                                        
                                        # Look for the script class
                                        for attr_name in dir(module):
                                            if "ControlNet" in attr_name and "Script" in attr_name:
                                                script_class = getattr(module, attr_name)
                                                # Create an instance
                                                script = script_class()
                                                # Add to scripts
                                                p.scripts.scripts.append(script)
                                                debug_print(f"Created ControlNet script instance from {path}")
                                                break
                                        
                                        if script:
                                            break
                                except Exception as e:
                                    debug_print(f"Error importing ControlNet module: {str(e)}")
                                    continue
                    except Exception as e:
                        debug_print(f"Error trying to import ControlNet directly: {str(e)}")
            except Exception as e:
                debug_print(f"Error searching global scripts: {str(e)}")
    
    if not script:
        available_scripts = [s.title() for s in p.scripts.scripts]
        debug_print(f"Available scripts: {available_scripts}")
        raise Exception(f"Script not found: {script_title}")
        
    return script

# Update the processing unit to include a specific script. 
# Manipulate the arg array and expected argument positions as required.
def add_forge_script_to_deforum_run(p: StableDiffusionProcessing, script_title : str, script_args : list):
    # Skip if script_args is None
    if script_args is None:
        debug_print(f"Skipping script: {script_title} as script_args is None")
        return

    # Initialize script_args_value if it doesn't exist
    if not hasattr(p, 'script_args_value') or p.script_args_value is None:
        p.script_args_value = []

    # shallow copy the script because we are changing its arg positions. Not copying breaks subsequent non-deforum runs.
    script = copy.copy(find_script(p, script_title))
    script.args_from = len(p.script_args_value)
    script.args_to = len(p.script_args_value) + len(script_args)
    
    # Initialize alwayson_scripts if it doesn't exist
    if not hasattr(p.scripts, 'alwayson_scripts') or p.scripts.alwayson_scripts is None:
        p.scripts.alwayson_scripts = []
        
    p.scripts.alwayson_scripts.append(script)
    p.script_args_value.extend(script_args)
    debug_print(f"Added script: {script.title()} with {len(script_args)} args, at positions {script.args_from}-{script.args_to} (of 0-{len(p.script_args_value)-1}.)")
