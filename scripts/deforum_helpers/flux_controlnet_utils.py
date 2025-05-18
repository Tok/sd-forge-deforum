"""
FLUX ControlNet Utilities for Deforum

This module handles registration and management of FLUX-specific ControlNet models,
not the Flux models themselves. It provides utilities for ensuring compatibility
between Deforum and the ControlNet extension when using Flux models.
"""

import os
import logging
from pathlib import Path
import sys
import importlib
import types

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DeforumFLUXControlNetUtils')

def debug_print(message):
    """Print debug messages for troubleshooting"""
    logger.info(f"[FLUX ControlNet Utils] {message}")

def is_controlnet_enabled():
    """Check if ControlNet is enabled in the system"""
    try:
        from modules import scripts
        
        # Check if ControlNet script is loaded and enabled
        for script_collection in [
            getattr(scripts, "scripts_txt2img", None), 
            getattr(scripts, "scripts_img2img", None)
        ]:
            if script_collection and hasattr(script_collection, "alwayson_scripts"):
                for script in script_collection.alwayson_scripts:
                    if hasattr(script, "title") and script.title().lower() == "controlnet":
                        debug_print("Found enabled ControlNet script")
                        return True
                        
        debug_print("ControlNet not enabled or not found")
        return False
    except Exception as e:
        debug_print(f"Error checking if ControlNet is enabled: {e}")
        return False

def register_flux_controlnet_models():
    """Find and register all FLUX ControlNet models in the system"""
    try:
        # Skip if ControlNet is not enabled
        if not is_controlnet_enabled():
            debug_print("ControlNet not enabled, skipping FLUX ControlNet model registration")
            return False
        
        from modules import shared
        
        # Find all FLUX ControlNet models in standard ControlNet paths
        flux_models = []
        for base_dir in [
            os.path.join(shared.models_path, "ControlNet"),
            os.path.join(shared.models_path, "controlnet"),
            os.path.join(shared.models_path, "xlabs", "controlnets")
        ]:
            if not os.path.exists(base_dir):
                continue
                
            # Find all FLUX models
            for file in os.listdir(base_dir):
                if 'flux' in file.lower() and file.endswith('.safetensors'):
                    model_path = os.path.join(base_dir, file)
                    flux_models.append((file, model_path))
        
        if not flux_models:
            debug_print("No FLUX ControlNet models found")
            return False
            
        debug_print(f"Found {len(flux_models)} FLUX ControlNet models")
        
        # Try to register with ControlNet global state
        try:
            # Try to directly access the global state first
            try:
                from lib_controlnet.global_state import controlnet_filename_dict, controlnet_models_list
                has_model_list = True
            except (ImportError, AttributeError):
                # Create dictionary if it doesn't exist
                debug_print("Could not import global_state directly, using fallback")
                try:
                    import sys
                    # Look for it in sys.modules
                    global_state_module = None
                    for name, mod in sys.modules.items():
                        if 'global_state' in name and hasattr(mod, 'controlnet_filename_dict'):
                            global_state_module = mod
                            debug_print(f"Found global_state module: {name}")
                            controlnet_filename_dict = global_state_module.controlnet_filename_dict
                            controlnet_models_list = getattr(global_state_module, 'controlnet_models_list', [])
                            has_model_list = hasattr(global_state_module, 'controlnet_models_list')
                            break
                    
                    if not global_state_module:
                        debug_print("Could not find global_state module, creating local dictionary")
                        controlnet_filename_dict = {}
                        controlnet_models_list = []
                        has_model_list = False
                except Exception as inner_e:
                    debug_print(f"Error in fallback: {inner_e}")
                    controlnet_filename_dict = {}
                    controlnet_models_list = []
                    has_model_list = False
            
            # Register each model
            for model_name, model_path in flux_models:
                controlnet_filename_dict[model_name] = model_path
                debug_print(f"Registered FLUX ControlNet model: {model_name}")
                
                # Also register without extension
                basename_no_ext = os.path.splitext(model_name)[0]
                controlnet_filename_dict[basename_no_ext] = model_path
                
                # Register in models list if it exists
                if has_model_list:
                    if model_name not in controlnet_models_list:
                        controlnet_models_list.append(model_name)
                    if basename_no_ext not in controlnet_models_list:
                        controlnet_models_list.append(basename_no_ext)
            
            # Make sure to initialize all ControlNet units
            ensure_controlnet_unit_attributes()
            
            return True
            
        except Exception as e:
            debug_print(f"Error registering with global_state: {e}")
            return False
            
    except Exception as e:
        debug_print(f"Error in register_flux_controlnet_models: {e}")
        import traceback
        traceback.print_exc()
        return False

def find_flux_controlnet_model(model_name, module_type=None):
    """Find a FLUX ControlNet model by name and/or module type"""
    try:
        # Skip if ControlNet is not enabled
        if not is_controlnet_enabled():
            debug_print("ControlNet not enabled, skipping FLUX ControlNet model lookup")
            return None

        from modules import shared
        
        # Check if the name already contains flux
        contains_flux = 'flux' in model_name.lower() if model_name else False
        
        # First, check for the exact model name
        if model_name:
            # Look in standard directories
            for base_dir in [
                os.path.join(shared.models_path, "ControlNet"),
                os.path.join(shared.models_path, "controlnet"),
                os.path.join(shared.models_path, "xlabs", "controlnets")
            ]:
                if not os.path.exists(base_dir):
                    continue
                    
                # Check exact match first
                exact_path = os.path.join(base_dir, model_name)
                if os.path.exists(exact_path):
                    debug_print(f"Found exact FLUX ControlNet model match: {exact_path}")
                    return exact_path
                    
                # Try with safetensors extension
                if not model_name.endswith('.safetensors'):
                    exact_path = os.path.join(base_dir, f"{model_name}.safetensors")
                    if os.path.exists(exact_path):
                        debug_print(f"Found FLUX ControlNet model with extension: {exact_path}")
                        return exact_path
        
        # If not found and we have a module type, try by module type
        if contains_flux or module_type:
            search_pattern = f"flux-{module_type}" if module_type else "flux"
            
            # Search for models by pattern
            for base_dir in [
                os.path.join(shared.models_path, "ControlNet"),
                os.path.join(shared.models_path, "controlnet"),
                os.path.join(shared.models_path, "xlabs", "controlnets")
            ]:
                if not os.path.exists(base_dir):
                    continue
                    
                # Find matching models
                matches = []
                for file in os.listdir(base_dir):
                    if search_pattern.lower() in file.lower() and file.endswith('.safetensors'):
                        matches.append(os.path.join(base_dir, file))
                
                if matches:
                    debug_print(f"Found {len(matches)} FLUX ControlNet models matching {search_pattern}")
                    # Return the first match
                    return matches[0]
        
        # Not found
        debug_print(f"Could not find FLUX ControlNet model for {model_name} with module {module_type}")
        return None
        
    except Exception as e:
        debug_print(f"Error in find_flux_controlnet_model: {e}")
        return None

def ensure_controlnet_unit_attributes():
    """
    Ensures all ControlNet units are properly initialized in the global namespace
    """
    try:
        # Skip if ControlNet is not enabled
        if not is_controlnet_enabled():
            debug_print("ControlNet not enabled, skipping unit attribute initialization")
            return False
            
        debug_print("Ensuring ControlNet unit attributes are properly set...")
        
        # First try to find and import deforum_controlnet
        deforum_controlnet = None
        
        # Look in sys.modules
        for name, module in sys.modules.items():
            if 'deforum_controlnet' in name:
                deforum_controlnet = module
                debug_print(f"Found deforum_controlnet module at {name}")
                break
        
        # If not found in sys.modules, try to import it
        if not deforum_controlnet:
            try:
                # First try direct import
                from deforum_helpers import deforum_controlnet
                debug_print("Imported deforum_controlnet module directly")
            except ImportError:
                # Then try to find it by path
                try:
                    script_dir = Path(__file__).resolve().parent
                    deforum_controlnet_path = script_dir / "deforum_controlnet.py"
                    
                    if deforum_controlnet_path.exists():
                        spec = importlib.util.spec_from_file_location("deforum_controlnet", deforum_controlnet_path)
                        deforum_controlnet = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(deforum_controlnet)
                        debug_print(f"Imported deforum_controlnet from {deforum_controlnet_path}")
                    else:
                        debug_print(f"Could not find deforum_controlnet.py at {deforum_controlnet_path}")
                        return False
                except Exception as import_e:
                    debug_print(f"Error importing deforum_controlnet: {import_e}")
                    return False
        
        # Now try to use setup_controlnet_units from deforum_controlnet
        if deforum_controlnet and hasattr(deforum_controlnet, 'setup_controlnet_units'):
            # Monkey patch the num_of_models function to increase max supported models
            if hasattr(deforum_controlnet, 'num_of_models'):
                original_num_of_models = deforum_controlnet.num_of_models
                
                def patched_num_of_models():
                    """Return increased maximum number of ControlNet models supported"""
                    # If original returns a constant like MAX_CONTROLNET_MODELS
                    original_value = original_num_of_models()
                    
                    # Try to get from shared options too as a fallback
                    from modules import shared
                    opts_value = getattr(shared.opts, 'control_net_unit_count', 
                                        getattr(shared.opts, 'control_net_max_models_num', 5))
                    
                    # Return the maximum to ensure compatibility
                    return max(original_value, opts_value, 5)
                
                # Apply the patch
                deforum_controlnet.num_of_models = patched_num_of_models
                debug_print("Patched num_of_models to increase max supported models")
            
            # Call setup_controlnet_units with a reasonable count
            try:
                from modules import shared
                unit_count = getattr(shared.opts, 'control_net_unit_count', 
                                    getattr(shared.opts, 'control_net_max_models_num', 5))
                
                deforum_controlnet.setup_controlnet_units(count=unit_count)
                debug_print(f"Initialized {unit_count} ControlNet units")
                return True
            except Exception as e:
                debug_print(f"Error setting up ControlNet units: {e}")
                return False
        else:
            debug_print("Could not find setup_controlnet_units function")
            return False
            
    except Exception as e:
        debug_print(f"Error in ensure_controlnet_unit_attributes: {e}")
        return False

def ensure_controlnet_unit_namespace(controlnet_args):
    """
    Ensure that all required ControlNet keys exist in the namespace
    This is important for persistence and ControlNetKeys compatibility
    """
    try:
        if not controlnet_args:
            debug_print("No controlnet_args provided")
            return False
            
        # Try to import the module
        deforum_controlnet = None
        try:
            from deforum_helpers import deforum_controlnet
        except ImportError:
            # Look in sys.modules
            for name, module in sys.modules.items():
                if 'deforum_controlnet' in name:
                    deforum_controlnet = module
                    break
        
        if not deforum_controlnet:
            debug_print("Could not find deforum_controlnet module")
            return False
            
        # Call the function if it exists
        if hasattr(deforum_controlnet, 'ensure_controlnet_keys_in_namespace'):
            result = deforum_controlnet.ensure_controlnet_keys_in_namespace(controlnet_args)
            debug_print(f"ensure_controlnet_keys_in_namespace result: {result}")
            return result
        else:
            debug_print("Could not find ensure_controlnet_keys_in_namespace function")
            return False
            
    except Exception as e:
        debug_print(f"Error in ensure_controlnet_unit_namespace: {e}")
        return False

def create_safe_hwc3():
    """Create a safe version of the HWC3 function that doesn't crash with assertion errors"""
    
    def safe_HWC3(x):
        """
        Safe replacement for the HWC3 function that won't crash with assertion errors
        
        This handles cases where the input image is not uint8, None, or has the wrong dimensions
        """
        import numpy as np
        
        try:
            # Handle None input
            if x is None:
                debug_print("HWC3 received None input, returning blank image")
                return np.zeros((64, 64, 3), dtype=np.uint8)
            
            # Handle non-array inputs
            if not isinstance(x, np.ndarray):
                try:
                    x = np.array(x)
                except:
                    debug_print("HWC3 received non-convertible input, returning blank image")
                    return np.zeros((64, 64, 3), dtype=np.uint8)
            
            # Convert to uint8 if needed, but don't assert
            if x.dtype != np.uint8:
                if np.issubdtype(x.dtype, np.floating):
                    if x.max() <= 1.0:
                        x = (x * 255).astype(np.uint8)
                    else:
                        x = x.astype(np.uint8)
                else:
                    x = x.astype(np.uint8)
            
            # Handle different dimensions
            if len(x.shape) == 2:
                # Grayscale to RGB
                x = np.stack([x, x, x], axis=2)
            elif len(x.shape) == 3:
                if x.shape[2] == 1:
                    # Single channel to RGB
                    x = np.concatenate([x, x, x], axis=2)
                elif x.shape[2] == 4:
                    # RGBA to RGB (drop alpha)
                    x = x[:, :, :3]
                elif x.shape[2] != 3:
                    # Something else, create blank image of same height/width
                    return np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
            elif len(x.shape) == 4:
                # Batched image, take first one
                x = x[0]
                return safe_HWC3(x)  # Recursive call with 3D input
            elif len(x.shape) != 3:
                # Something entirely wrong
                return np.zeros((512, 512, 3), dtype=np.uint8)
            
            return x
            
        except Exception as e:
            debug_print(f"Error in safe_HWC3: {e}")
            return np.zeros((512, 512, 3), dtype=np.uint8)
    
    return safe_HWC3

def apply_flux_controlnet_patches():
    """Apply all FLUX ControlNet model patches"""
    try:
        # Skip if ControlNet is not enabled
        if not is_controlnet_enabled():
            debug_print("ControlNet not enabled, skipping FLUX ControlNet patches")
            return False
            
        debug_print("Applying FLUX ControlNet model patches...")
        
        # Register FLUX ControlNet models
        register_success = register_flux_controlnet_models()
        
        # Initialize ControlNet unit attributes
        init_success = ensure_controlnet_unit_attributes()
        
        # Patch HWC3 function in modules_forge.utils
        hwc3_patched = False
        try:
            import sys
            
            # Look for modules_forge.utils
            for name, module in list(sys.modules.items()):
                if 'modules_forge.utils' in name and hasattr(module, 'HWC3'):
                    # Create our safe HWC3
                    safe_HWC3 = create_safe_hwc3()
                    
                    # Save original
                    if not hasattr(module, '_original_HWC3'):
                        module._original_HWC3 = module.HWC3
                    
                    # Replace with our version
                    module.HWC3 = safe_HWC3
                    debug_print(f"Patched HWC3 in {name}")
                    
                    hwc3_patched = True
                    break
        except Exception as e:
            debug_print(f"Error patching HWC3: {e}")
        
        debug_print(f"FLUX ControlNet patches applied - models registered: {register_success}, units initialized: {init_success}, HWC3 patched: {hwc3_patched}")
        return register_success or init_success or hwc3_patched
        
    except Exception as e:
        debug_print(f"Error applying FLUX ControlNet patches: {e}")
        return False

# Register models on module import
register_flux_controlnet_models() 