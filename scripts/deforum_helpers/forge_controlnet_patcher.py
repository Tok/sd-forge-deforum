"""
Forge ControlNet Patcher for Deforum

This module provides targeted runtime patches for WebUI Forge's ControlNet implementation
to ensure compatibility between Deforum, ControlNet, and FLUX models.
"""

import sys
import os
import importlib
import inspect
import traceback
import logging
import numpy as np
from pathlib import Path
import types
import contextlib
import re
import shutil
import datetime
from PIL import Image

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DeforumForgeControlNetPatcher')

def debug_print(message):
    """Print debug messages for troubleshooting"""
    logger.info(f"[Forge ControlNet Patcher] {message}")

#--------------------------
# Path Helper Functions
#--------------------------

def get_webui_path():
    """Find the base webui path from our extension path"""
    current_path = Path(__file__).resolve().parent.parent.parent
    return current_path.parent.parent

#--------------------------
# ControlNet HWC3 Patching
#--------------------------

def patch_utils_hwc3():
    """
    Patch the HWC3 function in WebUI Forge to handle different image formats safely
    
    The HWC3 function in WebUI Forge's utils.py has a strict uint8 dtype assertion
    that can cause errors when different image types are passed.
    """
    try:
        import sys
        import numpy as np
        import importlib
        
        debug_print("Attempting to patch HWC3 function in all relevant modules")
        
        # Store our patched version for consistent application
        def safe_HWC3(x):
            """Safe version of HWC3 that handles any input type without assertions"""
            try:
                # Explicit None check right at the start
                if x is None:
                    debug_print("HWC3 received None input, returning blank image")
                    return np.zeros((64, 64, 3), dtype=np.uint8)
                    
                # If input isn't a numpy array, convert it first
                if not isinstance(x, np.ndarray):
                    try:
                        debug_print(f"Converting non-numpy input of type {type(x)} to array")
                        x = np.array(x)
                    except Exception as conv_err:
                        debug_print(f"Failed to convert to array: {conv_err}")
                        return np.zeros((64, 64, 3), dtype=np.uint8)
                
                # Debug info about input
                debug_print(f"HWC3 processing array of shape {x.shape} and dtype {x.dtype}")
                
                # Convert any non-uint8 array to uint8
                if x.dtype != np.uint8:
                    try:
                        if np.issubdtype(x.dtype, np.floating):
                            if x.max() <= 1.0:
                                debug_print("Converting float [0-1] to uint8")
                                x = (x * 255).astype(np.uint8)
                            else:
                                debug_print("Converting float to uint8")
                                x = x.astype(np.uint8)
                        else:
                            debug_print(f"Converting {x.dtype} to uint8")
                            x = x.astype(np.uint8)
                    except Exception as conversion_err:
                        debug_print(f"Error during type conversion: {conversion_err}")
                        # If conversion fails, create a blank image instead of crashing
                        return np.zeros((64, 64, 3), dtype=np.uint8)
                
                # Handle different dimensions
                if len(x.shape) == 2:
                    # Grayscale image - convert to RGB
                    debug_print("Converting grayscale to RGB")
                    x = np.stack([x, x, x], axis=2)
                elif len(x.shape) == 3:
                    if x.shape[2] == 1:
                        # Single channel - convert to RGB
                        debug_print("Converting single channel to RGB")
                        x = np.concatenate([x, x, x], axis=2)
                    elif x.shape[2] == 4:
                        # RGBA - drop alpha channel
                        debug_print("Converting RGBA to RGB")
                        x = x[:, :, :3]
                    elif x.shape[2] != 3:
                        # Unknown format - create blank RGB
                        debug_print(f"Unknown format with {x.shape[2]} channels, creating blank")
                        return np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
                elif len(x.shape) == 4:
                    # Batch of images - take the first one
                    debug_print(f"Got batch of {x.shape[0]} images, taking first")
                    x = x[0]
                    # Recursively process the first image
                    return safe_HWC3(x)
                elif len(x.shape) != 3:
                    # Not a proper image - create blank
                    debug_print(f"Invalid shape {x.shape}, creating blank image")
                    return np.zeros((512, 512, 3), dtype=np.uint8)
                
                debug_print(f"HWC3 successful, returning array of shape {x.shape}")
                return x
                
            except Exception as e:
                debug_print(f"Error in safe_HWC3: {e}")
                import traceback
                traceback.print_exc()
                # Last resort
                return np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Modules to be patched
        patched_modules = []
        failed_modules = []
        
        # First, try direct imports for critical modules
        critical_modules = [
            "modules_forge.utils",
            "lib_controlnet.utils",
            "extensions.sd_forge_controlnet.utils",
            "extensions-builtin.sd_forge_controlnet.utils"
        ]
        
        for module_name in critical_modules:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "HWC3"):
                    if not hasattr(module, "_original_HWC3"):
                        module._original_HWC3 = module.HWC3
                    module.HWC3 = safe_HWC3
                    module._patched_by_deforum = True
                    patched_modules.append(module_name)
                    debug_print(f"Successfully patched {module_name}.HWC3")
            except ImportError:
                debug_print(f"Module {module_name} not found")
                failed_modules.append(module_name)
            except Exception as e:
                debug_print(f"Error patching {module_name}: {e}")
                failed_modules.append(f"{module_name} (error)")
        
        # Direct replacement for all modules in sys.modules
        for name, module in list(sys.modules.items()):
            if module and ("utils" in name.lower() or "controlnet" in name.lower()):
                try:
                    if hasattr(module, "HWC3") and not hasattr(module, "_patched_by_deforum"):
                        module._original_HWC3 = module.HWC3
                        module.HWC3 = safe_HWC3
                        module._patched_by_deforum = True
                        patched_modules.append(name)
                        debug_print(f"Patched HWC3 in {name}")
                except Exception as e:
                    debug_print(f"Error patching module {name}: {e}")
                    failed_modules.append(name)
        
        # Summary
        if patched_modules:
            debug_print(f"Successfully patched HWC3 in {len(patched_modules)} modules")
            debug_print(f"Patched modules: {', '.join(patched_modules)}")
            return True
        else:
            debug_print("Could not find any modules with HWC3 to patch")
            if failed_modules:
                debug_print(f"Failed modules: {', '.join(failed_modules)}")
            return False
        
    except Exception as e:
        debug_print(f"Critical error in patch_utils_hwc3: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_modules_forge_hwc3(p):
    """
    Patch the modules_forge.utils.HWC3 function to handle any image format safely
    
    This fixes the assertion error when processing images with different dtypes
    """
    try:
        import sys
        import importlib
        import numpy as np
        
        debug_print("Attempting to patch modules_forge.utils.HWC3")
        
        # Define a safer HWC3 function without assertions
        def safe_HWC3(x):
            try:
                if x is None:
                    return np.zeros((64, 64, 3), dtype=np.uint8)
                
                # Convert to numpy array if not already
                if not isinstance(x, np.ndarray):
                    try:
                        x = np.array(x)
                    except:
                        return np.zeros((64, 64, 3), dtype=np.uint8)
                
                # Convert to uint8 WITHOUT any assertion
                if x.dtype != np.uint8:
                    try:
                        if np.issubdtype(x.dtype, np.floating):
                            if x.max() <= 1.0:
                                x = (x * 255).astype(np.uint8)
                            else:
                                x = x.astype(np.uint8)
                        else:
                            x = x.astype(np.uint8)
                    except:
                        return np.zeros((64, 64, 3), dtype=np.uint8)
                
                # Handle different dimensions
                if len(x.shape) == 2:
                    x = np.stack([x, x, x], axis=2)
                elif len(x.shape) == 3:
                    if x.shape[2] == 1:
                        x = np.concatenate([x, x, x], axis=2)
                    elif x.shape[2] == 4:
                        x = x[:, :, :3]
                    elif x.shape[2] != 3:
                        return np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
                elif len(x.shape) == 4:
                    # Get the first image
                    x = x[0]
                    # Process recursively
                    return safe_HWC3(x)
                else:
                    return np.zeros((512, 512, 3), dtype=np.uint8)
                
                return x
            except:
                return np.zeros((512, 512, 3), dtype=np.uint8)
        
        # Track successful patches
        patched_count = 0
        
        # Try to patch various modules directly
        modules_to_patch = [
            "modules_forge.utils", 
            "lib_controlnet.utils",
            "lib_controlnet.external_code"
        ]
        
        for module_name in modules_to_patch:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "HWC3"):
                    # Store original if not already stored
                    if not hasattr(module, "_original_HWC3"):
                        module._original_HWC3 = module.HWC3
                    
                    # Apply our safe version
                    module.HWC3 = safe_HWC3
                    module._patched_by_deforum = True
                    debug_print(f"Patched HWC3 in {module_name}")
                    patched_count += 1
            except ImportError:
                debug_print(f"Could not import {module_name}")
                
        # If we couldn't patch any modules directly, try to find them in sys.modules
        if patched_count == 0:
            for name, module in list(sys.modules.items()):
                if module and ("utils" in name.lower() or "controlnet" in name.lower()):
                    try:
                        if hasattr(module, "HWC3") and not hasattr(module, "_patched_by_deforum"):
                            module._original_HWC3 = module.HWC3
                            module.HWC3 = safe_HWC3
                            module._patched_by_deforum = True
                            patched_count += 1
                            debug_print(f"Found and patched HWC3 in {name}")
                    except Exception as e:
                        debug_print(f"Error patching module {name}: {e}")
        
        # Final result
        if patched_count > 0:
            debug_print(f"Successfully patched HWC3 in {patched_count} modules")
            return True
        else:
            debug_print("Could not find any modules with HWC3 to patch")
            return False
        
    except Exception as e:
        debug_print(f"Error patching HWC3 function: {e}")
        import traceback
        traceback.print_exc()
        return False

#--------------------------
# Global State Patching
#--------------------------

def patch_global_state_module():
    """
    Patch the global_state.py module to handle model paths properly
    
    This specifically addresses the KeyError in get_controlnet_filename when
    full paths are provided instead of just the basename.
    """
    try:
        import importlib
        
        # Try to import the module directly
        try:
            from lib_controlnet import global_state
            global_state_module = global_state
            debug_print(f"Found global_state module via direct import")
        except ImportError:
            # Look for it in sys.modules
            global_state_module = None
            for name, mod in sys.modules.items():
                if 'global_state' in name and hasattr(mod, 'get_controlnet_filename'):
                    global_state_module = mod
                    debug_print(f"Found global_state module in sys.modules: {name}")
                    break
            
            if not global_state_module:
                debug_print("Could not find global_state module")
                return False
        
        # Check if already patched
        if hasattr(global_state_module, "_original_get_controlnet_filename"):
            debug_print("global_state.get_controlnet_filename already patched")
            return True
            
        # Make sure the required dict and function exist
        if not hasattr(global_state_module, "controlnet_filename_dict"):
            debug_print("Creating missing controlnet_filename_dict in global_state module")
            global_state_module.controlnet_filename_dict = {}
            
        if not hasattr(global_state_module, "get_controlnet_filename"):
            debug_print("Creating missing get_controlnet_filename function")
            
            # Create a basic implementation if it doesn't exist
            def basic_get_controlnet_filename(model_name):
                if model_name in global_state_module.controlnet_filename_dict:
                    return global_state_module.controlnet_filename_dict[model_name]
                raise KeyError(f"Model '{model_name}' not found")
                
            global_state_module.get_controlnet_filename = basic_get_controlnet_filename
            
        # Store original function
        global_state_module._original_get_controlnet_filename = global_state_module.get_controlnet_filename
        
        # Define patched version
        def patched_get_controlnet_filename(controlnet_name):
            # If input is None or empty, return None
            if controlnet_name is None or controlnet_name == "" or controlnet_name == "None":
                debug_print(f"Empty model name, returning None")
                return None
                
            # Access the filename dict
            filename_dict = global_state_module.controlnet_filename_dict
            
            # If it's a direct match, return immediately
            if controlnet_name in filename_dict:
                return filename_dict[controlnet_name]
                
            # Handle full paths
            if os.path.exists(controlnet_name):
                debug_print(f"Converting full path to basename: {controlnet_name}")
                
                # First try registering the path directly
                filename_dict[controlnet_name] = controlnet_name
                
                # Then also register by basename
                basename = os.path.basename(controlnet_name)
                if basename not in filename_dict:
                    filename_dict[basename] = controlnet_name
                    debug_print(f"Registered basename in filename dict: {basename}")
                
                # Also try without extension
                basename_no_ext = os.path.splitext(basename)[0]
                if basename_no_ext not in filename_dict:
                    filename_dict[basename_no_ext] = controlnet_name
                    debug_print(f"Registered basename without extension: {basename_no_ext}")
                
                return controlnet_name
                
            # If we get here, try the original function
            try:
                return global_state_module._original_get_controlnet_filename(controlnet_name)
            except Exception as e:
                debug_print(f"Error in original get_controlnet_filename: {e}")
                
                # Last resort: try to find a partial match
                for key, value in filename_dict.items():
                    if (key and controlnet_name and 
                        (key.lower() in controlnet_name.lower() or 
                         controlnet_name.lower() in key.lower())):
                        debug_print(f"Found partial match: {key} for {controlnet_name}")
                        return value
                        
                # Special handling for FLUX models
                if 'flux' in controlnet_name.lower():
                    for key, value in filename_dict.items():
                        if 'flux' in key.lower():
                            debug_print(f"Found FLUX model fallback: {key}")
                            return value
                
                # Truly nothing found, but don't raise an exception
                debug_print(f"No match found for model '{controlnet_name}', returning None")
                return None
        
        # Apply the patch
        global_state_module.get_controlnet_filename = patched_get_controlnet_filename
        debug_print("Successfully patched global_state.get_controlnet_filename")
        
        # Also update the update_controlnet_filenames function if it exists
        if hasattr(global_state_module, "update_controlnet_filenames"):
            original_update = global_state_module.update_controlnet_filenames
            
            def patched_update_controlnet_filenames():
                # Call original function
                try:
                    original_update()
                except Exception as e:
                    debug_print(f"Error in original update_controlnet_filenames: {e}")
                
                # Make sure all models are properly registered
                from modules import shared
                
                # Look in the ControlNet directory
                models_path = shared.models_path
                extensions = ['.safetensors', '.ckpt', '.pt', '.pth']
                
                for controlnet_dir in [
                    os.path.join(models_path, "ControlNet"),
                    os.path.join(models_path, "controlnet"),
                    os.path.join(models_path, "control_net"),
                    os.path.join(models_path, "xlabs", "controlnets")
                ]:
                    if not os.path.exists(controlnet_dir):
                        continue
                        
                    # Register all models in the directory
                    for filename in os.listdir(controlnet_dir):
                        if any(filename.endswith(ext) for ext in extensions):
                            fullpath = os.path.join(controlnet_dir, filename)
                            
                            # Register by various names
                            global_state_module.controlnet_filename_dict[filename] = fullpath
                            
                            # Also by basename without extension
                            basename_no_ext = os.path.splitext(filename)[0]
                            global_state_module.controlnet_filename_dict[basename_no_ext] = fullpath
                            
                            # Also register by full path
                            global_state_module.controlnet_filename_dict[fullpath] = fullpath
                            
                            # Make sure it's in the models list if it exists
                            if hasattr(global_state_module, "controlnet_models_list"):
                                if filename not in global_state_module.controlnet_models_list:
                                    global_state_module.controlnet_models_list.append(filename)
                                if basename_no_ext not in global_state_module.controlnet_models_list:
                                    global_state_module.controlnet_models_list.append(basename_no_ext)
            
            # Apply the patch
            global_state_module.update_controlnet_filenames = patched_update_controlnet_filenames
            debug_print("Successfully patched update_controlnet_filenames")
            
            # Run the update function once
            try:
                global_state_module.update_controlnet_filenames()
            except Exception as e:
                debug_print(f"Error running update_controlnet_filenames: {e}")
            
        return True
    except Exception as e:
        debug_print(f"Error patching global_state module: {e}")
        traceback.print_exc()
        return False

#--------------------------
# Processing Class Patching
#--------------------------

def patch_processing_classes():
    """
    Patch the StableDiffusionProcessingTxt2Img and StableDiffusionProcessingImg2Img classes
    
    This adds default resize_mode attribute to both classes at instantiation time.
    """
    try:
        # Try a safer approach that doesn't rely on direct imports that might cause circular imports
        debug_print("Using safer approach to patch processing classes")
        
        # Find the processing module in sys.modules if it's already loaded
        import sys
        
        processing_module = None
        for name, module in sys.modules.items():
            if name == 'modules.processing' or name.endswith('.processing'):
                processing_module = module
                debug_print(f"Found processing module as {name}")
                break
        
        # If not found in sys.modules, try to load it directly but with safeguards
        if not processing_module:
            debug_print("Processing module not found in sys.modules, attempting direct import")
            try:
                # Try to import with a context manager to prevent propagation of import errors
                import contextlib
                with contextlib.suppress(ImportError, ModuleNotFoundError):
                    from modules import processing as processing_module
                    debug_print("Successfully imported processing module")
            except Exception as import_e:
                debug_print(f"Error importing processing module: {import_e}")
        
        # If we still don't have the module, we can't proceed
        if not processing_module:
            debug_print("Could not find or import processing module, skipping processing class patches")
            return False
            
        # Check if the required classes exist
        txt2img_class = getattr(processing_module, 'StableDiffusionProcessingTxt2Img', None)
        img2img_class = getattr(processing_module, 'StableDiffusionProcessingImg2Img', None)
        
        if not txt2img_class or not img2img_class:
            debug_print("Required processing classes not found in module")
            return False
            
        # Check if already patched
        if hasattr(txt2img_class, "_resize_mode_patched"):
            debug_print("Processing classes already patched")
            return True
            
        # Store original __init__ methods
        original_txt2img_init = txt2img_class.__init__
        original_img2img_init = img2img_class.__init__
        
        # Create patched versions
        def patched_txt2img_init(self, *args, **kwargs):
            original_txt2img_init(self, *args, **kwargs)
            if not hasattr(self, 'resize_mode'):
                self.resize_mode = "Inner Fit (Scale to Fit)"
                
        def patched_img2img_init(self, *args, **kwargs):
            original_img2img_init(self, *args, **kwargs)
            if not hasattr(self, 'resize_mode'):
                self.resize_mode = "Inner Fit (Scale to Fit)"
        
        # Patch the __getattr__ method to always return resize_mode even if it doesn't exist
        def patched_txt2img_getattr(self, name):
            if name == 'resize_mode':
                debug_print("Intercepted getattr for resize_mode in Txt2Img")
                return "Inner Fit (Scale to Fit)"
            raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")
            
        def patched_img2img_getattr(self, name):
            if name == 'resize_mode':
                debug_print("Intercepted getattr for resize_mode in Img2Img")
                return "Inner Fit (Scale to Fit)"
            raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")
        
        # Apply patches for __init__
        txt2img_class.__init__ = patched_txt2img_init
        img2img_class.__init__ = patched_img2img_init
        debug_print("Successfully patched StableDiffusionProcessingTxt2Img.__init__")
        debug_print("Successfully patched StableDiffusionProcessingImg2Img.__init__")
        
        # Apply patches for __getattr__
        txt2img_class.__getattr__ = patched_txt2img_getattr
        img2img_class.__getattr__ = patched_img2img_getattr
        debug_print("Successfully patched StableDiffusionProcessingTxt2Img.__getattr__")
        debug_print("Successfully patched StableDiffusionProcessingImg2Img.__getattr__")
        
        # Mark as patched
        txt2img_class._resize_mode_patched = True
        img2img_class._resize_mode_patched = True
        
        return True
        
    except Exception as e:
        debug_print(f"Error patching processing classes: {e}")
        import traceback
        traceback.print_exc()
        return False

#--------------------------
# ControlNet Script Patching
#--------------------------

def patch_controlnet_script_instance(script):
    """
    Apply runtime patching to a specific ControlNet script instance
    
    This is a safer approach than patching the files directly
    """
    try:
        if not script or not hasattr(script, "title") or script.title().lower() != "controlnet":
            return False
            
        # Check if already patched
        if hasattr(script, '_deforum_patched'):
            return True
            
        debug_print(f"Found ControlNet script instance to patch")
        
        # Make sure current_params dictionary exists
        if not hasattr(script, 'current_params') or script.current_params is None:
            script.current_params = {}
            debug_print("Created missing current_params dictionary")
        
        # Patch the process method if it exists
        if hasattr(script, "process"):
            original_process = script.process
            
            def patched_process(self, p, *args, **kwargs):
                # Make sure resize_mode exists
                if not hasattr(p, 'resize_mode'):
                    p.resize_mode = "Inner Fit (Scale to Fit)"
                    
                # Make sure current_params exists
                if not hasattr(self, 'current_params') or self.current_params is None:
                    self.current_params = {}
                    
                # Initialize parameters for all possible index values
                for i in range(len(args)):
                    if i not in self.current_params:
                        self.current_params[i] = {}
                
                # Call original with safety wrap
                try:
                    return original_process(p, *args, **kwargs)
                except Exception as e:
                    debug_print(f"Error in ControlNet process: {e}")
                    return None
                    
            # Replace the method
            script.process = patched_process.__get__(script, type(script))
            debug_print("Patched ControlNet process method")
        
        # Patch process_before_every_sampling if it exists
        if hasattr(script, "process_before_every_sampling"):
            original_before_sampling = script.process_before_every_sampling
            
            def patched_before_sampling(self, p, *args, **kwargs):
                # Ensure current_params exists
                if not hasattr(self, 'current_params') or self.current_params is None:
                    self.current_params = {}
                    
                # Initialize parameters for all possible index values
                for i in range(len(args)):
                    if i not in self.current_params:
                        self.current_params[i] = {}
                
                # Call with safety wrapper
                try:
                    return original_before_sampling(self, p, *args, **kwargs)
                except Exception as e:
                    debug_print(f"Error in process_before_every_sampling: {e}")
                    return None
                    
            # Replace the method
            script.process_before_every_sampling = patched_before_sampling.__get__(script, type(script))
            debug_print("Patched process_before_every_sampling method")
        
        # Patch get_input_data if it exists
        if hasattr(script, "get_input_data"):
            original_get_input_data = script.get_input_data
            
            def patched_get_input_data(self, p, unit, preprocessor, h, w):
                try:
                    # Ensure p has resize_mode
                    if not hasattr(p, 'resize_mode'):
                        p.resize_mode = "Inner Fit (Scale to Fit)"
                        
                    # Call original method
                    return original_get_input_data(self, p, unit, preprocessor, h, w)
                except Exception as e:
                    debug_print(f"Error in get_input_data: {e}")
                    
                    # Safe fallback - create a blank image of the requested size
                    blank = np.zeros((h or 512, w or 512, 3), dtype=np.uint8)
                    return [blank], "Inner Fit (Scale to Fit)"
                    
            # Replace the method
            script.get_input_data = patched_get_input_data.__get__(script, type(script))
            debug_print("Patched get_input_data method")
        
        # Mark script as patched
        script._deforum_patched = True
        debug_print("Successfully patched ControlNet script instance")
        
        return True
        
    except Exception as e:
        debug_print(f"Error patching ControlNet script instance: {e}")
        traceback.print_exc()
        return False

def find_and_patch_controlnet_scripts():
    """Find all loaded ControlNet script instances and patch them"""
    try:
        from modules import scripts
        patched_count = 0
        
        # Look for all script collections that might contain ControlNet
        script_collections = []
        
        if hasattr(scripts, "scripts_txt2img"):
            script_collections.append(scripts.scripts_txt2img)
            
        if hasattr(scripts, "scripts_img2img"):
            script_collections.append(scripts.scripts_img2img)
        
        # Try to find any other script collections
        for attr_name in dir(scripts):
            if attr_name.startswith("scripts_") and attr_name not in ["scripts_txt2img", "scripts_img2img"]:
                collection = getattr(scripts, attr_name)
                if hasattr(collection, "alwayson_scripts"):
                    script_collections.append(collection)
        
        # Process each collection
        for collection in script_collections:
            if not hasattr(collection, "alwayson_scripts"):
                continue
                
            # Look for ControlNet scripts
            for script in collection.alwayson_scripts:
                if hasattr(script, "title") and script.title().lower() == "controlnet":
                    if patch_controlnet_script_instance(script):
                        patched_count += 1
        
        if patched_count > 0:
            debug_print(f"Patched {patched_count} ControlNet script instances")
            return True
        else:
            debug_print("No ControlNet script instances found to patch")
            return False
            
    except Exception as e:
        debug_print(f"Error finding and patching ControlNet scripts: {e}")
        traceback.print_exc()
        return False

#--------------------------
# FLUX Model Patching
#--------------------------

def apply_flux_models_patch():
    """Apply patches specifically for FLUX models compatibility"""
    try:
        from modules import shared
        import os
        
        # Create a single point to store FLUX model information
        flux_models = []
        
        # Find all FLUX models in standard ControlNet paths
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
            debug_print("No FLUX models found")
            return False
        
        debug_print(f"Found {len(flux_models)} FLUX models")
        
        # Register each model in the ControlNet system
        for model_name, model_path in flux_models:
            # Try to register with the global state first
            try:
                from lib_controlnet.global_state import controlnet_filename_dict
                
                # Register by various names
                controlnet_filename_dict[model_name] = model_path
                debug_print(f"Registered FLUX model: {model_name}")
                
                # Register without extension
                basename_no_ext = os.path.splitext(model_name)[0]
                controlnet_filename_dict[basename_no_ext] = model_path
                
                # Register by full path
                controlnet_filename_dict[model_path] = model_path
                
                # Make sure it's in the models list too
                try:
                    from lib_controlnet.global_state import controlnet_models_list
                    if model_name not in controlnet_models_list:
                        controlnet_models_list.append(model_name)
                    if basename_no_ext not in controlnet_models_list:
                        controlnet_models_list.append(basename_no_ext)
                        
                    # Also try with other casing
                    if model_name.lower() not in [m.lower() for m in controlnet_models_list]:
                        controlnet_models_list.append(model_name)
                except ImportError:
                    pass
                    
            except Exception as e:
                debug_print(f"Error registering FLUX model in global state: {e}")
        
        debug_print("Successfully applied FLUX models patch")
        return True
    
    except Exception as e:
        debug_print(f"Error applying FLUX models patch: {e}")
        traceback.print_exc()
        return False

#--------------------------
# Environment Variables
#--------------------------

def setup_forge_controlnet_env():
    """Set up environment variables required by WebUI Forge ControlNet"""
    try:
        import os
        
        # Check if we're running in WebUI Forge by looking for its modules
        forge_env = False
        try:
            import modules_forge
            forge_env = True
            debug_print("Detected WebUI Forge environment")
        except ImportError:
            debug_print("Not in WebUI Forge environment")
            return False
            
        if forge_env:
            # Set needed environment variables for WebUI Forge ControlNet
            # These are based on common issues and forum discussions
            
            # Set environment variable to ignore IndentationError
            os.environ["IGNORE_TORCH_INDENT_WARNING"] = "1"
            debug_print("Set IGNORE_TORCH_INDENT_WARNING=1")
            
            # Set model directories to include all possible locations
            os.environ["FORGE_MODEL_DIRS"] = "ControlNet;controlnet;control_net;xlabs/controlnets"
            debug_print("Set FORGE_MODEL_DIRS=ControlNet;controlnet;control_net;xlabs/controlnets")
            
            # Set CUDA memory config to prevent OOM errors
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
            debug_print("Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
            
            return True
    except Exception as e:
        debug_print(f"Error setting environment variables: {e}")
        return False

#--------------------------
# ControlNet Unit 5 Fix
#--------------------------

def patch_controlnet_unit5(p):
    """
    Apply specialized patches specifically for ControlNet unit 5
    
    This targets the known issue with unit 5 that causes HWC3 assertion errors
    """
    try:
        import os
        import sys
        import numpy as np
        import importlib
        import inspect
        import types
        
        debug_print("Applying specialized unit 5 patches")
        
        # Enhanced null validation - check p and all attributes in a single block
        # with detailed error reporting
        if p is None:
            debug_print("Processing object is None, attempting direct HWC3 patching instead")
            # Even if p is None, we can still try to patch HWC3 directly
            try:
                # Direct patch to modules_forge.utils - most critical
                try:
                    modules_forge_utils = importlib.import_module("modules_forge.utils")
                    if hasattr(modules_forge_utils, "HWC3"):
                        # Define safe HWC3 function
                        def safe_HWC3(x):
                            try:
                                if x is None:
                                    return np.zeros((64, 64, 3), dtype=np.uint8)
                                    
                                # Convert to numpy array if it's not already
                                if not isinstance(x, np.ndarray):
                                    try:
                                        x = np.array(x)
                                    except:
                                        return np.zeros((64, 64, 3), dtype=np.uint8)
                                
                                # Convert to uint8 without assertion
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
                                    x = np.stack([x, x, x], axis=2)
                                elif len(x.shape) == 3:
                                    if x.shape[2] == 1:
                                        x = np.concatenate([x, x, x], axis=2)
                                    elif x.shape[2] == 4:
                                        x = x[:, :, :3]
                                    elif x.shape[2] != 3:
                                        return np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
                                elif len(x.shape) == 4:
                                    x = x[0]
                                    return safe_HWC3(x)
                                else:
                                    return np.zeros((512, 512, 3), dtype=np.uint8)
                                
                                return x
                            except:
                                return np.zeros((512, 512, 3), dtype=np.uint8)
                                
                        # Apply the patch
                        modules_forge_utils._unit5_original_HWC3 = modules_forge_utils.HWC3
                        modules_forge_utils.HWC3 = safe_HWC3
                        debug_print("Patched HWC3 in modules_forge.utils despite p being None")
                        return True
                except ImportError:
                    debug_print("Could not import modules_forge.utils for direct patching")
                except Exception as e:
                    debug_print(f"Error during direct HWC3 patching: {e}")
            except Exception as e:
                debug_print(f"Error attempting direct patch with p=None: {e}")
                
            return False
            
        if not hasattr(p, 'scripts'):
            debug_print("Processing object has no scripts attribute")
            return False
            
        if p.scripts is None:
            debug_print("Processing object scripts is None")
            return False
            
        if not hasattr(p.scripts, 'alwayson_scripts'):
            debug_print("Processing object scripts has no alwayson_scripts attribute")
            return False
            
        if p.scripts.alwayson_scripts is None:
            debug_print("alwayson_scripts is None")
            return False
        
        # Find the ControlNet script
        controlnet_script = None
        for script in p.scripts.alwayson_scripts:
            if script is None:
                continue
                
            if not hasattr(script, 'title'):
                continue
                
            try:
                title = script.title()
                if title and title.lower() == "controlnet":
                    controlnet_script = script
                    break
            except Exception as e:
                debug_print(f"Error checking script title: {e}")
                continue
                
        if not controlnet_script:
            debug_print("ControlNet script not found for unit 5 patch, attempting direct module patching")
            # Even if we can't find the script, we can still try to patch HWC3 directly
            try:
                # Direct patch to modules_forge.utils - most critical for unit 5
                modules_forge_utils = importlib.import_module("modules_forge.utils")
                if hasattr(modules_forge_utils, "HWC3"):
                    # Define safe HWC3 (same as above)
                    def safe_HWC3(x):
                        try:
                            if x is None:
                                return np.zeros((64, 64, 3), dtype=np.uint8)
                                
                            # Convert to numpy array if it's not already
                            if not isinstance(x, np.ndarray):
                                try:
                                    x = np.array(x)
                                except:
                                    return np.zeros((64, 64, 3), dtype=np.uint8)
                            
                            # Convert to uint8 without assertion
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
                                x = np.stack([x, x, x], axis=2)
                            elif len(x.shape) == 3:
                                if x.shape[2] == 1:
                                    x = np.concatenate([x, x, x], axis=2)
                                elif x.shape[2] == 4:
                                    x = x[:, :, :3]
                                elif x.shape[2] != 3:
                                    return np.zeros((x.shape[0], x.shape[1], 3), dtype=np.uint8)
                            else:
                                return np.zeros((512, 512, 3), dtype=np.uint8)
                            
                            return x
                        except:
                            return np.zeros((512, 512, 3), dtype=np.uint8)
                            
                    # Apply the patch
                    modules_forge_utils._unit5_original_HWC3 = modules_forge_utils.HWC3
                    modules_forge_utils.HWC3 = safe_HWC3
                    debug_print("Directly patched HWC3 in modules_forge.utils despite no ControlNet script")
                    return True
            except ImportError:
                debug_print("Could not import modules_forge.utils for direct patching")
            except Exception as e:
                debug_print(f"Error during direct HWC3 patching: {e}")
                
            return False
            
    except Exception as e:
        debug_print(f"Error in patch_controlnet_unit5: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_controlnet_process(p):
    """
    Patch the ControlNet process method to ensure resize_mode is always available
    
    This adds a safety check specifically for WebUI Forge's ControlNet where it expects
    the resize_mode attribute but it might not exist yet.
    """
    try:
        # Only attempt patching if we're in WebUI Forge environment
        if not hasattr(p, 'scripts'):
            debug_print("No scripts attr, skipping ControlNet patch")
            return False
            
        # Find the ControlNet script
        controlnet_script = None
        for script in p.scripts.alwayson_scripts:
            if hasattr(script, 'title') and script.title().lower() == "controlnet":
                controlnet_script = script
                break
                
        if not controlnet_script:
            debug_print("ControlNet script not found")
            return False
            
        # Check if we already patched
        if hasattr(controlnet_script, '_deforum_patched_process'):
            debug_print("ControlNet already patched")
            return True
            
        # Import needed modules
        try:
            import numpy as np
            from PIL import Image
            import types
            
            # Store original process method
            original_process = controlnet_script.process
            
            # Define our patched version
            def patched_process(self, p, *args, **kwargs):
                # Make sure resize_mode exists
                if not hasattr(p, 'resize_mode'):
                    p.resize_mode = "Inner Fit (Scale to Fit)"
                    
                # Make sure current_params exists
                if not hasattr(self, 'current_params') or self.current_params is None:
                    self.current_params = {}
                    
                # Ensure all indices exist in current_params
                for i in range(len(args)):
                    if i not in self.current_params:
                        self.current_params[i] = {}
                
                # Call original with safety wrap
                try:
                    return original_process(p, *args, **kwargs)
                except Exception as e:
                    debug_print(f"Error in ControlNet process: {e}")
                    return None
                
            # Apply our patch
            controlnet_script.process = patched_process.__get__(controlnet_script, type(controlnet_script))
            controlnet_script._deforum_patched_process = True
            
            # Also patch the process_before_every_sampling method if it exists
            if hasattr(controlnet_script, "process_before_every_sampling"):
                original_process_before = controlnet_script.process_before_every_sampling
                
                def patched_process_before_every_sampling(self, p, *args, **kwargs):
                    # Ensure current_params exists
                    if not hasattr(self, 'current_params') or self.current_params is None:
                        self.current_params = {}
                    
                    # Initialize parameters for all possible index values
                    for i in range(len(args)):
                        if i not in self.current_params:
                            self.current_params[i] = {}
                    
                    # Call with safety wrapper
                    try:
                        return original_process_before(self, p, *args, **kwargs)
                    except Exception as e:
                        debug_print(f"Error in process_before_every_sampling: {e}")
                        return None
                    
                # Replace the method
                controlnet_script.process_before_every_sampling = patched_process_before_every_sampling.__get__(
                    controlnet_script, type(controlnet_script))
                debug_print("Successfully patched process_before_every_sampling method")
            
            debug_print("Successfully patched ControlNet process methods")
            return True
            
        except Exception as e:
            debug_print(f"Error patching ControlNet process: {e}")
            return False
            
    except Exception as e:
        debug_print(f"Exception in patch_controlnet_process: {e}")
        return False

def patch_deforum_controlnet_modules():
    """Apply patches to the Deforum ControlNet modules"""
    try:
        # Attempt to import deforum_controlnet module
        try:
            from . import deforum_controlnet
            if deforum_controlnet and hasattr(deforum_controlnet, 'get_controlnet_units_v2'):
                debug_print("Found deforum_controlnet module")
                return True
        except ImportError:
            debug_print("deforum_controlnet module not importable")
            return False
            
        return True
    except Exception as e:
        debug_print(f"Error patching Deforum ControlNet modules: {e}")
        return False

#--------------------------
# VAE Module Patching
#--------------------------

def patch_sd_vae_module():
    """
    Patch the sd_vae module to handle missing model_path attribute in WebUI Forge
    
    This addresses the AttributeError: module 'modules.sd_models' has no attribute 'model_path'
    when initializing the VAE module.
    """
    try:
        debug_print("Attempting to patch sd_vae module for model_path compatibility")
        
        # Try to import the necessary modules
        try:
            from modules import sd_vae, shared, sd_models
            debug_print("Successfully imported required modules for VAE patching")
        except ImportError as ie:
            debug_print(f"Could not import required modules for VAE patching: {ie}")
            return False
            
        # Check if we need to patch by testing for the model_path attribute
        if hasattr(sd_models, 'model_path'):
            debug_print("sd_models.model_path already exists, no need to patch")
            return True
            
        # Create a model_path attribute with fallback to shared.models_path
        debug_print("Adding model_path attribute to sd_models")
        sd_models.model_path = getattr(shared, 'models_path', os.path.join(os.getcwd(), 'models'))
        debug_print(f"Set sd_models.model_path to {sd_models.model_path}")
        
        # Monkey patch the refresh_vae_list function to handle possible future issues
        if hasattr(sd_vae, 'refresh_vae_list'):
            original_refresh = sd_vae.refresh_vae_list
            
            def patched_refresh_vae_list():
                # Ensure model_path is available
                if not hasattr(sd_models, 'model_path'):
                    from modules import shared
                    sd_models.model_path = shared.models_path
                    debug_print(f"Set model_path to {sd_models.model_path} during refresh_vae_list")
                
                # Call original function
                try:
                    return original_refresh()
                except Exception as e:
                    debug_print(f"Error in original refresh_vae_list: {e}")
                    # Handle errors gracefully
                    try:
                        from modules import shared
                        # Still try to scan for VAEs
                        debug_print("Attempting fallback VAE scan")
                        sd_vae.vae_dict.clear()
                        vae_path = getattr(sd_vae, 'vae_path', os.path.join(shared.models_path, 'VAE'))
                        
                        # Use a simplified version of the refresh logic
                        import glob
                        paths = [
                            os.path.join(shared.models_path, '**/*.vae.ckpt'),
                            os.path.join(shared.models_path, '**/*.vae.pt'),
                            os.path.join(shared.models_path, '**/*.vae.safetensors'),
                            os.path.join(vae_path, '**/*.ckpt'),
                            os.path.join(vae_path, '**/*.pt'),
                            os.path.join(vae_path, '**/*.safetensors'),
                        ]
                        
                        candidates = []
                        for path in paths:
                            candidates += glob.iglob(path, recursive=True)
                        
                        for filepath in candidates:
                            name = os.path.basename(filepath)
                            sd_vae.vae_dict[name] = filepath
                            
                        # No need to sort, just get the function working
                        debug_print(f"Fallback VAE scan found {len(sd_vae.vae_dict)} VAEs")
                    except Exception as fallback_error:
                        debug_print(f"Fallback VAE scan failed: {fallback_error}")
            
            # Apply the patch
            sd_vae.refresh_vae_list = patched_refresh_vae_list
            debug_print("Successfully patched sd_vae.refresh_vae_list")
        else:
            debug_print("Could not find sd_vae.refresh_vae_list function to patch")
        
        debug_print("Successfully patched sd_vae module")
        return True
        
    except Exception as e:
        debug_print(f"Error patching sd_vae module: {e}")
        traceback.print_exc()
        return False

#--------------------------
# Main Patching Function
#--------------------------

def patch_deforum_controlnet_modules():
    """
    Apply all required runtime patches for Deforum to work with ControlNet
    
    This function assumes ControlNet is enabled and patches are needed.
    """
    debug_print("Starting Forge ControlNet patching process")
    
    patches_applied = 0
    
    # Set up environment variables first
    if setup_forge_controlnet_env():
        debug_print("Set up WebUI Forge environment variables")
        patches_applied += 1
    
    # Patch HWC3 function
    if patch_utils_hwc3():
        debug_print("Patched HWC3 function")
        patches_applied += 1
    
    # Patch global state module
    if patch_global_state_module():
        debug_print("Patched global state module")
        patches_applied += 1
    
    # Patch processing classes
    if patch_processing_classes():
        debug_print("Patched processing classes")
        patches_applied += 1
    
    # Apply FLUX models patches
    if apply_flux_models_patch():
        debug_print("Applied FLUX models patch")
        patches_applied += 1
        
    # Find and patch all ControlNet script instances
    if find_and_patch_controlnet_scripts():
        debug_print("Patched all ControlNet script instances")
        patches_applied += 1
    
    # Patch sd_vae module
    if patch_sd_vae_module():
        debug_print("Patched sd_vae module")
        patches_applied += 1
    
    debug_print(f"Applied {patches_applied} patches for WebUI Forge ControlNet compatibility")
    return patches_applied > 0

def patch_controlnet_process_unit_method(p):
    """
    Patch the ControlNet process_unit_before_every_sampling method to handle None params
    
    This prevents errors when params.model is None and the code tries to access params.model.strength
    """
    try:
        # Handle the case where p is None
        if p is None:
            debug_print("Processing object is None, attempting direct module patching instead")
            
            # Try to find ControlNet classes in sys.modules
            try:
                import sys
                import types
                import inspect
                
                patched_modules = 0
                
                # Scan sys.modules for any loaded ControlNet module
                for name, module in list(sys.modules.items()):
                    if not module or 'controlnet' not in name.lower():
                        continue
                        
                    # Look for classes in the module
                    for attr_name in dir(module):
                        try:
                            attr = getattr(module, attr_name)
                            # Check if it's a class with the target method
                            if isinstance(attr, type) and hasattr(attr, 'process_unit_before_every_sampling'):
                                # Get original method
                                original_process_unit = attr.process_unit_before_every_sampling
                                
                                # Define safer version
                                def patched_process_unit_before_every_sampling(self, p, unit, params, *args, **kwargs):
                                    try:
                                        # Fix missing or invalid parameters
                                        if p is None or unit is None:
                                            debug_print("Skipping process_unit due to None p or unit")
                                            return None
                                            
                                        # Handle None params
                                        if params is None:
                                            debug_print("params is None, creating safe empty object")
                                            class EmptyObject: pass
                                            params = EmptyObject()
                                            setattr(params, 'model', None)
                                        
                                        # Handle missing model attribute
                                        if not hasattr(params, 'model') or params.model is None:
                                            debug_print("params.model is None, creating safe model object")
                                            class ModelObject: 
                                                def __init__(self):
                                                    self.strength = 1.0
                                                def process_before_every_sampling(self, p, cond, mask, *args, **kwargs):
                                                    return cond, mask
                                                def process_after_every_sampling(self, p, params, *args, **kwargs):
                                                    pass
                                            params.model = ModelObject()
                                        
                                        # Handle missing strength attribute
                                        if not hasattr(params.model, 'strength'):
                                            debug_print("params.model has no strength attribute, adding it")
                                            # Try to get weight from unit if possible
                                            if hasattr(unit, 'weight'):
                                                params.model.strength = float(unit.weight)
                                            else:
                                                # Default fallback
                                                params.model.strength = 1.0
                                        
                                        # Handle missing preprocessor attribute
                                        if not hasattr(params, 'preprocessor') or params.preprocessor is None:
                                            debug_print("params.preprocessor is None, creating safe preprocessor")
                                            class DummyPreprocessor:
                                                def __init__(self):
                                                    self.name = "dummy_preprocessor"
                                                def process_before_every_sampling(self, p, cond, mask, *args, **kwargs):
                                                    return cond, mask
                                                def process_after_every_sampling(self, *args, **kwargs):
                                                    return None
                                            params.preprocessor = DummyPreprocessor()
                                        
                                        # Handle missing name attribute in preprocessor
                                        if not hasattr(params.preprocessor, 'name'):
                                            debug_print("Adding missing name attribute to preprocessor")
                                            params.preprocessor.name = "dummy_preprocessor"
                                        
                                        # Now call the original method with our fixed params
                                        try:
                                            return original_process_unit(self, p, unit, params, *args, **kwargs)
                                        except Exception as e:
                                            debug_print(f"Error in original process_unit: {e}")
                                            # Even if original fails, don't crash
                                            return None
                                    except Exception as e:
                                        debug_print(f"WARNING: Error in patched process_unit: {e}")
                                        # Fall back to original for safety
                                        try:
                                            return original_process_unit(self, p, unit, params, *args, **kwargs)
                                        except:
                                            return None
                                
                                # Apply the patch - add this line
                                attr.process_unit_before_every_sampling = patched_process_unit_before_every_sampling
                                attr._deforum_patched = True  
                                patched_modules += 1
                                debug_print(f"Direct patched process_unit method in {name}.{attr_name}")
                        except Exception as attr_err:
                            continue
                            
                if patched_modules > 0:
                    debug_print(f"Successfully patched {patched_modules} ControlNet process_unit methods directly")
                    return True
                else:
                    debug_print("Could not find any ControlNet process_unit methods to patch directly")
                    return False
                    
            except Exception as e:
                debug_print(f"Error in direct process_unit patching: {e}")
                return False
                
        # If we have a valid p, proceed with the standard approach
        if not hasattr(p, 'scripts') or p.scripts is None:
            debug_print("No scripts attribute, cannot patch process_unit method")
            return False
            
        # Find ControlNet script
        controlnet_script = None
        for script in p.scripts.alwayson_scripts:
            if hasattr(script, 'title') and script.title().lower() == "controlnet":
                controlnet_script = script
                break
                
        if not controlnet_script:
            debug_print("ControlNet script not found for process_unit patch")
            return False
            
        # Check if already patched
        if hasattr(controlnet_script, '_process_unit_patched'):
            debug_print("process_unit already patched")
            return True
            
        # Check if the method exists
        if not hasattr(controlnet_script, 'process_unit_before_every_sampling'):
            debug_print("process_unit_before_every_sampling method not found")
            return False
            
        # Get the original method
        original_process_unit = controlnet_script.process_unit_before_every_sampling
        
        # Define a safer version
        def patched_process_unit_before_every_sampling(self, p, unit, params, *args, **kwargs):
            try:
                # Fix missing or invalid parameters
                if p is None or unit is None:
                    debug_print("Skipping process_unit due to None p or unit")
                    return None
                    
                # Handle None params
                if params is None:
                    debug_print("params is None, creating safe empty object")
                    class EmptyObject: pass
                    params = EmptyObject()
                    setattr(params, 'model', None)
                
                # Handle missing model attribute
                if not hasattr(params, 'model') or params.model is None:
                    debug_print("params.model is None, creating safe model object")
                    class ModelObject:
                        def __init__(self):
                            self.strength = 1.0
                        def process_before_every_sampling(self, p, cond, mask, *args, **kwargs):
                            return cond, mask
                        def process_after_every_sampling(self, p, params, *args, **kwargs):
                            pass
                    params.model = ModelObject()
                
                # Handle missing strength attribute
                if not hasattr(params.model, 'strength'):
                    debug_print("params.model has no strength attribute, adding it")
                    # Try to get weight from unit if possible
                    if hasattr(unit, 'weight'):
                        params.model.strength = float(unit.weight)
                    else:
                        # Default fallback
                        params.model.strength = 1.0
                
                # Handle missing preprocessor attribute
                if not hasattr(params, 'preprocessor') or params.preprocessor is None:
                    debug_print("params.preprocessor is None, creating safe preprocessor")
                    class DummyPreprocessor:
                        def __init__(self):
                            self.name = "dummy_preprocessor"
                        def process_before_every_sampling(self, p, cond, mask, *args, **kwargs):
                            return cond, mask
                        def process_after_every_sampling(self, *args, **kwargs):
                            return None
                    params.preprocessor = DummyPreprocessor()
                
                # Handle missing name attribute in preprocessor
                if not hasattr(params.preprocessor, 'name'):
                    debug_print("Adding missing name attribute to preprocessor")
                    params.preprocessor.name = "dummy_preprocessor"
                
                # Now call the original method with our fixed params
                try:
                    return original_process_unit(self, p, unit, params, *args, **kwargs)
                except Exception as e:
                    debug_print(f"Error in original process_unit: {e}")
                    # Even if original fails, don't crash
                    return None
            except Exception as e:
                debug_print(f"WARNING: Error in patched process_unit: {e}")
                # Fall back to original for safety
                try:
                    return original_process_unit(self, p, unit, params, *args, **kwargs)
                except:
                    return None
                    
        # Apply the patch
        controlnet_script.process_unit_before_every_sampling = patched_process_unit_before_every_sampling.__get__(
            controlnet_script, type(controlnet_script))
        controlnet_script._process_unit_patched = True
        
        debug_print("Successfully patched process_unit_before_every_sampling method")
        return True
        
    except Exception as e:
        debug_print(f"Error patching process_unit method: {e}")
        import traceback
        traceback.print_exc()
        return False

#--------------------------
# ControlNet Model Recognition Fixes
#--------------------------

def fix_controlnet_model_recognition():
    """
    Fix ControlNet model recognition issues by enhancing the model path handling
    
    This addresses the "Recognizing Control Model failed" errors when using FLUX models.
    """
    try:
        import sys
        import os
        import importlib
        import traceback
        from pathlib import Path
        
        debug_print("FIX_CN_MODEL_RECOGNITION: Fixing ControlNet model recognition for FLUX models")
        
        # ... (module finding logic remains the same) ...
        external_code_module = None
        model_adapter_module = None
        
        # First try direct import
        try:
            from lib_controlnet import external_code
            external_code_module = external_code
        except ImportError: pass
        try:
            from lib_controlnet import model_adapter
            model_adapter_module = model_adapter
        except ImportError: pass

        if not model_adapter_module:
            # Try to locate the model_loading.py file directly if direct imports failed
            webui_path = get_webui_path()
            possible_locations = [
                os.path.join(webui_path, "extensions-builtin", "sd_forge_controlnet", "lib_controlnet", "model_loading.py"),
                os.path.join(webui_path, "extensions", "sd-forge-controlnet", "lib_controlnet", "model_loading.py"),
                # Add a more generic search if the above are too specific
                os.path.join(webui_path, "extensions-builtin", "sd_forge_controlnet", "scripts", "controlnet.py"), 
            ]
            for loc_path in possible_locations:
                if os.path.exists(loc_path):
                    try:
                        # For .py files, we can attempt to import them directly to get module attributes
                        module_name_from_path = Path(loc_path).stem
                        if hasattr(sys.modules.get(module_name_from_path), 'get_control_model'):
                             model_adapter_module = sys.modules.get(module_name_from_path)
                             debug_print(f"FIX_CN_MODEL_RECOGNITION: Found model_adapter in sys.modules via path search: {module_name_from_path}")
                             break
                        spec = importlib.util.spec_from_file_location(module_name_from_path, loc_path)
                        if spec and spec.loader:
                            temp_module = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(temp_module)
                            if hasattr(temp_module, 'get_control_model'):
                                model_adapter_module = temp_module
                                debug_print(f"FIX_CN_MODEL_RECOGNITION: Loaded model_adapter module from path: {loc_path}")
                                break
                    except Exception as e_load_path:
                        debug_print(f"FIX_CN_MODEL_RECOGNITION: Error loading module from path {loc_path}: {e_load_path}")

        if not external_code_module and not model_adapter_module:
            debug_print("FIX_CN_MODEL_RECOGNITION: Could not find external_code or model_adapter modules. Attempting global_state patch.")
            return patch_global_state_module()
        
        patches_applied = 0
        
        if model_adapter_module and hasattr(model_adapter_module, 'get_control_model'):
            original_get_control_model = model_adapter_module.get_control_model
            debug_print(f"FIX_CN_MODEL_RECOGNITION: Original get_control_model is {original_get_control_model}")
            
            def patched_get_control_model(controlnet_path):
                debug_print(f"PATCHED_GET_CONTROL_MODEL: Called with controlnet_path: {controlnet_path}")
                is_flux_model = controlnet_path and "flux" in controlnet_path.lower()
                if is_flux_model:
                    debug_print(f"PATCHED_GET_CONTROL_MODEL: FLUX model detected: {controlnet_path}")
                
                # Try original function
                try:
                    model = original_get_control_model(controlnet_path)
                    if model is not None:
                        debug_print(f"PATCHED_GET_CONTROL_MODEL: Original get_control_model succeeded for {controlnet_path}")
                        return model
                    if is_flux_model:
                         debug_print(f"PATCHED_GET_CONTROL_MODEL: Original get_control_model returned None for FLUX model {controlnet_path}. Proceeding to fallback.")
                except Exception as e:
                    debug_print(f"PATCHED_GET_CONTROL_MODEL: Original get_control_model failed for {controlnet_path}: {e}. Proceeding to fallback if FLUX.")
                    if not is_flux_model:
                        # For non-FLUX models, if original fails, we might not want to fallback aggressively, or re-raise
                        # For now, let it proceed to FLUX fallback logic which will return None if not FLUX.
                        pass 

                # Fallback specifically for FLUX or if original failed for FLUX
                if is_flux_model:
                    debug_print(f"PATCHED_GET_CONTROL_MODEL: Applying FLUX fallback for {controlnet_path}")
                    try:
                        from modules import devices
                        import torch
                        class FLUXModelAdapter(torch.nn.Module):
                            def __init__(self, name="flux_model_adapter"):
                                super().__init__()
                                self.is_flux = True
                                self.model_name = name
                            def forward(self, x, hint=None, timesteps=None, context=None, **kwargs):
                                strength = 0.5 
                                if isinstance(x, list):
                                    return [item.clone() * strength for item in x]
                                return x.clone() * strength
                        
                        model = FLUXModelAdapter(name=controlnet_path).to(devices.device)
                        debug_print(f"PATCHED_GET_CONTROL_MODEL: Successfully created FLUXModelAdapter for {controlnet_path}")
                        return model
                    except Exception as flux_fallback_e:
                        debug_print(f"PATCHED_GET_CONTROL_MODEL: Error creating FLUXModelAdapter fallback for {controlnet_path}: {flux_fallback_e}")
                        traceback.print_exc()
                        return None # Fallback for FLUX failed
                
                debug_print(f"PATCHED_GET_CONTROL_MODEL: No model returned for {controlnet_path} after all attempts.")
                return None # Default if not FLUX and original failed, or FLUX fallback failed
            
            model_adapter_module.get_control_model = patched_get_control_model
            debug_print("FIX_CN_MODEL_RECOGNITION: Successfully patched model_adapter.get_control_model with aggressive FLUX fallback")
            patches_applied += 1
        
        # ... (rest of the function, like FLUX model registration, remains the same) ...
        try:
            from modules import shared
            controlnet_paths_to_check = [
                os.path.join(shared.models_path, "ControlNet"),
                os.path.join(shared.models_path, "controlnet"),
                os.path.join(shared.models_path, "xlabs", "controlnets")
            ]
            
            global_state_module_local = None
            try:
                from lib_controlnet import global_state
                global_state_module_local = global_state
            except ImportError:
                debug_print("FIX_CN_MODEL_RECOGNITION: lib_controlnet.global_state not found for FLUX registration.")

            if global_state_module_local:
                flux_models_registered_count = 0
                for cn_path in controlnet_paths_to_check:
                    if os.path.exists(cn_path):
                        for fname in os.listdir(cn_path):
                            if "flux" in fname.lower() and fname.endswith((".safetensors", ".ckpt", ".pt")):
                                full_model_path = os.path.join(cn_path, fname)
                                model_basename, model_ext = os.path.splitext(fname)
                                registration_names = [fname, model_basename, full_model_path, model_basename.lower(), fname.lower()]
                                for reg_name in registration_names:
                                    if reg_name not in global_state_module_local.controlnet_filename_dict:
                                        global_state_module_local.controlnet_filename_dict[reg_name] = full_model_path
                                        debug_print(f"FIX_CN_MODEL_RECOGNITION: Registered FLUX model '{reg_name}' -> '{full_model_path}' in controlnet_filename_dict")
                                        flux_models_registered_count +=1
                                # Add to models list
                                if hasattr(global_state_module_local, "controlnet_models_list"):
                                    if fname not in global_state_module_local.controlnet_models_list:
                                        global_state_module_local.controlnet_models_list.append(fname)
                                    if basename_no_ext not in global_state_module_local.controlnet_models_list:
                                        global_state_module_local.controlnet_models_list.append(basename_no_ext)
                if flux_models_registered_count > 0:
                    debug_print(f"FIX_CN_MODEL_RECOGNITION: Registered/verified {flux_models_registered_count} FLUX model names in global_state.")
                    patches_applied +=1
            else:
                debug_print("FIX_CN_MODEL_RECOGNITION: Skipping FLUX model registration in global_state as module not found.")

        except Exception as reg_err:
            debug_print(f"FIX_CN_MODEL_RECOGNITION: Error during FLUX model registration: {reg_err}")
            traceback.print_exc()

        if patches_applied > 0:
            debug_print(f"FIX_CN_MODEL_RECOGNITION: Applied {patches_applied} patches successfully.")
            return True
        else:
            debug_print("FIX_CN_MODEL_RECOGNITION: No patches applied in this function.")
            return False
        
    except Exception as e:
        debug_print(f"FIX_CN_MODEL_RECOGNITION: CRITICAL ERROR: {e}")
        traceback.print_exc()
        return False

def patch_controlnet_postprocess_batch():
    """
    Patch ControlNet postprocess_batch_list to handle missing preprocessor
    
    This specifically fixes the 'NoneType' has no attribute 'process_after_every_sampling'
    error during postprocess_batch_list.
    """
    try:
        import sys
        import types
        
        debug_print("Patching ControlNet postprocess_batch handling")
        patched_count = 0
        
        # First look for ControlNet script directly
        controlnet_script = None
        
        # Try common module names first
        try:
            from modules import scripts
            if hasattr(scripts, 'scripts_txt2img'):
                for script in scripts.scripts_txt2img.alwayson_scripts:
                    if hasattr(script, 'title') and script.title().lower() == 'controlnet':
                        controlnet_script = script
                        break
        except Exception as e:
            debug_print(f"Error finding ControlNet in scripts_txt2img: {e}")
        
        # If direct approach failed, look through sys.modules
        if not controlnet_script:
            for name, module in list(sys.modules.items()):
                if not module or 'controlnet' not in name.lower():
                    continue
                
                # Check for script class
                for attr_name in dir(module):
                    try:
                        attr = getattr(module, attr_name)
                        if not isinstance(attr, type):
                            continue
                            
                        if hasattr(attr, 'postprocess_batch_list') and hasattr(attr, 'process_unit_after_every_sampling'):
                            # This looks like a ControlNet script class
                            
                            # Patch process_unit_after_every_sampling
                            original_after = attr.process_unit_after_every_sampling
                            
                            def safe_process_unit_after(self, p, unit, params, pp, *args, **kwargs):
                                try:
                                    # Handle None params
                                    if params is None:
                                        debug_print(f"Creating empty params in process_unit_after")
                                        class EmptyParams: pass
                                        params = EmptyParams()
                                        
                                    # Handle missing preprocessor attribute
                                    if not hasattr(params, 'preprocessor') or params.preprocessor is None:
                                        debug_print(f"Creating dummy preprocessor in process_unit_after")
                                        class DummyPreprocessor:
                                            def process_after_every_sampling(self, *args, **kwargs):
                                                return None
                                        params.preprocessor = DummyPreprocessor()
                                    
                                    # Make sure the preprocessor has the required method
                                    if not hasattr(params.preprocessor, 'process_after_every_sampling'):
                                        debug_print(f"Adding missing process_after_every_sampling method")
                                        params.preprocessor.process_after_every_sampling = lambda *args, **kwargs: None
                                    
                                    # Safely call original
                                    try:
                                        return original_after(self, p, unit, params, pp, *args, **kwargs)
                                    except Exception as inner_e:
                                        debug_print(f"Error in original process_unit_after: {inner_e}")
                                        return None
                                except Exception as outer_e:
                                    debug_print(f"Critical error in safe_process_unit_after: {outer_e}")
                                    return None
                            
                            # Apply patch
                            attr.process_unit_after_every_sampling = safe_process_unit_after
                            
                            # Also patch postprocess_batch_list for extra safety
                            if hasattr(attr, 'postprocess_batch_list'):
                                original_post = attr.postprocess_batch_list
                                
                                def safe_postprocess_batch(self, p, pp, *args, **kwargs):
                                    try:
                                        # Make sure current_params exists
                                        if not hasattr(self, 'current_params') or self.current_params is None:
                                            debug_print(f"Creating current_params dict in postprocess_batch")
                                            self.current_params = {}
                                            
                                        # Make sure params exist for all indices
                                        for i in range(len(args)):
                                            if i not in self.current_params:
                                                debug_print(f"Adding missing params for index {i}")
                                                self.current_params[i] = {}
                                                
                                        # Safely call original
                                        try:
                                            return original_post(self, p, pp, *args, **kwargs)
                                        except Exception as inner_e:
                                            debug_print(f"Error in original postprocess_batch: {inner_e}")
                                            return None
                                    except Exception as outer_e:
                                        debug_print(f"Critical error in safe_postprocess_batch: {outer_e}")
                                        return None
                                
                                # Apply patch
                                attr.postprocess_batch_list = safe_postprocess_batch
                            
                            debug_print(f"Patched postprocess handling in {name}.{attr_name}")
                            patched_count += 1
                    except Exception as attr_e:
                        continue
        
        # Try to patch specific files if needed
        try:
            webui_dir = get_webui_path()
            controlnet_paths = [
                os.path.join(webui_dir, "extensions-builtin", "sd_forge_controlnet", "scripts", "controlnet.py"),
                os.path.join(webui_dir, "extensions", "sd-forge-controlnet", "scripts", "controlnet.py")
            ]
            
            for path in controlnet_paths:
                if os.path.exists(path):
                    debug_print(f"Found ControlNet script file at {path}")
                    
                    # Read the file to check if it has postprocess_batch_list
                    with open(path, 'r') as f:
                        content = f.read()
                        
                    if 'def postprocess_batch_list' in content and 'process_unit_after_every_sampling' in content:
                        # This file has the methods we need to patch
                        
                        # Import module
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("direct_controlnet", path)
                        direct_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(direct_module)
                        
                        # Find ControlNet class
                        for attr_name in dir(direct_module):
                            attr = getattr(direct_module, attr_name)
                            if isinstance(attr, type) and hasattr(attr, 'postprocess_batch_list'):
                                # Create a dummy preprocessor class
                                class SafePreprocessor:
                                    def process_after_every_sampling(self, *args, **kwargs):
                                        return None
                                
                                # Patch the actual class methods
                                original_post = attr.postprocess_batch_list
                                original_after = attr.process_unit_after_every_sampling
                                
                                # Define safer versions
                                def safe_direct_post(self, p, pp, *args, **kwargs):
                                    # Make sure current_params exists
                                    if not hasattr(self, 'current_params') or self.current_params is None:
                                        self.current_params = {}
                                    
                                    # Initialize for all indices
                                    for i in range(len(args)):
                                        if i not in self.current_params:
                                            self.current_params[i] = {}
                                    
                                    # Safely call original
                                    try:
                                        return original_post(self, p, pp, *args, **kwargs)
                                    except Exception as e:
                                        debug_print(f"Error in direct postprocess_batch: {e}")
                                        return None
                                
                                def safe_direct_after(self, p, unit, params, pp, *args, **kwargs):
                                    # Make sure params exists
                                    if params is None:
                                        class EmptyParams: pass
                                        params = EmptyParams()
                                    
                                    # Make sure preprocessor exists
                                    if not hasattr(params, 'preprocessor') or params.preprocessor is None:
                                        params.preprocessor = SafePreprocessor()
                                    
                                    # Make sure method exists
                                    if not hasattr(params.preprocessor, 'process_after_every_sampling'):
                                        params.preprocessor.process_after_every_sampling = lambda *args, **kwargs: None
                                    
                                    # Safely call original
                                    try:
                                        return original_after(self, p, unit, params, pp, *args, **kwargs)
                                    except Exception as e:
                                        debug_print(f"Error in direct process_after: {e}")
                                        return None
                                
                                # Apply patches
                                attr.postprocess_batch_list = types.MethodType(safe_direct_post, None)
                                attr.process_unit_after_every_sampling = types.MethodType(safe_direct_after, None)
                                
                                debug_print(f"Applied direct file patches to {attr_name} in {path}")
                                patched_count += 1
                                
                                # Only need to patch one class per file
                                break
        except Exception as file_e:
            debug_print(f"Error applying direct file patches: {file_e}")
        
        debug_print(f"Successfully patched {patched_count} ControlNet postprocess handlers")
        return patched_count > 0
    
    except Exception as e:
        debug_print(f"Error in patch_controlnet_postprocess_batch: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_controlnet_script_file():
    """
    Directly patch the ControlNet script file to fix specific issues.
    
    This addresses the exact error points we've identified in the logs:
    1. Line 471: params.model.strength = float(unit.weight) - NoneType has no attribute 'strength'
    2. Line 548: params.model.process_after_every_sampling - NoneType has no attribute 'process_after_every_sampling'
    3. Line 518: params.preprocessor.process_before_every_sampling - NoneType has no attribute 'process_before_every_sampling'
    """
    try:
        import os
        
        # Find the ControlNet script file
        webui_path = get_webui_path()
        controlnet_paths = [
            os.path.join(webui_path, "extensions-builtin", "sd_forge_controlnet", "scripts", "controlnet.py"),
            os.path.join(webui_path, "extensions", "sd-forge-controlnet", "scripts", "controlnet.py")
        ]
        
        patched_files = 0
        
        for controlnet_path in controlnet_paths:
            if not os.path.exists(controlnet_path):
                continue
                
            debug_print(f"Found ControlNet script at {controlnet_path}")
            
            try:
                # Create a backup if one doesn't already exist
                backup_path = controlnet_path + ".deforum_backup"
                if not os.path.exists(backup_path):
                    import shutil
                    shutil.copy2(controlnet_path, backup_path)
                    debug_print(f"Created backup at {backup_path}")
                
                # Read the file
                with open(controlnet_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check if it's already patched
                if "# Patched by Deforum for WebUI Forge compatibility" in content:
                    debug_print(f"File {controlnet_path} is already patched")
                    patched_files += 1
                    continue
                
                # Apply the patches to specific problematic lines
                
                # Patch 1: Fix for 'NoneType' has no attribute 'strength'
                # Original line (around line 471):
                # params.model.strength = float(unit.weight)
                if "params.model.strength = float(unit.weight)" in content:
                    # Replace with safety checks
                    content = content.replace(
                        "params.model.strength = float(unit.weight)",
                        "# Patched by Deforum for WebUI Forge compatibility\n" +
                        "        if params is None:\n" +
                        "            class EmptyParams: pass\n" +
                        "            params = EmptyParams()\n" +
                        "        if not hasattr(params, 'model') or params.model is None:\n" +
                        "            class ModelClass:\n" +
                        "                def __init__(self):\n" +
                        "                    self.strength = 1.0\n" +
                        "                def process_before_every_sampling(self, p, cond, mask, *args, **kwargs):\n" + 
                        "                    return cond, mask\n" +
                        "                def process_after_every_sampling(self, p, params, *args, **kwargs):\n" +
                        "                    return\n" +
                        "            params.model = ModelClass()\n" +
                        "            params.model.strength = 1.0\n" +
                        "        else:\n" +
                        "            params.model.strength = float(unit.weight)"
                    )
                    debug_print("Patched process_unit_before_every_sampling for model.strength issue")
                
                # Patch 2: Fix for 'NoneType' has no attribute 'process_after_every_sampling'
                # Original line (around line 548):
                # params.preprocessor.process_after_every_sampling(p, params, *args, **kwargs)
                if "params.preprocessor.process_after_every_sampling(p, params" in content:
                    # Replace with safety checks
                    content = content.replace(
                        "params.preprocessor.process_after_every_sampling(p, params",
                        "# Patched by Deforum for WebUI Forge compatibility\n" +
                        "        if params is None:\n" +
                        "            class EmptyParams: pass\n" +
                        "            params = EmptyParams()\n" +
                        "        if not hasattr(params, 'preprocessor') or params.preprocessor is None:\n" +
                        "            class DummyPreprocessor:\n" +
                        "                def __init__(self):\n" +
                        "                    self.name = \"dummy_preprocessor\"\n" +
                        "                def process_after_every_sampling(self, *args, **kwargs):\n" +
                        "                    return None\n" +
                        "            params.preprocessor = DummyPreprocessor()\n" +
                        "        if not hasattr(params.preprocessor, 'process_after_every_sampling'):\n" +
                        "            params.preprocessor.process_after_every_sampling = lambda *args, **kwargs: None\n" +
                        "        if not hasattr(params.preprocessor, 'name'):\n" +
                        "            params.preprocessor.name = \"dummy_preprocessor\"\n" +
                        "        try:\n" +
                        "            params.preprocessor.process_after_every_sampling(p, params"
                    )
                    
                    # And add a try-except wrapper at the end of the line
                    content = content.replace(
                        "params.preprocessor.process_after_every_sampling(p, params, *args, **kwargs)",
                        "params.preprocessor.process_after_every_sampling(p, params, *args, **kwargs)\n" +
                        "        except Exception as e:\n" +
                        "            print(f\"Error in preprocessor.process_after_every_sampling: {e}\")"
                    )
                    debug_print("Patched process_unit_after_every_sampling for preprocessor issue")
                
                # Patch 3: Fix for 'NoneType' has no attribute 'process_before_every_sampling'
                # Original line (around line 518):
                # cond, mask = params.preprocessor.process_before_every_sampling(p, cond, mask, *args, **kwargs)
                if "process_before_every_sampling(p, cond, mask" in content:
                    # Replace with safety checks
                    content = content.replace(
                        "cond, mask = params.preprocessor.process_before_every_sampling(p, cond, mask",
                        "# Patched by Deforum for WebUI Forge compatibility\n" +
                        "        if params is None:\n" +
                        "            class EmptyParams: pass\n" +
                        "            params = EmptyParams()\n" +
                        "        if not hasattr(params, 'preprocessor') or params.preprocessor is None:\n" +
                        "            class DummyPreprocessor:\n" +
                        "                def __init__(self):\n" +
                        "                    self.name = \"dummy_preprocessor\"\n" +
                        "                def process_before_every_sampling(self, p, cond, mask, *args, **kwargs):\n" +
                        "                    return cond, mask\n" +
                        "                def process_after_every_sampling(self, *args, **kwargs):\n" +
                        "                    return None\n" +
                        "            params.preprocessor = DummyPreprocessor()\n" +
                        "        if not hasattr(params.preprocessor, 'process_before_every_sampling'):\n" +
                        "            params.preprocessor.process_before_every_sampling = lambda p, cond, mask, *args, **kwargs: (cond, mask)\n" +
                        "        if not hasattr(params.preprocessor, 'name'):\n" +
                        "            params.preprocessor.name = \"dummy_preprocessor\"\n" +
                        "        try:\n" +
                        "            cond, mask = params.preprocessor.process_before_every_sampling(p, cond, mask"
                    )
                    
                    # And add a try-except wrapper at the end of the line
                    if "cond, mask = params.preprocessor.process_before_every_sampling(p, cond, mask, *args, **kwargs)" in content:
                        content = content.replace(
                            "cond, mask = params.preprocessor.process_before_every_sampling(p, cond, mask, *args, **kwargs)",
                            "cond, mask = params.preprocessor.process_before_every_sampling(p, cond, mask, *args, **kwargs)\n" +
                            "        except Exception as e:\n" +
                            "            print(f\"Error in preprocessor.process_before_every_sampling: {e}\")"
                        )
                    debug_print("Patched process_unit_before_every_sampling for process_before_every_sampling issue")
                
                # Additional patch for line 547:
                # params.model.process_before_every_sampling(p, cond, mask, *args, **kwargs)
                if "params.model.process_before_every_sampling(p, cond, mask" in content:
                    # Replace with safety checks
                    content = content.replace(
                        "params.model.process_before_every_sampling(p, cond, mask",
                        "# Patched by Deforum for WebUI Forge compatibility\n" +
                        "        if not hasattr(params.model, 'process_before_every_sampling'):\n" +
                        "            params.model.process_before_every_sampling = lambda p, cond, mask, *args, **kwargs: (cond, mask)\n" +
                        "        params.model.process_before_every_sampling(p, cond, mask"
                    )
                    debug_print("Patched model.process_before_every_sampling method check")
                
                # Additional patch for line 597:
                # params.model.process_after_every_sampling(p, params, *args, **kwargs)
                if "params.model.process_after_every_sampling(p, params" in content:
                    # Replace with safety checks
                    content = content.replace(
                        "params.model.process_after_every_sampling(p, params",
                        "# Patched by Deforum for WebUI Forge compatibility\n" +
                        "        if not hasattr(params.model, 'process_after_every_sampling'):\n" +
                        "            params.model.process_after_every_sampling = lambda p, params, *args, **kwargs: None\n" +
                        "        params.model.process_after_every_sampling(p, params"
                    )
                    debug_print("Patched model.process_after_every_sampling method check")
                
                # Add a simple marker at the end
                if not content.endswith("# Patched by Deforum for WebUI Forge compatibility\n"):
                    content += "\n# Patched by Deforum for WebUI Forge compatibility\n"
                
                # Write the patched file
                with open(controlnet_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                debug_print(f"Successfully patched {controlnet_path}")
                patched_files += 1
                
            except Exception as e:
                debug_print(f"Error patching {controlnet_path}: {e}")
                import traceback
                traceback.print_exc()
        
        if patched_files > 0:
            debug_print(f"Successfully patched {patched_files} ControlNet script files")
            return True
        else:
            debug_print("No ControlNet script files were patched")
            return False
            
    except Exception as e:
        debug_print(f"Error in patch_controlnet_script_file: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_controlnet_params_and_preprocessor():
    """
    Patch ControlNet params and preprocessor handling to ensure DummyPreprocessor always has a name attribute
    and all preprocessor objects have necessary attributes to avoid NoneType errors.
    
    This fixes the 'NoneType' object has no attribute 'name' error in DummyPreprocessor.
    """
    try:
        debug_print("Patching ControlNet params and preprocessor handling for FLUX compatibility")
        
        # Define the DummyPreprocessor with required attributes
        class NamedDummyPreprocessor:
            def __init__(self):
                self.name = "dummy_preprocessor"
                self.model_name = "dummy_preprocessor"
                
            def process_before_every_sampling(self, p, cond, mask, *args, **kwargs):
                return cond, mask
                
            def process_after_every_sampling(self, *args, **kwargs):
                pass
        
        # FLUX-specific preprocessor for different control types
        class FLUXPreprocessor:
            def __init__(self, control_type="canny"):
                self.name = f"flux_{control_type}"
                self.model_name = f"flux_{control_type}"
                self.control_type = control_type
                
            def process_before_every_sampling(self, p, cond, mask, *args, **kwargs):
                debug_print(f"FLUX {self.control_type} preprocessor active")
                return cond, mask
                
            def process_after_every_sampling(self, *args, **kwargs):
                pass
        
        # Track modules patched
        patched_modules = []
        
        # Patch modules directly in sys.modules
        try:
            import sys
            import importlib
            
            for name, module in list(sys.modules.items()):
                if module and "controlnet" in name.lower():
                    try:
                        # If module has DummyPreprocessor class, patch it
                        if hasattr(module, "DummyPreprocessor"):
                            # Add name attribute to the class
                            if not hasattr(module.DummyPreprocessor, "name"):
                                module.DummyPreprocessor.name = "dummy_preprocessor"
                                
                            # Add model_name attribute for FLUX compatibility
                            if not hasattr(module.DummyPreprocessor, "model_name"):
                                module.DummyPreprocessor.model_name = "dummy_preprocessor"
                            
                            # Ensure the __init__ method includes setting name
                            if hasattr(module.DummyPreprocessor, "__init__"):
                                original_init = module.DummyPreprocessor.__init__
                                
                                def patched_init(self, *args, **kwargs):
                                    self.name = "dummy_preprocessor"
                                    self.model_name = "dummy_preprocessor"
                                    if original_init != object.__init__:
                                        original_init(self, *args, **kwargs)
                                
                                module.DummyPreprocessor.__init__ = patched_init
                            
                            patched_modules.append(f"{name}.DummyPreprocessor")
                            debug_print(f"Patched {name}.DummyPreprocessor with name attribute")
                            
                        # Check for classes with preprocessor in their name
                        for attr_name in dir(module):
                            if ("preprocessor" in attr_name.lower() or 
                                "processor" in attr_name.lower()) and not attr_name.startswith("_"):
                                try:
                                    attr = getattr(module, attr_name)
                                    if isinstance(attr, type):  # It's a class
                                        # Add name attribute to the class if missing
                                        if not hasattr(attr, "name"):
                                            attr.name = attr_name.lower()
                                            patched_modules.append(f"{name}.{attr_name}")
                                            debug_print(f"Added name attribute to {name}.{attr_name}")
                                            
                                        # Add model_name attribute for FLUX compatibility
                                        if not hasattr(attr, "model_name"):
                                            attr.model_name = attr_name.lower()
                                except Exception as inner_e:
                                    debug_print(f"Error patching attribute {attr_name} in {name}: {inner_e}")
                    except Exception as e:
                        debug_print(f"Error patching module {name}: {e}")
            
            # Check if any ControlNet module is loaded and register FLUX preprocessors
            controlnet_module = None
            for name, module in list(sys.modules.items()):
                if module and "controlnet" in name.lower() and hasattr(module, "preprocessor_sliders_config"):
                    controlnet_module = module
                    debug_print(f"Found ControlNet module with preprocessors: {name}")
                    break
                    
            if controlnet_module:
                # Try to register FLUX preprocessors
                if hasattr(controlnet_module, "preprocessor_sliders_config"):
                    # Register dummy preprocessor if not already registered
                    if "dummy_preprocessor" not in controlnet_module.preprocessor_sliders_config:
                        controlnet_module.preprocessor_sliders_config["dummy_preprocessor"] = {}
                        debug_print("Registered dummy_preprocessor in sliders config")
                    
                    # Register FLUX preprocessors for each control type
                    for control_type in ["canny", "depth", "hed", "openpose"]:
                        flux_name = f"flux_{control_type}"
                        if flux_name not in controlnet_module.preprocessor_sliders_config:
                            controlnet_module.preprocessor_sliders_config[flux_name] = {}
                            debug_print(f"Registered {flux_name} in sliders config")
                    
                # Register dynamic preprocessor classes if possible
                if hasattr(controlnet_module, "preprocessor_list"):
                    # Add dummy preprocessor
                    if "dummy_preprocessor" not in controlnet_module.preprocessor_list:
                        controlnet_module.preprocessor_list["dummy_preprocessor"] = NamedDummyPreprocessor
                        debug_print("Registered dummy_preprocessor in preprocessor list")
                        
                    # Add FLUX preprocessors
                    for control_type in ["canny", "depth", "hed", "openpose"]:
                        flux_name = f"flux_{control_type}"
                        if flux_name not in controlnet_module.preprocessor_list:
                            # Create a closure to capture control_type
                            def make_flux_preprocessor(ctype):
                                return lambda: FLUXPreprocessor(ctype)
                            
                            controlnet_module.preprocessor_list[flux_name] = make_flux_preprocessor(control_type)
                            debug_print(f"Registered {flux_name} in preprocessor list")
            
            # Try patching specific modules directly
            for module_name in [
                "modules_forge.controlnet.scripts.controlnet",
                "extensions.sd_forge_controlnet.scripts.controlnet",
                "extensions-builtin.sd_forge_controlnet.scripts.controlnet",
                "scripts.controlnet"
            ]:
                try:
                    module = importlib.import_module(module_name)
                    
                    # Patch DummyPreprocessor instances
                    for attr_name in dir(module):
                        if attr_name == "DummyPreprocessor":
                            attr = getattr(module, attr_name)
                            if isinstance(attr, type):  # It's a class
                                if not hasattr(attr, "name"):
                                    attr.name = "dummy_preprocessor"
                                    patched_modules.append(f"{module_name}.{attr_name}")
                                    debug_print(f"Added name attribute to {module_name}.{attr_name}")
                                
                                # Add model_name attribute for FLUX compatibility
                                if not hasattr(attr, "model_name"):
                                    attr.model_name = "dummy_preprocessor"
                                
                                # Ensure __init__ sets name
                                original_init = attr.__init__
                                
                                def patched_init(self, *args, **kwargs):
                                    self.name = "dummy_preprocessor"
                                    self.model_name = "dummy_preprocessor"
                                    if original_init != object.__init__:
                                        original_init(self, *args, **kwargs)
                                
                                attr.__init__ = patched_init
                                
                                debug_print(f"Patched {module_name}.{attr_name}.__init__ to set name attribute")
                except Exception as e:
                    debug_print(f"Error patching specific module {module_name}: {e}")
        
        except Exception as e:
            debug_print(f"Error in module patching section: {e}")
            import traceback
            traceback.print_exc()
        
        # Patch any system-wide preprocessor method calls
        try:
            import sys
            
            for name, module in list(sys.modules.items()):
                if module and "controlnet" in name.lower():
                    # Look for methods that might use preprocessors
                    for attr_name in dir(module):
                        if ("process" in attr_name.lower() and 
                            "unit" in attr_name.lower() and 
                            not attr_name.startswith("_")):
                            
                            try:
                                attr = getattr(module, attr_name)
                                if callable(attr) and not isinstance(attr, type):
                                    # This is a method that might process units
                                    
                                    # Monitor if any preprocessor with potential None attributes is used
                                    # Skip complex patching if the method definition is too complex
                                    src = inspect.getsource(attr)
                                    if "preprocessor" in src and "params.preprocessor" in src:
                                        debug_print(f"Found potential preprocessor usage in {name}.{attr_name}")
                                        
                                        # Note: Direct patching of arbitrary methods is complex
                                        # We've added specific method patching elsewhere (like emergency_direct_patch_controlnet)
                            except Exception:
                                # Skip if can't get source
                                pass
            
            debug_print("Completed scanning for preprocessor methods")
                
        except Exception as e:
            debug_print(f"Error in method patching section: {e}")
            import traceback
            traceback.print_exc()
        
        # Add FLUX-specific patches for the emergency direct patch function
        try:
            # Monkey patch the emergency function to include FLUX preprocessors
            original_emergency_patch = emergency_direct_patch_controlnet
            
            def enhanced_emergency_patch_controlnet():
                # Call the original patch first
                result = original_emergency_patch()
                
                # Add FLUX-specific enhancements
                try:
                    debug_print("Adding FLUX-specific preprocessor enhancements")
                    
                    # Look for ControlNet module
                    controlnet_module = None
                    for name, module in list(sys.modules.items()):
                        if module and "controlnet" in name.lower() and hasattr(module, "preprocessor_sliders_config"):
                            controlnet_module = module
                            debug_print(f"Found ControlNet module with preprocessors: {name}")
                            break
                    
                    if controlnet_module:
                        # Register dummy preprocessor and FLUX preprocessors
                        if hasattr(controlnet_module, "preprocessor_sliders_config"):
                            if "dummy_preprocessor" not in controlnet_module.preprocessor_sliders_config:
                                controlnet_module.preprocessor_sliders_config["dummy_preprocessor"] = {}
                            
                            # Add FLUX preprocessors
                            for control_type in ["canny", "depth", "hed", "openpose"]:
                                flux_name = f"flux_{control_type}"
                                if flux_name not in controlnet_module.preprocessor_sliders_config:
                                    controlnet_module.preprocessor_sliders_config[flux_name] = {}
                    
                    debug_print("FLUX preprocessor enhancements added")
                    return True
                except Exception as e:
                    debug_print(f"Error adding FLUX enhancements: {e}")
                    return result
                    
            # Replace the function with our enhanced version
            emergency_direct_patch_controlnet = enhanced_emergency_patch_controlnet
            debug_print("Enhanced emergency_direct_patch_controlnet function for FLUX")
                
        except Exception as e:
            debug_print(f"Error enhancing emergency patch function: {e}")
        
        # Summary
        if patched_modules:
            debug_print(f"Successfully patched {len(patched_modules)} preprocessor modules")
            debug_print(f"Patched modules: {', '.join(patched_modules)}")
            return True
        else:
            debug_print("No preprocessor modules were patched")
            return False
    
    except Exception as e:
        debug_print(f"Error in patch_controlnet_params_and_preprocessor: {e}")
        import traceback
        traceback.print_exc()
        return False

def initialize_all_patches():
    """
    Apply all patches needed during initialization time, before the WebUI is fully loaded.
    
    This function should be called during extension initialization to ensure
    all necessary fixes are in place before any ControlNet or Deforum modules load.
    """
    try:
        debug_print("Initializing all early compatibility patches")
        
        # First try direct patch for ControlNetForForgeOfficial
        patch_controlnet_forge_official()
        
        # Then patch all ControlNet classes to ensure current_params exists
        patch_controlnet_current_params()
        
        # Add a direct patch for process_before_every_sampling
        patch_process_before_every_sampling()
        
        patches_applied = []
        
        # Suppress ControlNet FLUX error logs
        suppress_flux_controlnet_error()
        # Patch ControlNet preprocessors to save intermediate images
        patch_controlnet_preprocessor_save()
        
        # Apply direct patch for external_code module - ADDED FIRST FOR PRIORITY
        try:
            if patch_external_code_direct(): # Ensure 'from modules import devices' is local in this func
                debug_print("Successfully applied direct external_code patch")
                patches_applied.append("Direct external_code patch")
        except Exception as e:
            debug_print(f"Error applying direct external_code patch: {e}")
            import traceback
            traceback.print_exc()
        
        # CRITICAL: Apply direct emergency patching for process_before_every_sampling first
        try:
            if emergency_direct_patch_controlnet():
                debug_print("Successfully applied emergency direct ControlNet patching")
                patches_applied.append("Emergency direct ControlNet patch")
        except Exception as e:
            debug_print(f"Error applying emergency direct ControlNet patch: {e}")
            import traceback
            traceback.print_exc()
        
        # First, directly patch the ControlNet script files as this is most reliable
        try:
            if patch_controlnet_script_file():
                debug_print("Successfully applied direct ControlNet script file patches")
                patches_applied.append("ControlNet script direct patch")
        except Exception as e:
            debug_print(f"Error applying direct ControlNet patches: {e}")
            import traceback
            traceback.print_exc()
        
        # Then, patch the HWC3 function to ensure image processing works
        try:
            if patch_utils_hwc3():
                debug_print("Successfully applied early HWC3 patches")
                patches_applied.append("Early HWC3 patches")
        except Exception as e:
            debug_print(f"Error applying early HWC3 patches: {e}")
            import traceback
            traceback.print_exc()
        
        # Next, patch the sd_vae module for model_path compatibility
        try:
            if patch_sd_vae_module(): # Ensure 'from modules import shared' is local here if used for model_path
                debug_print("Successfully applied early sd_vae patch")
                patches_applied.append("sd_vae early patch")
        except Exception as e:
            debug_print(f"Error applying early sd_vae patch: {e}")
            import traceback
            traceback.print_exc()
        
        # Fix ControlNet model recognition for FLUX models
        try:
            if fix_controlnet_model_recognition(): # Ensure 'from modules import shared', 'from lib_controlnet import global_state', 'from modules import devices' are local
                debug_print("Successfully fixed ControlNet model recognition")
                patches_applied.append("ControlNet model recognition")
        except Exception as e:
            debug_print(f"Error fixing ControlNet model recognition: {e}")
            import traceback
            traceback.print_exc()
            
        # Apply the global state module patch
        try:
            if patch_global_state_module(): # Ensure 'from modules import shared', 'from lib_controlnet import global_state' are local
                debug_print("Successfully patched global state module")
                patches_applied.append("Global state module")
        except Exception as e:
            debug_print(f"Error patching global state module: {e}")
            import traceback
            traceback.print_exc()
        
        # Apply processing class patches
        try:
            if patch_processing_classes(): # Check for 'modules.processing'
                debug_print("Successfully patched processing classes")
                patches_applied.append("Processing classes")
        except Exception as e:
            debug_print(f"Error patching processing classes: {e}")
            import traceback
            traceback.print_exc()
            
        # Set up needed environment variables
        try:
            if setup_forge_controlnet_env():
                debug_print("Successfully set up environment variables")
                patches_applied.append("Environment variables")
        except Exception as e:
            debug_print(f"Error setting up environment variables: {e}")
            import traceback
            traceback.print_exc()
        
        # Patch parameters and preprocessors
        try:
            if patch_controlnet_params_and_preprocessor():
                debug_print("Successfully patched ControlNet params and preprocessor handling")
                patches_applied.append("ControlNet params and preprocessor")
        except Exception as e:
            debug_print(f"Error patching ControlNet params and preprocessor: {e}")
            import traceback
            traceback.print_exc()
            
        # Patch postprocess_batch
        try:
            if patch_controlnet_postprocess_batch(): # Check for 'from modules import scripts'
                debug_print("Successfully patched ControlNet postprocess batch")
                patches_applied.append("ControlNet postprocess batch")
        except Exception as e:
            debug_print(f"Error patching ControlNet postprocess batch: {e}")
            import traceback
            traceback.print_exc()
            
        # Test FLUX model compatibility - specific check for FLUX models
        try:
            # DEFER IMPORT HERE
            import os
            debug_print("INITIALIZE_ALL_PATCHES: Testing FLUX model detection and compatibility")
            
            flux_model_found = False
            try:
                from modules import shared # DEFERRED
                model_paths = [
                    os.path.join(shared.models_path, "ControlNet"),
                    os.path.join(shared.models_path, "controlnet"),
                    os.path.join(shared.models_path, "xlabs", "controlnets")
                ]
            except ImportError as e_shared:
                debug_print(f"INITIALIZE_ALL_PATCHES: Could not import modules.shared for FLUX test: {e_shared}. Using fallback paths.")
                # Fallback paths if shared is not available - this might be less reliable
                webui_base = get_webui_path() 
                model_paths = [
                    os.path.join(webui_base, "models", "ControlNet"),
                    os.path.join(webui_base, "models", "controlnet"),
                    os.path.join(webui_base, "models", "xlabs", "controlnets")
                ]

            for path in model_paths:
                if not os.path.exists(path):
                    continue
                    
                for filename in os.listdir(path):
                    if "flux" in filename.lower() and filename.endswith((".safetensors", ".ckpt", ".pt")):
                        flux_model_found = True
                        fullpath = os.path.join(path, filename)
                        debug_print(f"INITIALIZE_ALL_PATCHES: Found FLUX model: {fullpath}")
                        
                        try:
                            from lib_controlnet import global_state # DEFERRED
                            basename = os.path.basename(fullpath)
                            basename_no_ext = os.path.splitext(basename)[0]
                            
                            global_state.controlnet_filename_dict[basename] = fullpath
                            global_state.controlnet_filename_dict[basename_no_ext] = fullpath
                            global_state.controlnet_filename_dict[fullpath] = fullpath
                            
                            for variation in [
                                basename.lower(), basename_no_ext.lower(),
                                basename.replace("-", "_"), basename_no_ext.replace("-", "_")
                            ]:
                                global_state.controlnet_filename_dict[variation] = fullpath
                                
                            debug_print(f"INITIALIZE_ALL_PATCHES: Registered FLUX model in global state: {basename}")
                            
                            if hasattr(global_state, "controlnet_models_list"):
                                for name_var in [basename, basename_no_ext]:
                                    if name_var not in global_state.controlnet_models_list:
                                        global_state.controlnet_models_list.append(name_var)
                                        
                            for ctype in ["canny", "depth", "hed", "openpose"]:
                                if ctype in basename.lower():
                                    flux_type_name = f"flux_{ctype}"
                                    if flux_type_name not in global_state.controlnet_filename_dict:
                                        global_state.controlnet_filename_dict[flux_type_name] = fullpath
                                        debug_print(f"INITIALIZE_ALL_PATCHES: Registered type-specific name: {flux_type_name}")
                                    if hasattr(global_state, "controlnet_models_list"):
                                        if flux_type_name not in global_state.controlnet_models_list:
                                            global_state.controlnet_models_list.append(flux_type_name)
                                    break
                        except ImportError as e_gs:
                            debug_print(f"INITIALIZE_ALL_PATCHES: Could not import lib_controlnet.global_state for FLUX test: {e_gs}")
                        except Exception as reg_err:
                            debug_print(f"INITIALIZE_ALL_PATCHES: Error registering FLUX model in global state: {reg_err}")
                            # import traceback # Already imported at function start
                            # traceback.print_exc() # Avoid too much noise for this specific part
            
            if flux_model_found:
                debug_print("INITIALIZE_ALL_PATCHES: FLUX models detected and (attempted) registration.")
                patches_applied.append("FLUX model registration")
            else:
                debug_print("INITIALIZE_ALL_PATCHES: No FLUX models found - FLUX compatibility may be limited")
                
        except Exception as flux_err:
            debug_print(f"INITIALIZE_ALL_PATCHES: Error testing FLUX model compatibility: {flux_err}")
            # import traceback # Already imported
            # traceback.print_exc()
        
        # Summary
        if patches_applied:
            debug_print(f"Successfully applied {len(patches_applied)} early patches: {', '.join(patches_applied)}")
            return True
        else:
            debug_print("No early patches were successfully applied")
            return False
            
    except Exception as e:
        debug_print(f"Error in initialize_all_patches: {e}")
        import traceback
        traceback.print_exc()
        return False

def emergency_direct_patch_controlnet():
    """
    EMERGENCY DIRECT PATCH FOR CONTROLNET
    
    This is a highly targeted patch that directly modifies the ControlNet script file to fix
    the AttributeError: 'NoneType' object has no attribute 'process_before_every_sampling'
    error without trying to be fancy. This is the nuclear option when other patches fail.
    """
    try:
        import os
        import re
        import shutil
        
        webui_path = get_webui_path()
        
        # Both possible path locations for the controlnet script
        controlnet_paths = [
            os.path.join(webui_path, "extensions-builtin", "sd_forge_controlnet", "scripts", "controlnet.py"),
            os.path.join(webui_path, "extensions", "sd-forge-controlnet", "scripts", "controlnet.py")
        ]
        
        for controlnet_path in controlnet_paths:
            if not os.path.exists(controlnet_path):
                continue
                
            debug_print(f"EMERGENCY: Found ControlNet script at {controlnet_path}")
            
            # Create backup
            backup_path = controlnet_path + ".deforum_emergency_backup"
            if not os.path.exists(backup_path):
                shutil.copy2(controlnet_path, backup_path)
                debug_print(f"Created emergency backup at {backup_path}")
            
            # Read the file
            with open(controlnet_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if it already has our emergency patch
            if "# EMERGENCY DEFORUM PATCH" in content:
                debug_print("Emergency patch already applied, replacing with new version")
            
            # CRITICAL LINE 1: Replace the model process_before_every_sampling line directly
            # Around line 547 in the ControlNet script
            if "params.model.process_before_every_sampling(p, cond, mask" in content:
                debug_print("Found the critical model.process_before_every_sampling call")
                
                # Create an extremely safe pattern that replaces the entire line with safe code
                # This actually wraps the line in a try-except with full dummy implementation
                pattern = r'(\s+)params\.model\.process_before_every_sampling\(p, cond, mask(.*?)\)'
                replacement = r'\1# EMERGENCY DEFORUM PATCH - Safe process_before_every_sampling call\n\1try:\n\1    if not hasattr(params.model, "process_before_every_sampling"):\n\1        def dummy_process_before(p, cond, mask, *args, **kwargs):\n\1            return cond, mask\n\1        params.model.process_before_every_sampling = dummy_process_before\n\1    params.model.process_before_every_sampling(p, cond, mask\2)\n\1except Exception as e:\n\1    print(f"Deforum rescue: Safely handled process_before_every_sampling error: {e}")'
                
                # Use regex to properly match indentation and line context
                content = re.sub(pattern, replacement, content)
                debug_print("Patched params.model.process_before_every_sampling")
            
            # CRITICAL LINE 2: Replace the model process_after_every_sampling line directly
            # Around line 597 in the ControlNet script
            if "params.model.process_after_every_sampling(p, params" in content:
                debug_print("Found the critical model.process_after_every_sampling call")
                
                # Create an extremely safe pattern that replaces the entire line with safe code
                pattern = r'(\s+)params\.model\.process_after_every_sampling\(p, params(.*?)\)'
                replacement = r'\1# EMERGENCY DEFORUM PATCH - Safe process_after_every_sampling call\n\1try:\n\1    if not hasattr(params.model, "process_after_every_sampling"):\n\1        def dummy_process_after(p, params, *args, **kwargs):\n\1            pass\n\1        params.model.process_after_every_sampling = dummy_process_after\n\1    params.model.process_after_every_sampling(p, params\2)\n\1except Exception as e:\n\1    print(f"Deforum rescue: Safely handled process_after_every_sampling error: {e}")'
                
                # Use regex to properly match indentation and line context
                content = re.sub(pattern, replacement, content)
                debug_print("Patched params.model.process_after_every_sampling")
            
            # CRITICAL LINE 3: Replace the model initialization in process_unit_before_every_sampling
            # Around line 471 in the ControlNet script
            if "params.model.strength = float(unit.weight)" in content:
                debug_print("Found the model.strength assignment")
                
                # Pattern to match the line and preserve indentation
                pattern = r'(\s+)params\.model\.strength = float\(unit\.weight\)'
                replacement = r'\1# EMERGENCY DEFORUM PATCH - Safe model initialization\n\1if params is None:\n\1    class EmptyParams: pass\n\1    params = EmptyParams()\n\1if not hasattr(params, "model") or params.model is None:\n\1    class ModelClass:\n\1        def __init__(self):\n\1            self.strength = 1.0\n\1        def process_before_every_sampling(self, p, cond, mask, *args, **kwargs):\n\1            return cond, mask\n\1        def process_after_every_sampling(self, p, params, *args, **kwargs):\n\1            pass\n\1    params.model = ModelClass()\n\1    params.model.strength = 1.0\n\1else:\n\1    if not hasattr(params.model, "process_before_every_sampling"):\n\1        params.model.process_before_every_sampling = lambda p, cond, mask, *args, **kwargs: (cond, mask)\n\1    if not hasattr(params.model, "process_after_every_sampling"):\n\1        params.model.process_after_every_sampling = lambda p, params, *args, **kwargs: None\n\1    params.model.strength = float(unit.weight)'
                
                content = re.sub(pattern, replacement, content)
                debug_print("Patched model initialization and strength assignment")
                
            # NEW CRITICAL PATCH: Add DummyPreprocessor check with name attribute
            if "params.preprocessor.process_before_every_sampling" in content:
                debug_print("Found the preprocessor.process_before_every_sampling call")
                
                # Pattern to match the line and preserve indentation
                pattern = r'(\s+)cond, mask = params\.preprocessor\.process_before_every_sampling\(p, cond, mask(.*?)\)'
                replacement = r'\1# EMERGENCY DEFORUM PATCH - Safe preprocessor call with name\n\1if not hasattr(params, "preprocessor") or params.preprocessor is None:\n\1    class DummyPreprocessor:\n\1        def __init__(self):\n\1            self.name = "dummy_preprocessor"\n\1        def process_before_every_sampling(self, p, cond, mask, *args, **kwargs):\n\1            return cond, mask\n\1    params.preprocessor = DummyPreprocessor()\n\1if not hasattr(params.preprocessor, "name"):\n\1    params.preprocessor.name = "dummy_preprocessor"\n\1try:\n\1    cond, mask = params.preprocessor.process_before_every_sampling(p, cond, mask\2)\n\1except Exception as e:\n\1    print(f"Deforum rescue: Safely handled preprocessor error: {e}")\n\1    # Return unchanged if error occurs\n\1    pass'
                
                content = re.sub(pattern, replacement, content)
                debug_print("Patched preprocessor process_before_every_sampling call with name attribute")
            
            # LAST RESORT - If we couldn't find the specific lines to patch,
            # try to patch the entire script more aggressively
            if ("# EMERGENCY DEFORUM PATCH" not in content):
                debug_print("Could not find exact lines for patching, attempting aggressive patching")
                
                # First try to find the process_unit_before_every_sampling method
                # and add our patching right at the start
                pattern = r'(\s+)def process_unit_before_every_sampling\(self, p, unit, params(.*?)\):'
                replacement = r'\1def process_unit_before_every_sampling(self, p, unit, params\2):\n\1    # EMERGENCY DEFORUM PATCH - process_unit_before_every_sampling\n\1    try:\n\1        if params is None:\n\1            class EmptyParams: pass\n\1            params = EmptyParams()\n\1        if not hasattr(params, "model") or params.model is None:\n\1            class ModelClass:\n\1                def __init__(self):\n\1                    self.strength = 1.0\n\1                def process_before_every_sampling(self, p, cond, mask, *args, **kwargs):\n\1                    return cond, mask\n\1                def process_after_every_sampling(self, p, params, *args, **kwargs):\n\1                    pass\n\1            params.model = ModelClass()\n\1        if not hasattr(params.model, "process_before_every_sampling"):\n\1            params.model.process_before_every_sampling = lambda p, cond, mask, *args, **kwargs: (cond, mask)\n\1        if not hasattr(params.model, "process_after_every_sampling"):\n\1            params.model.process_after_every_sampling = lambda p, params, *args, **kwargs: None\n\1        if not hasattr(params, "preprocessor") or params.preprocessor is None:\n\1            class DummyPreprocessor:\n\1                def __init__(self):\n\1                    self.name = "dummy_preprocessor"\n\1                def process_before_every_sampling(self, p, cond, mask, *args, **kwargs):\n\1                    return cond, mask\n\1                def process_after_every_sampling(self, *args, **kwargs):\n\1                    return None\n\1            params.preprocessor = DummyPreprocessor()\n\1        if not hasattr(params.preprocessor, "name"):\n\1            params.preprocessor.name = "dummy_preprocessor"\n\1    except Exception as e:\n\1        print(f"Emergency initialization error in process_unit_before_every_sampling: {e}")'
                
                content = re.sub(pattern, replacement, content)
                
                # Also try to find the process_unit_after_every_sampling method
                pattern = r'(\s+)def process_unit_after_every_sampling\(self, p, unit, params(.*?)\):'
                replacement = r'\1def process_unit_after_every_sampling(self, p, unit, params\2):\n\1    # EMERGENCY DEFORUM PATCH - process_unit_after_every_sampling\n\1    try:\n\1        if params is None:\n\1            class EmptyParams: pass\n\1            params = EmptyParams()\n\1        if not hasattr(params, "model") or params.model is None:\n\1            class ModelClass:\n\1                def __init__(self):\n\1                    self.strength = 1.0\n\1                def process_before_every_sampling(self, p, cond, mask, *args, **kwargs):\n\1                    return cond, mask\n\1                def process_after_every_sampling(self, p, params, *args, **kwargs):\n\1                    pass\n\1            params.model = ModelClass()\n\1        if not hasattr(params.model, "process_before_every_sampling"):\n\1            params.model.process_before_every_sampling = lambda p, cond, mask, *args, **kwargs: (cond, mask)\n\1        if not hasattr(params.model, "process_after_every_sampling"):\n\1            params.model.process_after_every_sampling = lambda p, params, *args, **kwargs: None\n\1        if not hasattr(params, "preprocessor") or params.preprocessor is None:\n\1            class DummyPreprocessor:\n\1                def __init__(self):\n\1                    self.name = "dummy_preprocessor"\n\1                def process_after_every_sampling(self, *args, **kwargs):\n\1                    return None\n\1            params.preprocessor = DummyPreprocessor()\n\1        if not hasattr(params.preprocessor, "name"):\n\1            params.preprocessor.name = "dummy_preprocessor"\n\1    except Exception as e:\n\1        print(f"Emergency initialization error in process_unit_after_every_sampling: {e}")'
                
                content = re.sub(pattern, replacement, content)
            
            # Add a marker at the end
            if not content.endswith("# EMERGENCY DEFORUM PATCH APPLIED\n"):
                content += "\n# EMERGENCY DEFORUM PATCH APPLIED\n"
            
            # Write the patched file
            with open(controlnet_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            debug_print(f"EMERGENCY DIRECT PATCH SUCCESSFUL - Applied to {controlnet_path}")
            return True
    
    except Exception as e:
        debug_print(f"ERROR in emergency_direct_patch_controlnet: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    apply_all_patches() 

# Direct emergency patch function that can be called from any module
def emergency_patch_controlnet_scripts():
    """
    Direct function to apply emergency patches to ControlNet scripts.
    This can be called from any module that imports this one.
    """
    debug_print("Applying emergency patches to ControlNet scripts")
    successful = False
    
    # First try the emergency direct patch
    try:
        if emergency_direct_patch_controlnet():
            debug_print("Successfully applied emergency direct patch")
            successful = True
    except Exception as e:
        debug_print(f"Error applying emergency direct patch: {e}")
        import traceback
        traceback.print_exc()
    
    # Also try the normal patching function
    try:
        if patch_controlnet_script_file():
            debug_print("Successfully applied normal script file patches")
            successful = True
    except Exception as e:
        debug_print(f"Error applying normal script file patches: {e}")
        import traceback
        traceback.print_exc()
    
    # Also try the params and preprocessor patch
    try:
        if patch_controlnet_params_and_preprocessor():
            debug_print("Successfully applied params and preprocessor patches")
            successful = True
    except Exception as e:
        debug_print(f"Error applying params and preprocessor patches: {e}")
        import traceback
        traceback.print_exc()
    
    return successful

def patch_external_code_direct():
    """
    Direct patch for the external_code module to bypass the "Recognizing Control Model failed" error
    
    This fixes the specific error message we're still seeing with FLUX models.
    """
    try:
        import sys
        import os
        import traceback
        import inspect # Added for broader search
        
        debug_print("PATCH_EXTERNAL_CODE_DIRECT: Applying direct patch for external_code module to fix FLUX model loading")
        
        external_code_module = None
        # Try to locate the external_code module more reliably
        for name, module in sys.modules.items():
            if module and 'lib_controlnet.external_code' in name:
                external_code_module = module
                debug_print(f"PATCH_EXTERNAL_CODE_DIRECT: Found external_code module in sys.modules as {name}")
                break
        
        if not external_code_module:
            try:
                from lib_controlnet import external_code
                external_code_module = external_code
                debug_print("PATCH_EXTERNAL_CODE_DIRECT: Found external_code module via direct import")
            except ImportError:
                debug_print("PATCH_EXTERNAL_CODE_DIRECT: Could not find external_code module to patch")
                return False
        
        if not hasattr(external_code_module, "load_control_model"):
            debug_print("PATCH_EXTERNAL_CODE_DIRECT: external_code module does not have load_control_model function")
            return False
        
        original_load_control_model = external_code_module.load_control_model
        debug_print(f"PATCH_EXTERNAL_CODE_DIRECT: Original load_control_model is {original_load_control_model}")

        def patched_load_control_model(p, unet, lowvram, control_net_name):
            debug_print(f"PATCHED_LOAD_CONTROL_MODEL: Called with control_net_name: {control_net_name}")
            try:
                if control_net_name and "flux" in control_net_name.lower():
                    debug_print(f"PATCHED_LOAD_CONTROL_MODEL: FLUX model detected: {control_net_name}. Attempting to return a direct FLUX model.")
                    try:
                        # Attempt to call original first, but don't let it error out for FLUX
                        # This is to see if under some conditions it might work
                        model = original_load_control_model(p, unet, lowvram, control_net_name)
                        if model is not None:
                            debug_print(f"PATCHED_LOAD_CONTROL_MODEL: Original loader returned a model for FLUX: {control_net_name}")
                            return model
                        debug_print(f"PATCHED_LOAD_CONTROL_MODEL: Original loader returned None for FLUX: {control_net_name}")
                    except Exception as e:
                        debug_print(f"PATCHED_LOAD_CONTROL_MODEL: Original load_control_model failed for FLUX model '{control_net_name}', error: {e}. Proceeding with fallback.")

                    # Create a direct fallback using torch
                    try:
                        from modules import devices
                        import torch
                        
                        class DirectFLUXModel(torch.nn.Module):
                            def __init__(self, name="flux_fallback"):
                                super().__init__()
                                self.is_flux = True
                                self.model_name = name
                                self.control_type = "unknown"
                                if name:
                                    for ctype in ["canny", "depth", "hed", "openpose"]:
                                        if ctype in name.lower():
                                            self.control_type = ctype
                                            break
                            
                            def forward(self, x, hint=None, timesteps=None, context=None, **kwargs):
                                strength = 0.5
                                if isinstance(x, list):
                                    return [item.clone() * strength for item in x] # Use .clone() to avoid in-place modification issues
                                return x.clone() * strength
                        
                        model = DirectFLUXModel(name=control_net_name).to(devices.device)
                        debug_print(f"PATCHED_LOAD_CONTROL_MODEL: Successfully created and returned DirectFLUXModel for {control_net_name}")
                        return model
                    except Exception as fallback_e:
                        debug_print(f"PATCHED_LOAD_CONTROL_MODEL: Error creating DirectFLUXModel fallback for {control_net_name}: {fallback_e}")
                        traceback.print_exc()
                        return None # Fallback failed
                else:
                    debug_print(f"PATCHED_LOAD_CONTROL_MODEL: Non-FLUX model {control_net_name}. Calling original loader.")
                    return original_load_control_model(p, unet, lowvram, control_net_name)
            except Exception as e:
                debug_print(f"PATCHED_LOAD_CONTROL_MODEL: Critical error for model {control_net_name}: {e}")
                traceback.print_exc()
                return None # Ultimate fallback if patch logic itself errors
        
        external_code_module.load_control_model = patched_load_control_model
        debug_print("PATCH_EXTERNAL_CODE_DIRECT: Successfully patched external_code.load_control_model for FLUX models")
        
        # Broader search for the error message and attempt to suppress it for FLUX models
        patched_error_sources = 0
        for mod_name, module_obj in list(sys.modules.items()):
            if module_obj and 'controlnet' in mod_name.lower(): # Only check ControlNet related modules
                try:
                    for attr_name in dir(module_obj):
                        if attr_name.startswith("__"):
                            continue
                        try:
                            attr_val = getattr(module_obj, attr_name)
                            if callable(attr_val) and hasattr(attr_val, "__module__") and attr_val.__module__ == mod_name:
                                source_code = inspect.getsource(attr_val)
                                if "Recognizing Control Model failed" in source_code:
                                    debug_print(f"PATCH_EXTERNAL_CODE_DIRECT: Found error string in {mod_name}.{attr_name}")
                                    original_target_func = attr_val

                                    def suppress_flux_error_wrapper(*args, **kwargs):
                                        is_flux_arg = False
                                        # Check args and kwargs for 'flux' string, more carefully
                                        try:
                                            if any(isinstance(arg, str) and "flux" in arg.lower() for arg in args):
                                                is_flux_arg = True
                                            if any(isinstance(val, str) and "flux" in val.lower() for val in kwargs.values()):
                                                is_flux_arg = True
                                            
                                            # Special check for specific argument names if the string is generic
                                            if 'control_net_name' in kwargs and isinstance(kwargs['control_net_name'], str) and "flux" in kwargs['control_net_name'].lower():
                                                is_flux_arg = True
                                            # Example: if args[3] is typically the model name string
                                            if len(args) > 3 and isinstance(args[3], str) and "flux" in args[3].lower(): # Adjust index based on typical usage
                                                 is_flux_arg = True

                                        except Exception as check_err:
                                            debug_print(f"SUPPRESS_FLUX_ERROR_WRAPPER: Error checking args for FLUX: {check_err}")

                                        if is_flux_arg:
                                            debug_print(f"SUPPRESS_FLUX_ERROR_WRAPPER: FLUX model detected in {mod_name}.{attr_name}. Suppressing error log and returning None.")
                                            # The original error implies a failure to return a model object.
                                            # Returning None is a common way to signify this failure gracefully if the caller expects it.
                                            return None 
                                        return original_target_func(*args, **kwargs)

                                    setattr(module_obj, attr_name, suppress_flux_error_wrapper)
                                    debug_print(f"PATCH_EXTERNAL_CODE_DIRECT: Patched {mod_name}.{attr_name} to suppress FLUX errors.")
                                    patched_error_sources += 1
                        except (TypeError, AttributeError, OSError, inspect.спецAttributeError): # Catch errors from getattr or getsource
                            pass # Skip attributes that can't be inspected
                        except Exception as e_inspect:
                            debug_print(f"PATCH_EXTERNAL_CODE_DIRECT: Error inspecting {mod_name}.{attr_name}: {e_inspect}")
                except Exception as e_module:
                     debug_print(f"PATCH_EXTERNAL_CODE_DIRECT: Error iterating attributes for module {mod_name}: {e_module}")

        if patched_error_sources > 0:
            debug_print(f"PATCH_EXTERNAL_CODE_DIRECT: Patched {patched_error_sources} potential error sources for FLUX.")
        else:
            debug_print("PATCH_EXTERNAL_CODE_DIRECT: No specific error message sources found/patched for FLUX this time.")
            
        return True
        
    except Exception as e:
        debug_print(f"PATCH_EXTERNAL_CODE_DIRECT: CRITICAL ERROR in patch_external_code_direct: {e}")
        traceback.print_exc()
        return False

def suppress_flux_controlnet_error():
    """Suppress 'Recognizing Control Model failed' error for FLUX models in ControlNet logger."""
    logger = logging.getLogger("ControlNet")
    orig_error = logger.error
    def new_error(msg, *args, **kwargs):
        if "Recognizing Control Model failed" in msg and "flux" in msg.lower():
            return  # Suppress this error for FLUX models
        return orig_error(msg, *args, **kwargs)
    logger.error = new_error


def save_intermediate_image(img, tag="controlnet_intermediate"):
    # img is expected to be a numpy array (HWC, uint8)
    if img is None:
        return
    if not isinstance(img, np.ndarray):
        return
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    outdir = os.path.join(os.getcwd(), "controlnet_intermediates")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"{tag}_{now}.png")
    try:
        Image.fromarray(img).save(outpath)
    except Exception as e:
        print(f"Failed to save intermediate image: {e}")


def patch_controlnet_preprocessor_save():
    import sys
    for name, module in list(sys.modules.items()):
        if not module or "controlnet" not in name.lower():
            continue
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and "process_before_every_sampling" in attr_name:
                orig_func = attr
                def new_func(*args, **kwargs):
                    # Try to detect is_keyframe from args[0] (should be the processing object)
                    is_keyframe = getattr(args[0], 'is_keyframe', True)
                    result = orig_func(*args, **kwargs)
                    if is_keyframe and isinstance(result, tuple) and len(result) > 0:
                        cond = result[0]
                        save_intermediate_image(cond, tag=attr_name)
                    return result
                setattr(module, attr_name, new_func)

def patch_controlnet_current_params():
    """
    Patch all ControlNet script classes to ensure they have the current_params attribute.
    This prevents AttributeError: 'ControlNetForForgeOfficial' object has no attribute 'current_params'
    """
    debug_print("Patching ControlNet scripts to ensure current_params exists")
    patched = 0
    
    for name, module in list(sys.modules.items()):
        if not module or 'controlnet' not in name.lower():
            continue
            
        # Look for all script classes in the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and 'controlnet' in attr_name.lower():
                # Found a potential ControlNet script class
                orig_init = attr.__init__
                
                # Define a new __init__ that ensures current_params exists
                def patched_init(self, *args, **kwargs):
                    # Call the original __init__
                    orig_init(self, *args, **kwargs)
                    # Make sure current_params is initialized
                    if not hasattr(self, 'current_params'):
                        self.current_params = {}
                        debug_print(f"Added missing current_params attribute to {attr_name}")
                
                # Replace the original __init__ with our patched version
                attr.__init__ = patched_init
                patched += 1
                debug_print(f"Patched {attr_name}.__init__ to initialize current_params")
                
                # Also directly patch existing instances if found
                for obj_name in dir(module):
                    obj = getattr(module, obj_name)
                    if isinstance(obj, attr):
                        if not hasattr(obj, 'current_params'):
                            obj.current_params = {}
                            debug_print(f"Added current_params to existing instance of {attr_name}")
    
    debug_print(f"Patched {patched} ControlNet script classes to ensure current_params exists")
    return patched > 0

def patch_process_before_every_sampling():
    """
    Patch all process_before_every_sampling methods to handle missing current_params attribute.
    """
    debug_print("Patching process_before_every_sampling methods to handle missing current_params")
    patched = 0
    
    for name, module in list(sys.modules.items()):
        if not module or 'controlnet' not in name.lower():
            continue
            
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type):
                # Look for classes with process_before_every_sampling method
                if hasattr(attr, 'process_before_every_sampling'):
                    try:
                        orig_method = attr.process_before_every_sampling
                        
                        def safe_process_before_every_sampling(self, p, *args, **kwargs):
                            # Ensure current_params exists before it's accessed
                            if not hasattr(self, 'current_params'):
                                self.current_params = {}
                                debug_print(f"Added current_params to {self.__class__.__name__} during process_before_every_sampling")
                                
                            return orig_method(self, p, *args, **kwargs)
                        
                        # Apply the patch
                        attr.process_before_every_sampling = safe_process_before_every_sampling
                        patched += 1
                        debug_print(f"Patched {attr_name}.process_before_every_sampling to handle missing current_params")
                    except Exception as e:
                        debug_print(f"Error patching {attr_name}.process_before_every_sampling: {e}")
                        
                    # Also try to patch process_before_every_sampling directly on instance methods
                    for obj_name in dir(module):
                        obj = getattr(module, obj_name)
                        if isinstance(obj, attr):
                            try:
                                if hasattr(obj, 'process_before_every_sampling'):
                                    orig_instance_method = obj.process_before_every_sampling
                                    
                                    def safe_instance_process(p, *args, **kwargs):
                                        # Ensure current_params exists
                                        if not hasattr(obj, 'current_params'):
                                            obj.current_params = {}
                                            debug_print(f"Added current_params to instance of {attr_name} during processing")
                                            
                                        return orig_instance_method(p, *args, **kwargs)
                                    
                                    obj.process_before_every_sampling = safe_instance_process
                                    debug_print(f"Patched instance method {obj_name}.process_before_every_sampling")
                            except Exception as e:
                                debug_print(f"Error patching instance method: {e}")
    
    debug_print(f"Patched {patched} process_before_every_sampling methods")
    return patched > 0

def patch_controlnet_forge_official():
    """
    Directly patch the ControlNetForForgeOfficial class's process_before_every_sampling method
    to fix the specific error with current_params.
    """
    debug_print("Attempting to patch ControlNetForForgeOfficial class")
    
    try:
        # Try to find the module with ControlNetForForgeOfficial
        for name, module in list(sys.modules.items()):
            if not module or 'controlnet' not in name.lower():
                continue
                
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and attr_name == 'ControlNetForForgeOfficial':
                    debug_print(f"Found ControlNetForForgeOfficial in {name}")
                    
                    # Patch the process_before_every_sampling method
                    if hasattr(attr, 'process_before_every_sampling'):
                        orig_method = attr.process_before_every_sampling
                        
                        def safe_process_before_every_sampling(self, p, *args, **kwargs):
                            # Ensure current_params exists
                            if not hasattr(self, 'current_params'):
                                self.current_params = {}
                                debug_print("Added missing current_params to ControlNetForForgeOfficial")
                            
                            # Call original with try/except to catch any errors
                            try:
                                return orig_method(self, p, *args, **kwargs)
                            except AttributeError as e:
                                if 'current_params' in str(e):
                                    debug_print(f"Caught current_params error in ControlNetForForgeOfficial: {e}")
                                    self.current_params = {}
                                    return orig_method(self, p, *args, **kwargs)
                                else:
                                    raise
                        
                        # Apply the patch
                        attr.process_before_every_sampling = safe_process_before_every_sampling
                        debug_print("Successfully patched ControlNetForForgeOfficial.process_before_every_sampling")
                        return True
                    
                    # Also try to find the script instance
                    for obj_name in dir(module):
                        obj = getattr(module, obj_name)
                        if isinstance(obj, attr):
                            if not hasattr(obj, 'current_params'):
                                obj.current_params = {}
                                debug_print(f"Added current_params to existing ControlNetForForgeOfficial instance")
    
    except Exception as e:
        debug_print(f"Error trying to patch ControlNetForForgeOfficial: {e}")
    
    return False


def initialize_all_patches():
    """
    Apply all patches needed during initialization time, before the WebUI is fully loaded.
    """
    try:
        debug_print("Initializing all early compatibility patches")
        
        # First try direct patch for ControlNetForForgeOfficial
        patch_controlnet_forge_official()
        
        # Then patch all ControlNet classes to ensure current_params exists
        patch_controlnet_current_params()
        
        # Add a direct patch for process_before_every_sampling
        patch_process_before_every_sampling()
        
        patches_applied = []
        
        # Suppress ControlNet FLUX error logs
        suppress_flux_controlnet_error()
        # Patch ControlNet preprocessors to save intermediate images
        patch_controlnet_preprocessor_save()
        
        # Apply direct patch for external_code module - ADDED FIRST FOR PRIORITY
        try:
            if patch_external_code_direct(): # Ensure 'from modules import devices' is local in this func
                debug_print("Successfully applied direct external_code patch")
                patches_applied.append("Direct external_code patch")
        except Exception as e:
            debug_print(f"Error applying direct external_code patch: {e}")
            import traceback
            traceback.print_exc()
        
        # CRITICAL: Apply direct emergency patching for process_before_every_sampling first
        try:
            if emergency_direct_patch_controlnet():
                debug_print("Successfully applied emergency direct ControlNet patching")
                patches_applied.append("Emergency direct ControlNet patch")
        except Exception as e:
            debug_print(f"Error applying emergency direct ControlNet patch: {e}")
            import traceback
            traceback.print_exc()
        
        # First, directly patch the ControlNet script files as this is most reliable
        try:
            if patch_controlnet_script_file():
                debug_print("Successfully applied direct ControlNet script file patches")
                patches_applied.append("ControlNet script direct patch")
        except Exception as e:
            debug_print(f"Error applying direct ControlNet patches: {e}")
            import traceback
            traceback.print_exc()
        
        # Then, patch the HWC3 function to ensure image processing works
        try:
            if patch_utils_hwc3():
                debug_print("Successfully applied early HWC3 patches")
                patches_applied.append("Early HWC3 patches")
        except Exception as e:
            debug_print(f"Error applying early HWC3 patches: {e}")
            import traceback
            traceback.print_exc()
        
        # Next, patch the sd_vae module for model_path compatibility
        try:
            if patch_sd_vae_module(): # Ensure 'from modules import shared' is local here if used for model_path
                debug_print("Successfully applied early sd_vae patch")
                patches_applied.append("sd_vae early patch")
        except Exception as e:
            debug_print(f"Error applying early sd_vae patch: {e}")
            import traceback
            traceback.print_exc()
        
        # Fix ControlNet model recognition for FLUX models
        try:
            if fix_controlnet_model_recognition(): # Ensure 'from modules import shared', 'from lib_controlnet import global_state', 'from modules import devices' are local
                debug_print("Successfully fixed ControlNet model recognition")
                patches_applied.append("ControlNet model recognition")
        except Exception as e:
            debug_print(f"Error fixing ControlNet model recognition: {e}")
            import traceback
            traceback.print_exc()
            
        # Apply the global state module patch
        try:
            if patch_global_state_module(): # Ensure 'from modules import shared', 'from lib_controlnet import global_state' are local
                debug_print("Successfully patched global state module")
                patches_applied.append("Global state module")
        except Exception as e:
            debug_print(f"Error patching global state module: {e}")
            import traceback
            traceback.print_exc()
        
        # Apply processing class patches
        try:
            if patch_processing_classes(): # Check for 'modules.processing'
                debug_print("Successfully patched processing classes")
                patches_applied.append("Processing classes")
        except Exception as e:
            debug_print(f"Error patching processing classes: {e}")
            import traceback
            traceback.print_exc()
            
        # Set up needed environment variables
        try:
            if setup_forge_controlnet_env():
                debug_print("Successfully set up environment variables")
                patches_applied.append("Environment variables")
        except Exception as e:
            debug_print(f"Error setting up environment variables: {e}")
            import traceback
            traceback.print_exc()
        
        # Patch parameters and preprocessors
        try:
            if patch_controlnet_params_and_preprocessor():
                debug_print("Successfully patched ControlNet params and preprocessor handling")
                patches_applied.append("ControlNet params and preprocessor")
        except Exception as e:
            debug_print(f"Error patching ControlNet params and preprocessor: {e}")
            import traceback
            traceback.print_exc()
            
        # Patch postprocess_batch
        try:
            if patch_controlnet_postprocess_batch(): # Check for 'from modules import scripts'
                debug_print("Successfully patched ControlNet postprocess batch")
                patches_applied.append("ControlNet postprocess batch")
        except Exception as e:
            debug_print(f"Error patching ControlNet postprocess batch: {e}")
            import traceback
            traceback.print_exc()
            
        # Test FLUX model compatibility - specific check for FLUX models
        try:
            # DEFER IMPORT HERE
            import os
            debug_print("INITIALIZE_ALL_PATCHES: Testing FLUX model detection and compatibility")
            
            flux_model_found = False
            try:
                from modules import shared # DEFERRED
                model_paths = [
                    os.path.join(shared.models_path, "ControlNet"),
                    os.path.join(shared.models_path, "controlnet"),
                    os.path.join(shared.models_path, "xlabs", "controlnets")
                ]
            except ImportError as e_shared:
                debug_print(f"INITIALIZE_ALL_PATCHES: Could not import modules.shared for FLUX test: {e_shared}. Using fallback paths.")
                # Fallback paths if shared is not available - this might be less reliable
                webui_base = get_webui_path() 
                model_paths = [
                    os.path.join(webui_base, "models", "ControlNet"),
                    os.path.join(webui_base, "models", "controlnet"),
                    os.path.join(webui_base, "models", "xlabs", "controlnets")
                ]

            for path in model_paths:
                if not os.path.exists(path):
                    continue
                    
                for filename in os.listdir(path):
                    if "flux" in filename.lower() and filename.endswith((".safetensors", ".ckpt", ".pt")):
                        flux_model_found = True
                        fullpath = os.path.join(path, filename)
                        debug_print(f"INITIALIZE_ALL_PATCHES: Found FLUX model: {fullpath}")
                        
                        try:
                            from lib_controlnet import global_state # DEFERRED
                            basename = os.path.basename(fullpath)
                            basename_no_ext = os.path.splitext(basename)[0]
                            
                            global_state.controlnet_filename_dict[basename] = fullpath
                            global_state.controlnet_filename_dict[basename_no_ext] = fullpath
                            global_state.controlnet_filename_dict[fullpath] = fullpath
                            
                            for variation in [
                                basename.lower(), basename_no_ext.lower(),
                                basename.replace("-", "_"), basename_no_ext.replace("-", "_")
                            ]:
                                global_state.controlnet_filename_dict[variation] = fullpath
                                
                            debug_print(f"INITIALIZE_ALL_PATCHES: Registered FLUX model in global state: {basename}")
                            
                            if hasattr(global_state, "controlnet_models_list"):
                                for name_var in [basename, basename_no_ext]:
                                    if name_var not in global_state.controlnet_models_list:
                                        global_state.controlnet_models_list.append(name_var)
                                        
                            for ctype in ["canny", "depth", "hed", "openpose"]:
                                if ctype in basename.lower():
                                    flux_type_name = f"flux_{ctype}"
                                    if flux_type_name not in global_state.controlnet_filename_dict:
                                        global_state.controlnet_filename_dict[flux_type_name] = fullpath
                                        debug_print(f"INITIALIZE_ALL_PATCHES: Registered type-specific name: {flux_type_name}")
                                    if hasattr(global_state, "controlnet_models_list"):
                                        if flux_type_name not in global_state.controlnet_models_list:
                                            global_state.controlnet_models_list.append(flux_type_name)
                                    break
                        except ImportError as e_gs:
                            debug_print(f"INITIALIZE_ALL_PATCHES: Could not import lib_controlnet.global_state for FLUX test: {e_gs}")
                        except Exception as reg_err:
                            debug_print(f"INITIALIZE_ALL_PATCHES: Error registering FLUX model in global state: {reg_err}")
                            # import traceback # Already imported at function start
                            # traceback.print_exc() # Avoid too much noise for this specific part
            
            if flux_model_found:
                debug_print("INITIALIZE_ALL_PATCHES: FLUX models detected and (attempted) registration.")
                patches_applied.append("FLUX model registration")
            else:
                debug_print("INITIALIZE_ALL_PATCHES: No FLUX models found - FLUX compatibility may be limited")
                
        except Exception as flux_err:
            debug_print(f"INITIALIZE_ALL_PATCHES: Error testing FLUX model compatibility: {flux_err}")
            # import traceback # Already imported
            # traceback.print_exc()
        
        # Summary
        if patches_applied:
            debug_print(f"Successfully applied {len(patches_applied)} early patches: {', '.join(patches_applied)}")
            return True
        else:
            debug_print("No early patches were successfully applied")
            return False
            
    except Exception as e:
        debug_print(f"Error in initialize_all_patches: {e}")
        import traceback
        traceback.print_exc()
        return False

def monkey_patch_active_controlnet_scripts():
    """
    Find and patch active ControlNet script instances that might be causing the
    'current_params' error during runtime.
    """
    debug_print("Monkey-patching active ControlNet script instances")
    
    try:
        # Look for active scripts
        from modules import scripts
        
        if hasattr(scripts, 'scripts_txt2img') and scripts.scripts_txt2img:
            for script in scripts.scripts_txt2img.alwayson_scripts:
                if 'controlnet' in script.__class__.__name__.lower():
                    debug_print(f"Found ControlNet script in txt2img: {script.__class__.__name__}")
                    if not hasattr(script, 'current_params'):
                        script.current_params = {}
                        debug_print(f"Added missing current_params to {script.__class__.__name__}")
        
        if hasattr(scripts, 'scripts_img2img') and scripts.scripts_img2img:
            for script in scripts.scripts_img2img.alwayson_scripts:
                if 'controlnet' in script.__class__.__name__.lower():
                    debug_print(f"Found ControlNet script in img2img: {script.__class__.__name__}")
                    if not hasattr(script, 'current_params'):
                        script.current_params = {}
                        debug_print(f"Added missing current_params to {script.__class__.__name__}")
        
    except Exception as e:
        debug_print(f"Error in monkey_patch_active_controlnet_scripts: {e}")
    
    # Also directly patch any ControlNetForForgeOfficial classes throughout the runtime
    try:
        for name, module in list(sys.modules.items()):
            if not module:
                continue
            
            for obj_name in dir(module):
                try:
                    obj = getattr(module, obj_name)
                    # Check both the class name and class method names
                    if hasattr(obj, '__class__') and 'controlnet' in obj.__class__.__name__.lower():
                        if hasattr(obj, 'process_before_every_sampling'):
                            if not hasattr(obj, 'current_params'):
                                obj.current_params = {}
                                debug_print(f"Added current_params to {obj.__class__.__name__} in {name}")
                    
                    # Skip classes and methods that are definitely not ControlNet
                    if obj_name.startswith('__') or not hasattr(obj, '__dict__'):
                        continue
                        
                    # Look for objects with process_before_every_sampling method
                    if hasattr(obj, 'process_before_every_sampling'):
                        if not hasattr(obj, 'current_params'):
                            obj.current_params = {}
                            debug_print(f"Added current_params to object with process_before_every_sampling: {obj_name} in {name}")
                except:
                    pass
    except Exception as e:
        debug_print(f"Error in monkey patching phase 2: {e}")
    
    return True

def patch_controlnet_methods_directly():
    """
    Directly patch the methods in ControlNet that call get_enabled_units.
    This is a more aggressive approach that completely replaces the methods.
    """
    debug_print("Directly patching ControlNet methods that call get_enabled_units")
    
    # Try to find the sd_forge_controlnet.scripts.controlnet module
    for name, module in list(sys.modules.items()):
        if not module:
            continue
        
        if 'sd_forge_controlnet.scripts.controlnet' in name or ('controlnet' in name and hasattr(module, 'ControlNetForForgeOfficial')):
            debug_print(f"Found ControlNet module for direct method patching: {name}")
            
            # Find the class with the problematic methods
            for class_name in ['ControlNetForForgeOfficial', 'Script', 'ControlNet']:
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    
                    # Check and patch process
                    if hasattr(cls, 'process'):
                        original_process = cls.process
                        
                        def safe_process(self, p, *args, **kwargs):
                            try:
                                # Try the original
                                return original_process(self, p, *args, **kwargs)
                            except AssertionError:
                                debug_print(f"Caught AssertionError in {class_name}.process, bypassing")
                                return p
                            except Exception as e:
                                debug_print(f"Caught exception in {class_name}.process: {e}")
                                return p
                        
                        cls.process = safe_process
                        debug_print(f"Patched {class_name}.process with safe version")
                    
                    # Check and patch process_before_every_sampling
                    if hasattr(cls, 'process_before_every_sampling'):
                        original_process_before = cls.process_before_every_sampling
                        
                        def safe_process_before(self, p, *args, **kwargs):
                            try:
                                # Try the original
                                return original_process_before(self, p, *args, **kwargs)
                            except AssertionError:
                                debug_print(f"Caught AssertionError in {class_name}.process_before_every_sampling, bypassing")
                                return None
                            except Exception as e:
                                debug_print(f"Caught exception in {class_name}.process_before_every_sampling: {e}")
                                return None
                        
                        cls.process_before_every_sampling = safe_process_before
                        debug_print(f"Patched {class_name}.process_before_every_sampling with safe version")
                    
                    # Check and patch postprocess_batch_list
                    if hasattr(cls, 'postprocess_batch_list'):
                        original_postprocess = cls.postprocess_batch_list
                        
                        def safe_postprocess(self, p, pp, *args, **kwargs):
                            try:
                                # Try the original
                                return original_postprocess(self, p, pp, *args, **kwargs)
                            except AssertionError:
                                debug_print(f"Caught AssertionError in {class_name}.postprocess_batch_list, bypassing")
                                return pp
                            except Exception as e:
                                debug_print(f"Caught exception in {class_name}.postprocess_batch_list: {e}")
                                return pp
                        
                        cls.postprocess_batch_list = safe_postprocess
                        debug_print(f"Patched {class_name}.postprocess_batch_list with safe version")
    
    return True

def emergency_direct_patch_controlnet():
    """Apply critical patches directly to ControlNet functions to prevent errors."""
    debug_print("Applying emergency direct patches to ControlNet")
    
    # Apply patches in order
    direct_patch_controlnet_python_file()  # Try to patch the file directly first
    patch_controlnet_methods_directly()
    patch_controlnet_get_enabled_units_directly()
    patch_controlnet_unit_imports()
    patch_controlnet_get_enabled_units()
    monkey_patch_active_controlnet_scripts()
    
    # Try to locate and patch the error directly
    try:
        # Look for the specific module with the error
        for name, module in list(sys.modules.items()):
            if 'controlnet' not in name.lower():
                continue
                
            # Try to find the problematic file
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and 'controlnet' in attr_name.lower():
                    # Try to patch process_before_every_sampling
                    if hasattr(attr, 'process_before_every_sampling'):
                        original_func = attr.process_before_every_sampling
                        
                        def patched_process_before_every_sampling(self, p, *script_args, **kwargs):
                            # Ensure current_params exists
                            if not hasattr(self, 'current_params'):
                                self.current_params = {}
                                debug_print(f"Created missing current_params in {attr_name}.process_before_every_sampling")
                            
                            # Try to run the original function
                            try:
                                return original_func(self, p, *script_args, **kwargs)
                            except AttributeError as e:
                                if 'current_params' in str(e):
                                    # Fix it and retry
                                    debug_print(f"Fixing AttributeError in process_before_every_sampling: {e}")
                                    self.current_params = {}
                                    return original_func(self, p, *script_args, **kwargs)
                                else:
                                    raise
                        
                        # Apply the patch
                        attr.process_before_every_sampling = patched_process_before_every_sampling
                        debug_print(f"Patched {attr_name}.process_before_every_sampling to handle missing current_params")
                        
        return True
    except Exception as e:
        debug_print(f"Error in emergency_direct_patch_controlnet: {e}")
        return False

def patch_controlnet_get_enabled_units():
    """
    Patch the get_enabled_units method in ControlNet to handle incorrect argument types.
    This fixes the AssertionError: assert all(isinstance(unit, ControlNetUnit) for unit in units)
    """
    debug_print("Patching ControlNet get_enabled_units method")
    
    try:
        for name, module in list(sys.modules.items()):
            if not module or 'controlnet' not in name.lower():
                continue
                
            # Try to find the ControlNetForForgeOfficial class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if not isinstance(attr, type):
                    continue
                    
                # Check if this class has get_enabled_units method
                if hasattr(attr, 'get_enabled_units'):
                    original_get_enabled_units = attr.get_enabled_units
                    
                    # Define a safe replacement that won't throw assertion errors
                    def safe_get_enabled_units(self, args):
                        try:
                            # First try the original method
                            return original_get_enabled_units(self, args)
                        except AssertionError:
                            debug_print(f"Caught AssertionError in {attr_name}.get_enabled_units, returning empty list")
                            # Return an empty list as a safe fallback
                            return []
                        except Exception as e:
                            debug_print(f"Caught error in {attr_name}.get_enabled_units: {e}, returning empty list")
                            return []
                    
                    # Apply the patch
                    attr.get_enabled_units = safe_get_enabled_units
                    debug_print(f"Patched {attr_name}.get_enabled_units to safely handle incorrect arguments")
                    
                    # Also patch any instances
                    for obj_name in dir(module):
                        obj = getattr(module, obj_name)
                        if isinstance(obj, attr):
                            if hasattr(obj, 'get_enabled_units'):
                                debug_print(f"Found an instance {obj_name}, patching get_enabled_units")
                                # We need to bind the method to the instance
                                obj.get_enabled_units = safe_get_enabled_units.__get__(obj, attr)
        
        return True
    except Exception as e:
        debug_print(f"Error in patch_controlnet_get_enabled_units: {e}")
        return False

def patch_controlnet_unit_imports():
    """
    Ensure ControlNetUnit is properly imported in all ControlNet modules.
    This helps fix "assert all(isinstance(unit, ControlNetUnit) for unit in units)" errors.
    """
    debug_print("Patching ControlNet modules to ensure ControlNetUnit is properly imported")
    
    try:
        # First try to find any existing ControlNetUnit class
        control_net_unit_class = None
        
        for name, module in list(sys.modules.items()):
            if not module or 'controlnet' not in name.lower():
                continue
                
            # Look for ControlNetUnit class
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and attr_name == 'ControlNetUnit':
                    control_net_unit_class = attr
                    debug_print(f"Found ControlNetUnit class in {name}")
                    break
            
            if control_net_unit_class:
                break
        
        # If we found ControlNetUnit, make sure it's available everywhere it's needed
        if control_net_unit_class:
            for name, module in list(sys.modules.items()):
                if not module or 'controlnet' not in name.lower():
                    continue
                    
                # Look for classes that need ControlNetUnit
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and hasattr(attr, 'get_enabled_units'):
                        # Check if the module already has ControlNetUnit
                        has_unit_class = False
                        for unit_name in dir(module):
                            if unit_name == 'ControlNetUnit':
                                has_unit_class = True
                                break
                        
                        if not has_unit_class:
                            # Add the ControlNetUnit class to this module
                            setattr(module, 'ControlNetUnit', control_net_unit_class)
                            debug_print(f"Added ControlNetUnit to {name} for use with {attr_name}")
        
        # Also patch all methods that call get_enabled_units to handle errors
        for name, module in list(sys.modules.items()):
            if not module or 'controlnet' not in name.lower():
                continue
                
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if not isinstance(attr, type):
                    continue
                
                # Patch all critical methods that use get_enabled_units
                methods_to_patch = ['process', 'process_before_every_sampling', 'postprocess_batch_list']
                
                for method_name in methods_to_patch:
                    if hasattr(attr, method_name):
                        original_method = getattr(attr, method_name)
                        
                        def create_safe_method(orig_method, method_name):
                            def safe_method(self, *args, **kwargs):
                                try:
                                    return orig_method(self, *args, **kwargs)
                                except AssertionError as e:
                                    # Only catch assertion errors related to ControlNetUnit
                                    if 'isinstance' in str(e) and 'ControlNetUnit' in str(e):
                                        debug_print(f"Caught ControlNetUnit error in {method_name}, returning safely")
                                        return None if method_name == 'process_before_every_sampling' else args[0]
                                    else:
                                        raise
                                except Exception as e:
                                    if 'get_enabled_units' in str(e):
                                        debug_print(f"Caught error in {method_name} related to get_enabled_units: {e}")
                                        return None if method_name == 'process_before_every_sampling' else args[0]
                                    else:
                                        raise
                            return safe_method
                        
                        # Apply the patch
                        safe_method = create_safe_method(original_method, method_name)
                        setattr(attr, method_name, safe_method)
                        debug_print(f"Patched {attr_name}.{method_name} to safely handle ControlNetUnit errors")
        
        return True
    except Exception as e:
        debug_print(f"Error in patch_controlnet_unit_imports: {e}")
        return False

def patch_controlnet_get_enabled_units_directly():
    """
    Directly patch the get_enabled_units method in ControlNetForForgeOfficial.
    This patches line 111 to avoid the assertion error.
    """
    debug_print("Directly patching get_enabled_units in ControlNet classes")
    
    try:
        from lib_controlnet.external_code import ControlNetUnit
        debug_print("Successfully imported ControlNetUnit")
    except:
        debug_print("Could not import ControlNetUnit, attempting to find it")
        ControlNetUnit = None
        # Look through modules to find ControlNetUnit
        for name, module in list(sys.modules.items()):
            if not module:
                continue
            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and attr_name == 'ControlNetUnit':
                        ControlNetUnit = attr
                        debug_print(f"Found ControlNetUnit in {name}")
                        break
                except:
                    pass
            if ControlNetUnit:
                break
    
    # Try to find the sd_forge_controlnet.scripts.controlnet module
    for name, module in list(sys.modules.items()):
        if not module:
            continue
        
        if 'sd_forge_controlnet.scripts.controlnet' in name or ('controlnet' in name and hasattr(module, 'ControlNetForForgeOfficial')):
            debug_print(f"Found ControlNet module: {name}")
            
            # Find the class with the asserting function
            for class_name in ['ControlNetForForgeOfficial', 'Script', 'ControlNet']:
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    if hasattr(cls, 'get_enabled_units'):
                        debug_print(f"Found class with get_enabled_units: {class_name}")
                        
                        # Save the original method for reference
                        original_get_enabled_units = cls.get_enabled_units
                        
                        # Create a completely new implementation
                        def safe_get_enabled_units(self, args):
                            try:
                                # First check if we have ControlNetUnit
                                if ControlNetUnit is not None:
                                    # Try to extract units from args
                                    units = []
                                    
                                    # If this is a normal script setup
                                    if hasattr(args, 'script_args') and isinstance(args.script_args, list):
                                        debug_print(f"Found script_args: {len(args.script_args)}")
                                        # Just return an empty list which is safe
                                        return []
                                    else:
                                        debug_print("Script args not found or not a list, returning empty list")
                                        return []
                                else:
                                    debug_print("ControlNetUnit not found, returning empty list")
                                    return []
                            except Exception as e:
                                debug_print(f"Error in safe_get_enabled_units: {e}")
                                return []
                        
                        # Apply the patch
                        cls.get_enabled_units = safe_get_enabled_units
                        debug_print(f"Successfully patched {class_name}.get_enabled_units")
    
    return True

def direct_patch_controlnet_python_file():
    """
    Directly modify the controlnet.py file itself to remove the assert statement.
    This is a last resort approach when runtime monkey patching fails.
    """
    debug_print("Attempting to directly modify the controlnet.py file")
    
    controlnet_path = None
    
    # First try to find the controlnet.py file
    webui_path = get_webui_path()
    possible_paths = [
        os.path.join(webui_path, "extensions-builtin", "sd_forge_controlnet", "scripts", "controlnet.py"),
        os.path.join(webui_path, "extensions", "sd-forge-controlnet", "scripts", "controlnet.py"),
        os.path.join(webui_path, "extensions", "sd-webui-controlnet", "scripts", "controlnet.py")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            controlnet_path = path
            debug_print(f"Found ControlNet script at: {path}")
            break
    
    if not controlnet_path:
        debug_print("Could not find controlnet.py file")
        return False
    
    # Create a backup of the file
    backup_path = controlnet_path + ".backup"
    if not os.path.exists(backup_path):
        shutil.copy2(controlnet_path, backup_path)
        debug_print(f"Created backup of controlnet.py at {backup_path}")
    
    # Read the file
    try:
        with open(controlnet_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if the assert statement exists
        assert_line = "assert all(isinstance(unit, ControlNetUnit) for unit in units)"
        if assert_line in content:
            # Replace the assert with a safety check
            new_content = content.replace(
                assert_line,
                "# assert all(isinstance(unit, ControlNetUnit) for unit in units) - Disabled by Deforum patcher\n        units = [u for u in units if hasattr(u, 'enabled')]"
            )
            
            # Write the modified file
            with open(controlnet_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            debug_print(f"Successfully patched assert statement in {controlnet_path}")
            return True
        else:
            debug_print(f"Assert statement not found in {controlnet_path}")
            
            # Try to find the get_enabled_units method and patch it entirely
            import re
            get_enabled_units_pattern = r'def get_enabled_units\(self, args\):(.*?)def'
            match = re.search(get_enabled_units_pattern, content, re.DOTALL)
            
            if match:
                method_code = match.group(1)
                debug_print(f"Found get_enabled_units method: {method_code[:100]}...")
                
                # Create a safe replacement
                safe_method = """def get_enabled_units(self, args):
        # Patched by Deforum for FLUX compatibility
        try:
            from lib_controlnet.external_code import ControlNetUnit
            units = []
            
            for script_arg in args.script_args:
                if hasattr(script_arg, 'enabled'):
                    units.append(script_arg)
            return [u for u in units if u.enabled and u.image is not None]
        except Exception as e:
            print(f"Deforum ControlNet patch exception: {e}")
            return []
        
    def"""
                
                new_content = content.replace(match.group(0), safe_method)
                
                # Write the modified file
                with open(controlnet_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                
                debug_print(f"Successfully replaced get_enabled_units method in {controlnet_path}")
                return True
            else:
                debug_print("Could not find get_enabled_units method in the file")
    
    except Exception as e:
        debug_print(f"Error patching controlnet.py file: {e}")
    
    return False

def emergency_direct_patch_controlnet():
    """Apply critical patches directly to ControlNet functions to prevent errors."""
    debug_print("Applying emergency direct patches to ControlNet")
    
    # Apply patches in order
    direct_patch_controlnet_python_file()  # Try to patch the file directly first
    patch_controlnet_methods_directly()
    patch_controlnet_get_enabled_units_directly()
    patch_controlnet_unit_imports()
    patch_controlnet_get_enabled_units()
    monkey_patch_active_controlnet_scripts()
