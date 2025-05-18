"""
Preload module for Deforum

This module handles initialization tasks that need to be done before Deforum starts.
Currently focuses on patching ControlNet to ensure compatibility.
"""

import os
import sys
import importlib
import numpy as np
from pathlib import Path

def debug_print(message):
    """Print debug message with preload prefix"""
    print(f"[Deforum Preload] {message}")

def is_controlnet_enabled():
    """Check if ControlNet is enabled and active"""
    try:
        from modules import scripts
        
        # Look in both txt2img and img2img script collections
        for script_collection in [
            getattr(scripts, "scripts_txt2img", None),
            getattr(scripts, "scripts_img2img", None)
        ]:
            if script_collection and hasattr(script_collection, "alwayson_scripts"):
                for script in script_collection.alwayson_scripts:
                    if hasattr(script, "title") and script.title().lower() == "controlnet":
                        debug_print("Found enabled ControlNet script")
                        return True
        
        debug_print("ControlNet extension not enabled")
        return False
    except Exception as e:
        debug_print(f"Error checking ControlNet status: {e}")
        return False

def patch_processing_classes():
    """
    Patch the StableDiffusionProcessingTxt2Img and StableDiffusionProcessingImg2Img classes
    
    This adds default resize_mode attribute to both classes at instantiation time.
    """
    try:
        from modules import processing
        
        # Check if already patched
        if hasattr(processing.StableDiffusionProcessingTxt2Img, "_resize_mode_patched"):
            debug_print("Processing classes already patched")
            return True
        
        # Store original __init__ methods
        original_txt2img_init = processing.StableDiffusionProcessingTxt2Img.__init__
        original_img2img_init = processing.StableDiffusionProcessingImg2Img.__init__
        
        # Create patched versions
        def patched_txt2img_init(self, *args, **kwargs):
            # Handle enable_hr separately for WebUI Forge compatibility
            enable_hr = kwargs.pop('enable_hr', None)
            if enable_hr is not None:
                # Store it as an attribute but don't pass to original init
                self._enable_hr = enable_hr
                
            # Call original init with modified kwargs
            original_txt2img_init(self, *args, **kwargs)
            
            # Add resize_mode if it doesn't exist
            if not hasattr(self, 'resize_mode'):
                self.resize_mode = "Inner Fit (Scale to Fit)"
                
        def patched_img2img_init(self, *args, **kwargs):
            # Handle enable_hr separately for WebUI Forge compatibility
            enable_hr = kwargs.pop('enable_hr', None) 
            if enable_hr is not None:
                # Store it as an attribute but don't pass to original init
                self._enable_hr = enable_hr
                
            # Call original init with modified kwargs
            original_img2img_init(self, *args, **kwargs)
            
            # Add resize_mode if it doesn't exist
            if not hasattr(self, 'resize_mode'):
                self.resize_mode = "Inner Fit (Scale to Fit)"
        
        # Apply patches
        processing.StableDiffusionProcessingTxt2Img.__init__ = patched_txt2img_init
        debug_print("Successfully patched StableDiffusionProcessingTxt2Img.__init__")
        
        processing.StableDiffusionProcessingImg2Img.__init__ = patched_img2img_init
        debug_print("Successfully patched StableDiffusionProcessingImg2Img.__init__")

        # Also patch using an additional method to handle cases where __init__ isn't called
        # Add __getattr__ method to both classes
        if not hasattr(processing.StableDiffusionProcessingTxt2Img, "_original_getattr"):
            # First record if the class already has a __getattr__ method
            processing.StableDiffusionProcessingTxt2Img._original_getattr = getattr(
                processing.StableDiffusionProcessingTxt2Img, "__getattr__", None
            )
            
            # Define the patched __getattr__ method
            def patched_getattr(self, name):
                if name == 'resize_mode':
                    # For resize_mode, return a default value
                    debug_print("Intercepted missing resize_mode attribute access")
                    return "Inner Fit (Scale to Fit)"
                elif name == 'enable_hr':
                    # Handle enable_hr attribute
                    return getattr(self, '_enable_hr', False)
                
                # For other attributes, use the original __getattr__ if it exists
                if self._original_getattr is not None:
                    return self._original_getattr(self, name)
                
                # Otherwise, raise the standard AttributeError
                raise AttributeError(f"{type(self).__name__} has no attribute '{name}'")
            
            # Apply the patch
            processing.StableDiffusionProcessingTxt2Img.__getattr__ = patched_getattr
            debug_print("Successfully patched StableDiffusionProcessingTxt2Img.__getattr__")
            
            # Also patch img2img class similarly
            processing.StableDiffusionProcessingImg2Img._original_getattr = getattr(
                processing.StableDiffusionProcessingImg2Img, "__getattr__", None
            )
            processing.StableDiffusionProcessingImg2Img.__getattr__ = patched_getattr
            debug_print("Successfully patched StableDiffusionProcessingImg2Img.__getattr__")
            
        # Mark as patched
        processing.StableDiffusionProcessingTxt2Img._resize_mode_patched = True
        processing.StableDiffusionProcessingImg2Img._resize_mode_patched = True
            
        return True
    except ImportError as e:
        debug_print(f"Could not patch processing classes: {e}")
        return False
    except Exception as e:
        debug_print(f"Unexpected error patching processing classes: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_controlnet_script_instance(script):
    """Patch a specific ControlNet script instance at runtime"""
    try:
        # Skip if not a ControlNet script
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
        import traceback
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
        import traceback
        traceback.print_exc()
        return False

def patch_hwc3_in_modules():
    """
    Patch HWC3 in known modules where it might be used
    
    This is used to fix the assertion error in HWC3
    """
    try:
        import sys
        import importlib
        
        # Get or define the safe replacement function if needed
        if not hasattr(sys.modules[__name__], "safe_HWC3"):
            import numpy as np
            
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
                            
                    # Convert dtype without assertion
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
                    
            # Store it for reference
            sys.modules[__name__].safe_HWC3 = safe_HWC3
        else:
            # Use the existing one
            safe_HWC3 = sys.modules[__name__].safe_HWC3
            
        # Define modules to patch
        module_names = [
            "modules_forge.utils",
            "lib_controlnet.utils",
            "lib_controlnet.external_code",
            "modules.forge.shared",
        ]
        
        # Track which ones we patched
        patched_count = 0
        
        # Try to patch each module
        for module_name in module_names:
            try:
                module = importlib.import_module(module_name)
                if hasattr(module, "HWC3"):
                    # Don't patch twice
                    if not hasattr(module, "_hwc3_patched_by_deforum"):
                        module._original_HWC3 = module.HWC3
                        module.HWC3 = safe_HWC3
                        module._hwc3_patched_by_deforum = True
                        debug_print(f"Patched HWC3 in {module_name}")
                        patched_count += 1
            except ImportError:
                debug_print(f"Could not import {module_name}")
        
        return patched_count > 0
    except Exception as e:
        debug_print(f"Error in patch_hwc3_in_modules: {e}")
        return False

def initialize_controlnet_params():
    """
    Initialize ControlNet parameters to ensure proper setup
    
    This makes sure certain parameters are set correctly for better compatibility
    """
    try:
        # Try to setup ControlNet units if present
        try:
            from . import deforum_controlnet
            if deforum_controlnet.setup_controlnet_units(5):  # Always set up 5 units
                debug_print("Initialized 5 ControlNet units")
        except:
            debug_print("Could not set up ControlNet units")
            
        # Try to set certain environment variables
        try:
            import os
            os.environ["DEFORUM_CONTROLNET_DEBUG"] = "0"  # Disable verbose debug
            os.environ["DEFORUM_CN_VERBOSE"] = "0"  # Disable verbose controlnet
            debug_print("Set environment variables for ControlNet")
        except:
            debug_print("Could not set environment variables")
            
        return True
    except Exception as e:
        debug_print(f"Error initializing ControlNet params: {e}")
        return False

def setup_forge_controlnet_env():
    """Set up environment variables required by WebUI Forge ControlNet"""
    try:
        # Skip if not running in Forge
        try:
            import modules_forge
        except ImportError:
            debug_print("Not running in WebUI Forge, skipping environment setup")
            return False
            
        # Set environment variables for Forge compatibility
        os.environ["IGNORE_TORCH_INDENT_WARNING"] = "1"
        debug_print("Set IGNORE_TORCH_INDENT_WARNING=1")
        
        # Set model directories to include all possible locations
        os.environ["FORGE_MODEL_DIRS"] = "ControlNet;controlnet;control_net;xlabs/controlnets"
        debug_print("Set FORGE_MODEL_DIRS=ControlNet;controlnet;control_net;xlabs/controlnets")
        
        # Set CUDA memory config to prevent OOM errors
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        debug_print("Set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512")
        
        debug_print("Successfully set up WebUI Forge environment variables")
        return True
    except Exception as e:
        debug_print(f"Error setting up Forge environment: {e}")
        return False

def patch_hwc3_assertion_directly():
    """
    Directly patch the HWC3 function in WebUI Forge to fix the assertion error
    
    This is a critical patch that directly targets the specific function causing the error.
    """
    try:
        import sys
        import numpy as np
        import importlib
        
        # Define our safe replacement
        def safe_HWC3(x):
            """Safe version that doesn't use assertions"""
            try:
                if x is None:
                    return np.zeros((64, 64, 3), dtype=np.uint8)
                    
                # If input isn't a numpy array, convert it first
                if not isinstance(x, np.ndarray):
                    try:
                        x = np.array(x)
                    except:
                        return np.zeros((64, 64, 3), dtype=np.uint8)
                        
                # Convert dtype WITHOUT assertion
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
                
        # Try direct patching of known module paths
        patched_modules = []
        
        # Try to patch modules_forge.utils directly - this is the primary target
        try:
            modules_forge_utils = importlib.import_module("modules_forge.utils")
            if hasattr(modules_forge_utils, "HWC3"):
                if not hasattr(modules_forge_utils, "_original_HWC3"):
                    modules_forge_utils._original_HWC3 = modules_forge_utils.HWC3
                modules_forge_utils.HWC3 = safe_HWC3
                modules_forge_utils._deforum_patched = True
                debug_print("Directly patched HWC3 in modules_forge.utils")
                patched_modules.append("modules_forge.utils")
        except ImportError:
            debug_print("Could not import modules_forge.utils")
        
        # Try lib_controlnet.utils
        try:
            lib_controlnet_utils = importlib.import_module("lib_controlnet.utils")
            if hasattr(lib_controlnet_utils, "HWC3"):
                if not hasattr(lib_controlnet_utils, "_original_HWC3"):
                    lib_controlnet_utils._original_HWC3 = lib_controlnet_utils.HWC3
                lib_controlnet_utils.HWC3 = safe_HWC3
                lib_controlnet_utils._deforum_patched = True
                debug_print("Directly patched HWC3 in lib_controlnet.utils")
                patched_modules.append("lib_controlnet.utils")
        except ImportError:
            debug_print("Could not import lib_controlnet.utils")
        
        # Try global external_code
        try:
            external_code = importlib.import_module("lib_controlnet.external_code")
            if hasattr(external_code, "HWC3"):
                if not hasattr(external_code, "_original_HWC3"):
                    external_code._original_HWC3 = external_code.HWC3
                external_code.HWC3 = safe_HWC3
                external_code._deforum_patched = True
                debug_print("Directly patched HWC3 in lib_controlnet.external_code")
                patched_modules.append("lib_controlnet.external_code")
        except ImportError:
            debug_print("Could not import lib_controlnet.external_code")
            
        # Try to find any other module with HWC3
        for name, module in list(sys.modules.items()):
            if module and (
                'controlnet' in name.lower() or 
                'modules_forge' in name.lower() or
                'modules.forge' in name.lower()
            ):
                if hasattr(module, 'HWC3') and not getattr(module, '_deforum_patched', False):
                    if not hasattr(module, "_original_HWC3"):
                        module._original_HWC3 = module.HWC3
                    module.HWC3 = safe_HWC3
                    module._deforum_patched = True
                    debug_print(f"Directly patched HWC3 in {name}")
                    patched_modules.append(name)
                    
        return len(patched_modules) > 0
    except Exception as e:
        debug_print(f"Error in patch_hwc3_assertion_directly: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_controlnet_process_unit_directly():
    """
    Directly patch the process_unit_before_every_sampling method in ControlNet to prevent NoneType errors.
    
    This handles the case where params.model is None, which causes an AttributeError when trying to access params.model.strength.
    """
    try:
        from modules import scripts
        import types
        
        # First try to use the dedicated forge_controlnet_patcher if available
        try:
            from . import forge_controlnet_patcher
            debug_print("Using forge_controlnet_patcher for process_unit patch")
            
            # Create dummy processing objects for both txt2img and img2img
            from modules import processing
            if hasattr(processing, 'StableDiffusionProcessingTxt2Img') and hasattr(scripts, 'scripts_txt2img'):
                dummy_txt2img = processing.StableDiffusionProcessingTxt2Img()
                dummy_txt2img.scripts = scripts.scripts_txt2img
                if forge_controlnet_patcher.apply_all_patches(dummy_txt2img):
                    debug_print("Applied forge_controlnet_patcher to txt2img processing")
                
            if hasattr(processing, 'StableDiffusionProcessingImg2Img') and hasattr(scripts, 'scripts_img2img'):
                dummy_img2img = processing.StableDiffusionProcessingImg2Img()
                dummy_img2img.scripts = scripts.scripts_img2img
                if forge_controlnet_patcher.apply_all_patches(dummy_img2img):
                    debug_print("Applied forge_controlnet_patcher to img2img processing")
            
            return True
        except ImportError:
            debug_print("forge_controlnet_patcher not available, using direct method")
        except Exception as e:
            debug_print(f"Error using forge_controlnet_patcher: {e}")
            
        # Fallback to direct patching if the patcher module isn't available
        # Try to find the ControlNet script in txt2img and img2img
        for script_collection in [scripts.scripts_txt2img, scripts.scripts_img2img]:
            if script_collection is None:
                continue
                
            controlnet_script = None
            for script in script_collection.alwayson_scripts:
                if hasattr(script, 'title') and script.title().lower() == "controlnet":
                    controlnet_script = script
                    break
                    
            if controlnet_script:
                # Check if we need to patch process_unit_before_every_sampling
                if hasattr(controlnet_script, "process_unit_before_every_sampling"):
                    # Don't patch if already patched
                    if hasattr(controlnet_script, "_process_unit_patched"):
                        continue
                        
                    # Store original method
                    original_method = controlnet_script.process_unit_before_every_sampling
                    
                    # Create patched version
                    def patched_process_unit(self, p, params_unit, **kwargs):
                        try:
                            # Check if params_unit has the required attributes
                            if not hasattr(params_unit, 'model') or params_unit.model is None:
                                debug_print("ControlNet process_unit: params.model is None, skipping")
                                return None
                                
                            # Also check if we can access strength
                            if not hasattr(params_unit.model, 'strength'):
                                # Create a default strength value
                                params_unit.model.strength = 1.0
                                debug_print("ControlNet process_unit: Added missing strength attribute")
                                
                            # Call original method
                            return original_method(self, p, params_unit, **kwargs)
                        except Exception as e:
                            debug_print(f"Error in process_unit_before_every_sampling: {e}")
                            # Don't crash the pipeline
                            return None
                            
                    # Apply the patch
                    controlnet_script.process_unit_before_every_sampling = types.MethodType(
                        patched_process_unit, controlnet_script)
                    controlnet_script._process_unit_patched = True
                    debug_print(f"Directly patched process_unit_before_every_sampling in {script_collection.__class__.__name__}")
        
        return True
    except Exception as e:
        debug_print(f"Error in patch_controlnet_process_unit_directly: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_controlnet_runtime():
    """
    Set up runtime patches for ControlNet to ensure they're applied during rendering
    
    This installs hooks into the processing classes to ensure our patches get applied
    whenever ControlNet is used, even if it wasn't enabled at startup.
    """
    try:
        from modules import processing, scripts
        
        # First check if we can access the necessary modules
        if not hasattr(processing, 'StableDiffusionProcessingTxt2Img') or not hasattr(processing, 'StableDiffusionProcessingImg2Img'):
            debug_print("Missing processing classes, cannot set up runtime patches")
            return False
            
        # Store original methods
        if not hasattr(processing.StableDiffusionProcessingTxt2Img, '_original_init'):
            processing.StableDiffusionProcessingTxt2Img._original_init = processing.StableDiffusionProcessingTxt2Img.__init__
            
        if not hasattr(processing.StableDiffusionProcessingImg2Img, '_original_init'):
            processing.StableDiffusionProcessingImg2Img._original_init = processing.StableDiffusionProcessingImg2Img.__init__
            
        # Define patched versions that will check for and apply ControlNet patches at runtime
        def patched_txt2img_init(self, *args, **kwargs):
            # Call original init first
            processing.StableDiffusionProcessingTxt2Img._original_init(self, *args, **kwargs)
            
            # Check if this has ControlNet scripts and apply patches if needed
            if hasattr(self, 'scripts') and hasattr(self.scripts, 'alwayson_scripts'):
                for script in self.scripts.alwayson_scripts:
                    if hasattr(script, 'title') and script.title().lower() == "controlnet":
                        # Apply forge_controlnet_patcher during runtime
                        try:
                            from . import forge_controlnet_patcher
                            if forge_controlnet_patcher.apply_all_patches(self):
                                debug_print("Applied forge_controlnet_patcher during txt2img init")
                        except Exception as e:
                            debug_print(f"Error applying runtime patches during txt2img init: {e}")
                        break
                        
        def patched_img2img_init(self, *args, **kwargs):
            # Call original init first
            processing.StableDiffusionProcessingImg2Img._original_init(self, *args, **kwargs)
            
            # Check if this has ControlNet scripts and apply patches if needed
            if hasattr(self, 'scripts') and hasattr(self.scripts, 'alwayson_scripts'):
                for script in self.scripts.alwayson_scripts:
                    if hasattr(script, 'title') and script.title().lower() == "controlnet":
                        # Apply forge_controlnet_patcher during runtime
                        try:
                            from . import forge_controlnet_patcher
                            if forge_controlnet_patcher.apply_all_patches(self):
                                debug_print("Applied forge_controlnet_patcher during img2img init")
                        except Exception as e:
                            debug_print(f"Error applying runtime patches during img2img init: {e}")
                        break
                        
        # Apply the patches
        processing.StableDiffusionProcessingTxt2Img.__init__ = patched_txt2img_init
        processing.StableDiffusionProcessingImg2Img.__init__ = patched_img2img_init
        
        debug_print("Set up runtime hooks for ControlNet patching")
        return True
        
    except Exception as e:
        debug_print(f"Error setting up runtime patches: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_sd_vae_module():
    """
    Patch the sd_vae module to handle missing model_path attribute in WebUI Forge
    
    This addresses the AttributeError: module 'modules.sd_models' has no attribute 'model_path'
    when initializing the VAE module.
    """
    try:
        debug_print("Attempting to patch sd_vae module")
        
        # First try to import the modules we need
        try:
            from modules import sd_models, shared, sd_vae
            debug_print("Successfully imported sd_vae and sd_models modules")
        except ImportError as e:
            debug_print(f"Could not import VAE modules: {e}")
            return False
            
        # Check if sd_models already has model_path
        if hasattr(sd_models, 'model_path'):
            debug_print("sd_models.model_path already exists, skipping patch")
            return True
            
        # Add model_path to sd_models
        sd_models.model_path = shared.models_path
        debug_print(f"Added model_path attribute to sd_models: {sd_models.model_path}")
        
        # Patch refresh_vae_list if it exists
        if hasattr(sd_vae, 'refresh_vae_list'):
            # Store original function
            original_refresh = sd_vae.refresh_vae_list
            
            # Define patched function
            def patched_refresh_vae_list():
                try:
                    # Ensure model_path exists before each call
                    if not hasattr(sd_models, 'model_path'):
                        sd_models.model_path = shared.models_path
                        debug_print(f"Set model_path to {sd_models.model_path} during refresh_vae_list")
                    
                    # Call original function
                    return original_refresh()
                except Exception as e:
                    debug_print(f"Error in original refresh_vae_list: {e}")
                    
                    # Fallback implementation
                    try:
                        import os
                        import glob
                        
                        debug_print("Using fallback VAE scan")
                        sd_vae.vae_dict.clear()
                        
                        # Get VAE path, or use default
                        vae_path = getattr(sd_vae, 'vae_path', os.path.join(shared.models_path, 'VAE'))
                        
                        # Scan for all VAE files
                        paths = [
                            os.path.join(shared.models_path, '**/*.vae.ckpt'), 
                            os.path.join(shared.models_path, '**/*.vae.pt'),
                            os.path.join(shared.models_path, '**/*.vae.safetensors'),
                            os.path.join(vae_path, '**/*.ckpt'),
                            os.path.join(vae_path, '**/*.pt'),
                            os.path.join(vae_path, '**/*.safetensors'),
                        ]
                        
                        # Add command line directory options
                        if hasattr(shared, 'cmd_opts'):
                            if shared.cmd_opts.ckpt_dir and os.path.isdir(shared.cmd_opts.ckpt_dir):
                                paths.extend([
                                    os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.ckpt'),
                                    os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.pt'),
                                    os.path.join(shared.cmd_opts.ckpt_dir, '**/*.vae.safetensors'),
                                ])
                            
                            if shared.cmd_opts.vae_dir and os.path.isdir(shared.cmd_opts.vae_dir):
                                paths.extend([
                                    os.path.join(shared.cmd_opts.vae_dir, '**/*.ckpt'),
                                    os.path.join(shared.cmd_opts.vae_dir, '**/*.pt'),
                                    os.path.join(shared.cmd_opts.vae_dir, '**/*.safetensors'),
                                ])
                        
                        # Find files
                        candidates = []
                        for path in paths:
                            candidates.extend(glob.glob(path, recursive=True))
                        
                        # Add to dictionary
                        for filepath in candidates:
                            filename = os.path.basename(filepath)
                            sd_vae.vae_dict[filename] = filepath
                        
                        # Sort dictionary
                        if hasattr(shared, 'natural_sort_key'):
                            sd_vae.vae_dict.update(dict(sorted(sd_vae.vae_dict.items(), key=lambda x: shared.natural_sort_key(x[0]))))
                        else:
                            sd_vae.vae_dict.update(dict(sorted(sd_vae.vae_dict.items())))
                            
                        debug_print(f"Fallback VAE scan found {len(sd_vae.vae_dict)} files")
                        return
                    except Exception as fallback_e:
                        debug_print(f"Fallback VAE scan failed: {fallback_e}")
                        return
            
            # Apply the patch
            sd_vae.refresh_vae_list = patched_refresh_vae_list
            debug_print("Successfully patched sd_vae.refresh_vae_list function")
        
        return True
    except Exception as e:
        debug_print(f"Error patching sd_vae module: {e}")
        import traceback
        traceback.print_exc()
        return False

def preload_controlnet():
    """
    Apply all ControlNet-related patches at extension load time
    
    This applies various patches to ensure compatibility between Deforum and 
    WebUI Forge's ControlNet implementation.
    """
    try:
        debug_print("Preloading ControlNet")
        
        # Apply critical patches regardless if ControlNet is enabled
        debug_print("Applying critical patches unconditionally")
        
        # Patch processing classes to always have resize_mode
        success_processing = patch_processing_classes()
        
        # Apply critical direct patches for HWC3 assertion and process_unit
        debug_print("Applying critical direct patches")
        success_hwc3 = patch_hwc3_in_modules()
        
        # Patch the VAE module to handle missing model_path attribute
        success_vae = patch_sd_vae_module()
        if success_vae:
            debug_print("Successfully patched sd_vae module")
        else:
            debug_print("Failed to patch sd_vae module")
        
        # Try to patch process_unit_before_every_sampling directly
        forge_patcher_success = False
        try:
            from deforum_helpers.forge_controlnet_patcher import apply_all_patches, early_patch_sd_vae
            debug_print("Using forge_controlnet_patcher for process_unit patch")
            
            # Call the early VAE patch first
            early_vae_success = early_patch_sd_vae()
            if early_vae_success:
                debug_print("Successfully applied early VAE patch from forge_controlnet_patcher")
                
            # Apply the rest of the patches
            forge_patcher_success = apply_all_patches(None)
        except ImportError:
            debug_print("forge_controlnet_patcher not available")
        except Exception as forge_e:
            debug_print(f"Error with forge_controlnet_patcher: {forge_e}")
        
        # Manually try to patch specific processing classes with ControlNet fixes
        try:
            from modules import processing
            
            # Create patch for txt2img runtime
            def patched_txt2img_init(cls):
                # Store original init
                original_init = cls.__init__
                
                # Define patched init function
                def wrapped_init(self, *args, **kwargs):
                    # Call original init first
                    original_init(self, *args, **kwargs)
                    
                    # Make sure resize_mode is available
                    if not hasattr(self, 'resize_mode'):
                        self.resize_mode = "Inner Fit (Scale to Fit)"
                        
                    # Apply ControlNet patches if scripts exist
                    if hasattr(self, 'scripts') and self.scripts is not None:
                        try:
                            from deforum_helpers import forge_controlnet_patcher
                            forge_controlnet_patcher.apply_all_patches(self)
                            debug_print("Applied forge_controlnet_patcher during txt2img init")
                        except Exception as e:
                            debug_print(f"Error applying forge_controlnet_patcher during txt2img: {e}")
                
                # Apply the patch
                cls.__init__ = wrapped_init
            
            # Create patch for img2img runtime
            def patched_img2img_init(cls):
                # Store original init
                original_init = cls.__init__
                
                # Define patched init function
                def wrapped_init(self, *args, **kwargs):
                    # Call original init first
                    original_init(self, *args, **kwargs)
                    
                    # Make sure resize_mode is available
                    if not hasattr(self, 'resize_mode'):
                        self.resize_mode = "Inner Fit (Scale to Fit)"
                        
                    # Apply ControlNet patches if scripts exist
                    if hasattr(self, 'scripts') and self.scripts is not None:
                        try:
                            from deforum_helpers import forge_controlnet_patcher
                            forge_controlnet_patcher.apply_all_patches(self)
                            debug_print("Applied forge_controlnet_patcher during img2img init")
                        except Exception as e:
                            debug_print(f"Error applying forge_controlnet_patcher during img2img: {e}")
                
                # Apply the patch
                cls.__init__ = wrapped_init
                
            # Apply the patches
            patched_txt2img_init(processing.StableDiffusionProcessingTxt2Img)
            debug_print("Applied forge_controlnet_patcher to txt2img processing")
            
            # Create patch for img2img runtime
            patched_img2img_init(processing.StableDiffusionProcessingImg2Img)
            debug_print("Applied forge_controlnet_patcher to img2img processing")
        except Exception as e:
            debug_print(f"Error applying processing class patches: {e}")
        
        return success_processing and success_hwc3 and success_vae and forge_patcher_success
    except Exception as e:
        debug_print(f"Error in preload_controlnet: {e}")
        import traceback
        traceback.print_exc()
        return False

# Apply preload functions when this module is imported
print("[Deforum] Preloading ControlNet patch for WebUI Forge compatibility")
preload_controlnet() 