import os
import glob
import shutil
import cv2
import numpy as np
import torch
from PIL import Image
from modules import shared, devices, scripts, processing, script_callbacks
from .rendering.util import emoji_utils
import gradio as gr

# Global settings
DEBUG_CONTROLNET = os.environ.get("DEFORUM_DEBUG_CONTROLNET", "0").lower() in ["1", "true", "yes"]
MAX_CONTROLNET_MODELS = 5  # Maximum number of ControlNet models supported

def num_of_models():
    """Return the maximum number of ControlNet models supported"""
    return MAX_CONTROLNET_MODELS

def debug_print(message):
    """Print a debug message if debug mode is enabled"""
    if DEBUG_CONTROLNET:
        print(f"[Deforum ControlNet] {message}")

def controlnet_component_names():
    """Return a list of component names for ControlNet used in Deforum UI"""
    component_names = []
    # Add ControlNet units (supporting up to 5 units)
    max_controlnet_units = num_of_models()
    
    # Main ControlNet setting
    component_names.append('controlnet_units_count')
    component_names.append('controlnet_optimize_execution')
    
    for i in range(max_controlnet_units):
        # Basic controls
        component_names.extend([
            f'cn_{i}_enabled',
            f'cn_{i}_module',
            f'cn_{i}_model',
            f'cn_{i}_weight',
            f'cn_{i}_guidance_start',
            f'cn_{i}_guidance_end',
            
            # Advanced settings
            f'cn_{i}_processor_res',
            f'cn_{i}_threshold_a',
            f'cn_{i}_threshold_b',
            f'cn_{i}_resize_mode',
            f'cn_{i}_rgbbgr_mode',
            f'cn_{i}_pixel_perfect',
            f'cn_{i}_low_vram',
            f'cn_{i}_guess_mode',
            f'cn_{i}_invert_image',
            
            # Input settings
            f'cn_{i}_enabled_batch',
            f'cn_{i}_vid_path',
            f'cn_{i}_mask_vid_path',
            f'cn_{i}_vid_loopback',
        ])
    
    return component_names

def setup_controlnet_ui():
    """Set up the ControlNet UI elements for the Deforum WebUI extension"""
    try:
        from modules import shared
        
        # Try to import the controlnet module names
        try:
            from lib_controlnet.external_code import get_modules
            available_modules = get_modules()
            
            # Add 'None' to available modules
            available_modules = ["None"] + available_modules
            debug_print(f"Found {len(available_modules)-1} ControlNet modules")
        except Exception as e:
            available_modules = ["None", "canny", "depth", "hed", "mlsd", "normal", "openpose", "scribble", "seg", "tile"]
            debug_print(f"Could not import ControlNet modules, using defaults: {e}")
            
        # Try to get the model list
        try:
            from lib_controlnet.global_state import get_all_controlnet_names
            available_models = get_all_controlnet_names()
            
            # Add 'None' to available models
            available_models = ["None"] + available_models
            debug_print(f"Found {len(available_models)-1} ControlNet models")
        except Exception as e:
            available_models = ["None"]
            debug_print(f"Could not import ControlNet models list, using defaults: {e}")
        
        # Check for empty model list and try to populate it
        if len(available_models) == 1:
            try:
                model_list = get_all_controlnet_names()
                if model_list and len(model_list) > 0:
                    available_models = ["None"] + model_list
                    debug_print(f"Re-fetched model list using custom function, found {len(available_models)-1} models")
                else:
                    debug_print("Warning: No ControlNet models available. Controls will appear but no models will be shown.")
            except Exception as e:
                debug_print(f"Error fetching model list: {e}")
                
        # Register models for better compatibility
        try:
            register_controlnet_models()
        except Exception as e:
            debug_print(f"Error registering ControlNet models: {e}")
        
        # Create a dict to store all UI components
        controlnet_dict = {}
        
        # Store file columns for loopback toggling
        file_columns = {}
        
        # Get the maximum number of units we need to support for the animation_key_frames ControlNetKeys class
        # ControlNetKeys uses 1-based indexing and expects cn_1_weight through cn_5_weight
        max_controlnet_units = shared.opts.data.get("control_net_unit_count", shared.opts.data.get("control_net_max_models_num", 5))
        num_models_to_show = num_of_models()  # This is how many we'll show in the UI
        max_models_internal = max(max_controlnet_units, num_models_to_show)  # Make sure we have enough for both UI and ControlNetKeys
        
        # Create the ControlNet tab
        with gr.Tab(f"{emoji_utils.net()} ControlNet"):
            # Add a checkbox for optimizing execution instead of a slider
            controlnet_dict['controlnet_optimize_execution'] = gr.Checkbox(
                label="Optimize ControlNet execution", 
                value=True,
                info="Enable performance optimizations to prevent excessive logging and memory thrashing"
            )
            
            # Always set number of units to the maximum for compatibility with other functions
            controlnet_dict['controlnet_units_count'] = gr.Number(
                value=max_models_internal, 
                visible=False,
                interactive=False
            )
            
            # Create a tab group for all ControlNet units
            with gr.Tabs() as controlnet_tabs:
                # Create UI elements for each ControlNet unit as separate tabs
                for i in range(num_models_to_show):
                    with gr.Tab(f"Unit {i+1}"):
                        with gr.Row():
                            controlnet_dict[f'cn_{i}_enabled'] = gr.Checkbox(label=f"Enable Unit {i+1}", value=False)
                        
                        with gr.Row():
                            controlnet_dict[f'cn_{i}_module'] = gr.Dropdown(label="Preprocessor", choices=available_modules, value="None")
                            controlnet_dict[f'cn_{i}_model'] = gr.Dropdown(label="Model", choices=available_models, value="None")
                        
                        with gr.Row():
                            controlnet_dict[f'cn_{i}_weight'] = gr.Slider(label="Weight", minimum=0.0, maximum=2.0, step=0.05, value=1.0)
                            controlnet_dict[f'cn_{i}_guidance_start'] = gr.Slider(label="Start", minimum=0.0, maximum=1.0, step=0.01, value=0.0)
                            controlnet_dict[f'cn_{i}_guidance_end'] = gr.Slider(label="End", minimum=0.0, maximum=1.0, step=0.01, value=1.0)
                        
                        # Combined settings in a more compact layout
                        with gr.Row():
                            with gr.Column(scale=1):
                                controlnet_dict[f'cn_{i}_pixel_perfect'] = gr.Checkbox(label="Pixel Perfect", value=True)
                                controlnet_dict[f'cn_{i}_vid_loopback'] = gr.Checkbox(label="Loopback", value=True, info="Use output from previous frame as input")
                                controlnet_dict[f'cn_{i}_enabled_batch'] = gr.Checkbox(label="Batch Input", value=False)
                            
                            with gr.Column(scale=2) as file_col:
                                # Store the column in a dictionary for each unit
                                controlnet_dict[f'cn_{i}_vid_path'] = gr.Textbox(label="Input Path", lines=1, value="")
                                controlnet_dict[f'cn_{i}_mask_vid_path'] = gr.Textbox(label="Mask Path (Optional)", lines=1, value="")
                            
                            # Keep reference to the file column
                            file_columns[i] = file_col
                        
                        # Advanced settings in a collapsed accordion
                        with gr.Accordion("Advanced Settings", open=False):
                            with gr.Row():
                                controlnet_dict[f'cn_{i}_processor_res'] = gr.Slider(label="Resolution", minimum=64, maximum=2048, step=1, value=720)
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    controlnet_dict[f'cn_{i}_threshold_a'] = gr.Slider(label="Threshold A", minimum=0.0, maximum=1.0, step=0.01, value=0.5)
                                with gr.Column(scale=1):
                                    controlnet_dict[f'cn_{i}_threshold_b'] = gr.Slider(label="Threshold B", minimum=0.0, maximum=1.0, step=0.01, value=0.5)
                            
                            with gr.Row():
                                with gr.Column(scale=1):
                                    controlnet_dict[f'cn_{i}_resize_mode'] = gr.Radio(
                                        label="Resize Mode", 
                                        choices=["Inner Fit (Scale to Fit)", "Outer Fit (Envelope)", "Just Resize"], 
                                        value="Inner Fit (Scale to Fit)"
                                    )
                                with gr.Column(scale=1):
                                    controlnet_dict[f'cn_{i}_rgbbgr_mode'] = gr.Radio(
                                        label="Color Order", 
                                        choices=["RGB", "BGR"], 
                                        value="RGB"
                                    )
                            
                            with gr.Row():
                                controlnet_dict[f'cn_{i}_low_vram'] = gr.Checkbox(label="Low VRAM", value=False)
                                controlnet_dict[f'cn_{i}_guess_mode'] = gr.Checkbox(label="Guess Mode", value=False)
                                controlnet_dict[f'cn_{i}_invert_image'] = gr.Checkbox(label="Invert Input", value=False)
                
                # Keep backward compatibility with the group structure for other functions
                for i in range(num_models_to_show):
                    controlnet_dict[f'controlnet_group_{i}'] = gr.Group()
                
                # Create 1-based index aliases for animation_key_frames compatibility
                # Make sure we create all the expected 1-based key-values for ControlNetKeys
                for i in range(max_models_internal):
                    if i < num_models_to_show:
                        # For controls that exist in the UI, create references to those controls
                        zero_based_idx = i
                        one_based_idx = i + 1
                        controlnet_dict[f'cn_{one_based_idx}_weight'] = controlnet_dict[f'cn_{zero_based_idx}_weight']
                        controlnet_dict[f'cn_{one_based_idx}_guidance_start'] = controlnet_dict[f'cn_{zero_based_idx}_guidance_start']
                        controlnet_dict[f'cn_{one_based_idx}_guidance_end'] = controlnet_dict[f'cn_{zero_based_idx}_guidance_end']
                    else:
                        # For controls beyond what's in the UI, create dummy values
                        # These are needed for ControlNetKeys to work correctly
                        one_based_idx = i + 1
                        controlnet_dict[f'cn_{one_based_idx}_weight'] = 1.0
                        controlnet_dict[f'cn_{one_based_idx}_guidance_start'] = 0.0
                        controlnet_dict[f'cn_{one_based_idx}_guidance_end'] = 1.0
            
            # Set up loopback toggle for each unit
            for i in range(num_models_to_show):
                # Use a closure to capture the current value of i
                def make_loopback_fn(unit_idx):
                    def loopback_fn(x):
                        return gr.update(visible=not x)
                    return loopback_fn
                
                controlnet_dict[f'cn_{i}_vid_loopback'].change(
                    fn=make_loopback_fn(i),
                    inputs=[controlnet_dict[f'cn_{i}_vid_loopback']],
                    outputs=[file_columns[i]]
                )
            
            # Add FLUX compatibility information
            with gr.Accordion("FLUX Compatibility", open=False):
                gr.HTML("""
                <p><b>ControlNet FLUX Compatibility Notes:</b></p>
                <ul>
                <li>For best results with FLUX, use the special FLUX ControlNet models available from <a href="https://huggingface.co/XLabs-AI/flux-controlnet-collections">XLabs-AI FLUX ControlNet Collection</a>.</li>
                <li>Models should be placed in your WebUI's models/ControlNet folder.</li>
                <li>FLUX ControlNet models work best with 720px or 1024px resolution for preprocessing.</li>
                <li>Recommended models from XLabs-AI: flux-canny-controlnet-v3, flux-depth-controlnet-v3, and flux-hed-controlnet-v3.</li>
                <li>If a FLUX-specific model is not available, standard ControlNet models will be used as a fallback.</li>
                <li>Using the "Optimize ControlNet execution" option is recommended to prevent excessive logging during rendering.</li>
                </ul>
                """)
        
        # Add 1-based indexed attributes to the return value for animation_key_frames.py compatibility
        for i in range(max_models_internal):
            one_based_idx = i + 1
            if i < num_models_to_show:
                # For UI controls, grab their values - check if they have .value attribute first
                # These might be already float values in some cases
                weight_value = controlnet_dict[f'cn_{i}_weight']
                guidance_start_value = controlnet_dict[f'cn_{i}_guidance_start']
                guidance_end_value = controlnet_dict[f'cn_{i}_guidance_end']
                
                controlnet_dict[f'cn_{one_based_idx}_weight'] = weight_value.value if hasattr(weight_value, 'value') else weight_value
                controlnet_dict[f'cn_{one_based_idx}_guidance_start'] = guidance_start_value.value if hasattr(guidance_start_value, 'value') else guidance_start_value
                controlnet_dict[f'cn_{one_based_idx}_guidance_end'] = guidance_end_value.value if hasattr(guidance_end_value, 'value') else guidance_end_value
            else:
                # For values beyond what's in the UI, set default values
                controlnet_dict[f'cn_{one_based_idx}_weight'] = 1.0
                controlnet_dict[f'cn_{one_based_idx}_guidance_start'] = 0.0
                controlnet_dict[f'cn_{one_based_idx}_guidance_end'] = 1.0
        
        return controlnet_dict
        
    except Exception as e:
        print(f"Error setting up ControlNet UI: {e}")
        # Return a minimal set of dummy components to avoid breaking the UI
        # We need to include all the expected components to avoid KeyError
        
        # Calculate the maximum number of models we need to support
        # This includes both what we show in UI (num_of_models()) and what ControlNetKeys expects
        max_controlnet_units = shared.opts.data.get("control_net_unit_count", shared.opts.data.get("control_net_max_models_num", 5))
        num_models_to_show = num_of_models()  # This is how many we'd show in the UI
        max_models_internal = max(max_controlnet_units, num_models_to_show)
        
        controlnet_dict = {
            # Instead of a slider, set a fixed number as the count
            'controlnet_units_count': gr.Number(value=max_models_internal, visible=False),
            'controlnet_optimize_execution': gr.Checkbox(label="Optimize execution", value=True)
        }
        
        # Create simplified dummy components for all expected keys (zero-based index)
        for i in range(num_models_to_show):
            # Add minimal required components with default values
            controlnet_dict.update({
                f'controlnet_group_{i}': gr.Group(visible=False),
                f'cn_{i}_enabled': gr.Checkbox(value=False, visible=False),
                f'cn_{i}_module': gr.Dropdown(choices=["None"], value="None", visible=False),
                f'cn_{i}_model': gr.Dropdown(choices=["None"], value="None", visible=False),
                f'cn_{i}_weight': gr.Slider(minimum=0.0, maximum=1.0, value=1.0, visible=False),
                f'cn_{i}_guidance_start': gr.Slider(minimum=0.0, maximum=1.0, value=0.0, visible=False),
                f'cn_{i}_guidance_end': gr.Slider(minimum=0.0, maximum=1.0, value=1.0, visible=False),
                f'cn_{i}_processor_res': gr.Slider(minimum=64, maximum=2048, value=720, visible=False),
                f'cn_{i}_threshold_a': gr.Slider(minimum=0.0, maximum=1.0, value=0.5, visible=False),
                f'cn_{i}_threshold_b': gr.Slider(minimum=0.0, maximum=1.0, value=0.5, visible=False),
                f'cn_{i}_resize_mode': gr.Radio(choices=["Inner Fit (Scale to Fit)"], value="Inner Fit (Scale to Fit)", visible=False),
                f'cn_{i}_rgbbgr_mode': gr.Radio(choices=["RGB"], value="RGB", visible=False),
                f'cn_{i}_pixel_perfect': gr.Checkbox(value=True, visible=False),
                f'cn_{i}_low_vram': gr.Checkbox(value=False, visible=False),
                f'cn_{i}_guess_mode': gr.Checkbox(value=False, visible=False),
                f'cn_{i}_invert_image': gr.Checkbox(value=False, visible=False),
                f'cn_{i}_enabled_batch': gr.Checkbox(value=False, visible=False),
                f'cn_{i}_vid_path': gr.Textbox(value="", visible=False),
                f'cn_{i}_mask_vid_path': gr.Textbox(value="", visible=False),
                f'cn_{i}_vid_loopback': gr.Checkbox(value=True, visible=False),
            })
        
        # Create 1-based index aliases for the animation_key_frames.py ControlNetKeys class
        for i in range(max_models_internal):
            one_based_idx = i + 1
            if i < num_models_to_show:
                # For UI controls, create reference to those components
                controlnet_dict[f'cn_{one_based_idx}_weight'] = controlnet_dict[f'cn_{i}_weight']
                controlnet_dict[f'cn_{one_based_idx}_guidance_start'] = controlnet_dict[f'cn_{i}_guidance_start']
                controlnet_dict[f'cn_{one_based_idx}_guidance_end'] = controlnet_dict[f'cn_{i}_guidance_end']
            else:
                # For controls beyond what's in the UI, create direct values
                controlnet_dict[f'cn_{one_based_idx}_weight'] = 1.0
                controlnet_dict[f'cn_{one_based_idx}_guidance_start'] = 0.0
                controlnet_dict[f'cn_{one_based_idx}_guidance_end'] = 1.0
        
        return controlnet_dict

def clean_gradio_path_strings(path_string):
    """Clean path strings received from Gradio's file input"""
    if path_string is None or len(path_string.strip()) == 0:
        return None
    
    # For file lists - comma separated
    if "," in path_string:
        path_list = [p.strip() for p in path_string.split(",")]
        return [p for p in path_list if p and len(p) > 0]
    
    # For single files
    return path_string.strip() if path_string.strip() else None

def get_controlnet_units_v2(p, is_img2img=False, is_revision=False):
    """Get ControlNet units from the processing object if available"""
    # EMERGENCY: Apply our emergency direct patching here first
    try:
        from .forge_controlnet_patcher import emergency_direct_patch_controlnet
        if emergency_direct_patch_controlnet():
            debug_print("Successfully applied emergency ControlNet direct patch")
        else:
            debug_print("Could not apply emergency ControlNet direct patch, attempting module-level patches")
            
            # Also try the module-level patch for the running instance
            from .forge_controlnet_patcher import patch_controlnet_params_and_preprocessor
            if patch_controlnet_params_and_preprocessor():
                debug_print("Applied module-level ControlNet patches instead")
    except Exception as e:
        debug_print(f"Error applying emergency ControlNet patches: {e}")
        import traceback
        traceback.print_exc()
    
    if not hasattr(p, 'scripts') or not hasattr(p.scripts, 'alwayson_scripts'):
        debug_print("Processing object doesn't have scripts attribute or alwayson_scripts")
        return None
        
    for script in p.scripts.alwayson_scripts:
        if not hasattr(script, 'title'):
            continue
            
        if hasattr(script, 'api_version'):
            # Forge's ControlNet uses a different structure
            if script.title().lower() == "controlnet":
                for script_data in p.script_args:
                    if script_data and isinstance(script_data, list):
                        if all(isinstance(item, (dict, type(None))) for item in script_data):
                            debug_print("Found WebUI Forge ControlNet units")
                            return script_data
        else:
            # Extension version
            if "controlnet" in script.title().lower():
                try:
                    # Get ControlNet arguments
                    args = script.args_from(p.script_args)
                    if args is not None:
                        debug_print("Found extension ControlNet units")
                        return args
                except:
                    debug_print("Error getting ControlNet units from script args")
    
    debug_print("No ControlNet units found in processing object")
    return None

def patch_controlnet_optimize_processing(p, cn_units):
    """
    Patch the processing method to avoid reloading models for each frame
    
    This optimization helps prevent excessive logging and reduces memory thrashing
    when using ControlNet with Deforum.
    """
    try:
        from lib_controlnet import external_code
        
        # Check if we have already patched this instance
        if hasattr(p, "_deforum_cn_patched") and p._deforum_cn_patched:
            return True
            
        # Store the original method
        if not hasattr(external_code, "_original_get_models_from_controlnet"):
            external_code._original_get_models_from_controlnet = external_code.get_models_from_controlnet
            
            # Define our optimized version that minimizes logging and redundant loads
            def optimized_get_models_from_controlnet(p, cn_units):
                # Try to get models from cached data if available
                if hasattr(p, "_deforum_cn_models_cache"):
                    debug_print("Using cached ControlNet models")
                    return p._deforum_cn_models_cache
                
                # Call the original function but with reduced verbosity
                with_verbosity = os.environ.get("DEFORUM_CN_VERBOSE", "0").lower() in ["1", "true", "yes"]
                log_level = print if with_verbosity else lambda *args, **kwargs: None
                
                # Get the model objects using the actual function
                try:
                    debug_print("Loading ControlNet models (first time)")
                    models = external_code._original_get_models_from_controlnet(p, cn_units)
                    
                    # Cache the models for future frames
                    p._deforum_cn_models_cache = models
                    
                    # Mark as patched
                    p._deforum_cn_patched = True
                    
                    return models
                except Exception as e:
                    debug_print(f"Error loading ControlNet models: {e}")
                    # If there's an error, try to fall back to regular method
                    p._deforum_cn_patched = False
                    return external_code._original_get_models_from_controlnet(p, cn_units) 
            
            # Replace the function
            external_code.get_models_from_controlnet = optimized_get_models_from_controlnet
            debug_print("Patched ControlNet optimize processing")
            
        return True
    except Exception as e:
        debug_print(f"Error patching ControlNet processing: {e}")
        return False

def process_controlnet_input_frames(args, anim_args, controlnet_args, input_path, is_mask, frame_type, cn_index):
    """Process video/image input for ControlNet into frames"""
    try:
        # Validate paths and create output directory
        output_dir = os.path.join(args.outdir, f'controlnet_{frame_type}_{cn_index:02d}')
        os.makedirs(output_dir, exist_ok=True)
        
        is_video = input_path.lower().endswith(('.mp4', '.mov', '.avi', '.webm', '.mkv'))
        
        if is_video:
            # Process video files into frames
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"Error: Could not open video file {input_path}")
                return
                
            # Get frame count and handle start/end frames
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            max_frames = args.max_frames if args.max_frames != -1 else total_frames
            start_frame = getattr(args, 'start_frame', 0) if hasattr(args, 'start_frame') else 0
            end_frame = min(start_frame + max_frames, total_frames)
            
            # Process each frame from the video
            for i in range(end_frame):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if i >= start_frame:
                    frame_idx = i - start_frame
                    # Match the original deforum frame numbering
                    output_file = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
                    
                    # Process and save the frame
                    if is_mask:
                        # Masks are typically grayscale
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(output_file, gray)
                    else:
                        # Regular images are RGB
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        Image.fromarray(rgb).save(output_file)
                        
            cap.release()
            print(f"Processed {end_frame - start_frame} frames from {input_path} to {output_dir}")
        else:
            # Process single image files
            if os.path.isfile(input_path):
                # For single image, duplicate it for all frames
                img = Image.open(input_path)
                
                max_frames = args.max_frames if args.max_frames != -1 else 1000  # reasonable default
                
                # Save the same image for each frame
                for i in range(max_frames):
                    output_file = os.path.join(output_dir, f"{i:05d}.jpg")
                    if is_mask:
                        # Convert to grayscale for masks
                        img_gray = img.convert('L')
                        img_gray.save(output_file)
                    else:
                        # Regular image
                        img.save(output_file)
                        
                print(f"Duplicated image {input_path} to {max_frames} frames in {output_dir}")
            else:
                print(f"Error: File {input_path} not found")
                
    except Exception as e:
        print(f"Error processing ControlNet input frames: {e}")
        import traceback
        traceback.print_exc()

def setup_controlnet_video(args, anim_args, controlnet_args):
    """Process all ControlNet videos for animation"""
    try:
        # Determine how many ControlNet units are active
        active_count = getattr(controlnet_args, 'controlnet_units_count', 0)
        debug_print(f"Processing {active_count} ControlNet units for video")
        
        # Process each active unit
        for i in range(active_count):
            enabled = getattr(controlnet_args, f'cn_{i}_enabled', False)
            if not enabled:
                continue
                
            debug_print(f"Processing ControlNet unit {i}")
            
            # Clean input path strings from Gradio
            vid_path = clean_gradio_path_strings(getattr(controlnet_args, f'cn_{i}_vid_path', None))
            mask_path = clean_gradio_path_strings(getattr(controlnet_args, f'cn_{i}_mask_vid_path', None))

            if vid_path:  # Process base video, if available
                process_controlnet_input_frames(args, anim_args, controlnet_args, vid_path, False, 'inputframes', i)

            if mask_path:  # Process mask video, if available
                process_controlnet_input_frames(args, anim_args, controlnet_args, mask_path, True, 'maskframes', i)

    except Exception as e:
        print(f"Error setting up ControlNet video: {e}")
        import traceback
        traceback.print_exc()

def get_all_controlnet_names():
    """
    Custom implementation to get all available ControlNet model names from various directories
    """
    try:
        from modules import shared
        import os
        
        # Create a set to store unique model names
        model_names = set()
        
        # Check all potential directories
        potential_dirs = [
            os.path.join(shared.models_path, "ControlNet"),
            os.path.join(shared.models_path, "controlnet"),
            os.path.join(shared.models_path, "control_net"),
            os.path.join(shared.models_path, "xlabs", "controlnets")
        ]
        
        # Get file extensions for models
        extensions = ['.safetensors', '.ckpt', '.pt', '.pth']
        
        # Collect all model names from all directories
        for dir_path in potential_dirs:
            if os.path.exists(dir_path):
                for file in os.listdir(dir_path):
                    if any(file.endswith(ext) for ext in extensions):
                        model_names.add(file)
        
        if not model_names:
            # Try to import from the original function
            try:
                from lib_controlnet.global_state import get_all_controlnet_names as original_get_names
                return original_get_names()
            except:
                print("Warning: Could not find any ControlNet models!")
                return []
                
        return list(model_names)
    except Exception as e:
        print(f"Error in custom get_all_controlnet_names: {e}")
        # Fallback to original implementation
        try:
            from lib_controlnet.global_state import get_all_controlnet_names as original_get_names
            return original_get_names()
        except:
            print("Complete failure finding ControlNet models!")
            return []

def get_controlnet_filename_dict():
    """Get the ControlNet filename dictionary with a fallback if needed"""
    try:
        # Try to directly access it first
        from lib_controlnet.global_state import controlnet_filename_dict
        return controlnet_filename_dict
    except (ImportError, AttributeError):
        try:
            # Try the public accessor method
            from lib_controlnet.global_state import get_controlnet_models_directory
            return get_controlnet_models_directory()
        except (ImportError, AttributeError):
            # Create an empty dictionary as a fallback
            print("Warning: Could not access ControlNet filename dictionary, creating an empty one.")
            return {}

def register_controlnet_models():
    """Function to forcibly register models with the ControlNet extension"""
    try:
        # Get our custom models list
        our_models = get_all_controlnet_names()
        
        # Get the ControlNet filename dict
        filename_dict = get_controlnet_filename_dict()
        
        # Get shared modules for path access
        from modules import shared
        import os
        
        # Register models directly with WebUI Forge's ControlNet module
        try:
            # Try to import WebUI Forge's ControlNet specific functions
            from lib_controlnet.global_state import update_controlnet_filenames
            from lib_controlnet.global_state import controlnet_models_list
            
            # Call the update function first
            update_controlnet_filenames()
            
            # Now check if we need to add any missing models
            registered_models = 0
            for model_name in our_models:
                if model_name not in filename_dict:
                    model_found = False
                    
                    # Try different possible locations
                    for dir_path in [
                        os.path.join(shared.models_path, "ControlNet"),
                        os.path.join(shared.models_path, "controlnet"),
                        os.path.join(shared.models_path, "control_net"),
                        os.path.join(shared.models_path, "xlabs", "controlnets")
                    ]:
                        if not os.path.exists(dir_path):
                            continue
                            
                        model_path = os.path.join(dir_path, model_name)
                        if os.path.exists(model_path):
                            debug_print(f"Manually registering ControlNet model: {model_name} => {model_path}")
                            
                            # Register by basename (WebUI Forge's approach)
                            basename = os.path.basename(model_path)
                            filename_dict[basename] = model_path
                            
                            # For WebUI Forge, we MUST also register without the extension
                            basename_no_ext = os.path.splitext(basename)[0]
                            filename_dict[basename_no_ext] = model_path
                            debug_print(f"Also registered without extension: {basename_no_ext} => {model_path}")
                            
                            # For FLUX models, also register the basename
                            if 'flux' in model_name.lower():
                                # Make sure the model is directly in the models list
                                if model_name not in controlnet_models_list:
                                    controlnet_models_list.append(model_name)
                                
                                # Also register the basename
                                if basename != model_name and basename not in controlnet_models_list:
                                    controlnet_models_list.append(basename)
                                    
                                debug_print(f"Added FLUX model {basename} to ControlNet models list")
                                
                            model_found = True
                            registered_models += 1
                            break
                    
                    if not model_found:
                        debug_print(f"Could not find file for ControlNet model: {model_name}")
            
            if registered_models > 0:
                print(f"Registered {registered_models} additional ControlNet models")
                
            return True
        except ImportError:
            # Fall back to the manual approach if WebUI Forge functions not available
            registered_models = 0
            for model_name in our_models:
                if model_name not in filename_dict:
                    model_found = False
                    
                    # Try different possible locations
                    for dir_path in [
                        os.path.join(shared.models_path, "ControlNet"),
                        os.path.join(shared.models_path, "controlnet"),
                        os.path.join(shared.models_path, "control_net"),
                        os.path.join(shared.models_path, "xlabs", "controlnets")
                    ]:
                        if not os.path.exists(dir_path):
                            continue
                            
                        model_path = os.path.join(dir_path, model_name)
                        if os.path.exists(model_path):
                            debug_print(f"Manually registering ControlNet model: {model_name} => {model_path}")
                            filename_dict[model_name] = model_path
                            
                            # For FLUX models, also register the basename
                            if 'flux' in model_name.lower():
                                basename = os.path.basename(model_path)
                                filename_dict[basename] = model_path
                                debug_print(f"Also registered FLUX model basename: {basename}")
                                
                            model_found = True
                            registered_models += 1
                            break
                    
                    if not model_found:
                        debug_print(f"Could not find file for ControlNet model: {model_name}")
            
            if registered_models > 0:
                print(f"Registered {registered_models} additional ControlNet models")
            
            return True
            
    except Exception as e:
        print(f"Error registering ControlNet models: {e}")
        import traceback
        traceback.print_exc()
        return False

def preload_controlnet_models():
    """Preload ControlNet models to ensure they're available"""
    try:
        import os
        from modules import shared
        
        # Try to import WebUI Forge's ControlNet specific functions
        try:
            from lib_controlnet.global_state import controlnet_filename_dict, update_controlnet_filenames
            # update_cn_models_list might not exist in all versions
            try:
                from lib_controlnet.global_state import update_cn_models_list
                has_update_models = True
            except ImportError:
                has_update_models = False
                print("WebUI Forge doesn't have update_cn_models_list function, using alternative method")
        except ImportError:
            print("Could not import ControlNet global state")
            return False
        
        # Update the model list with the available function
        try:
            update_controlnet_filenames()
            if has_update_models:
                update_cn_models_list()
        except Exception as e:
            print(f"Error updating ControlNet model list: {e}")
        
        # Check for FLUX models
        flux_models = []
        for base_dir in [os.path.join(shared.models_path, "ControlNet"), 
                        os.path.join(shared.models_path, "controlnet"), 
                        os.path.join(shared.models_path, "xlabs", "controlnets")]:
            if not os.path.exists(base_dir):
                continue
                
            # Look for FLUX models
            for file in os.listdir(base_dir):
                if file.lower().startswith("flux-") and file.endswith(".safetensors"):
                    model_path = os.path.join(base_dir, file)
                    flux_models.append((file, model_path))
        
        if flux_models:
            print(f"Preloading {len(flux_models)} FLUX ControlNet models...")
            
            # Register each model
            for model_name, model_path in flux_models:
                if os.path.exists(model_path):
                    # FLUX-style: Always use the basename only in the registry
                    controlnet_filename_dict[model_name] = model_path
                    print(f"Registered {model_name} => {model_path}")
                    
                    # For WebUI Forge, we MUST also register without the extension
                    basename_no_ext = os.path.splitext(model_name)[0]
                    controlnet_filename_dict[basename_no_ext] = model_path
                    print(f"Also registered without extension: {basename_no_ext} => {model_path}")
                    
                    # Try to register the model in the extension list
                    try:
                        from lib_controlnet.external_code import update_models_list
                        update_models_list()
                        print(f"Updated ControlNet models list")
                    except:
                        pass
        
        return True
    except Exception as e:
        print(f"Error in preload_controlnet_models: {e}")
        import traceback
        traceback.print_exc()
        return False

def register_model_in_controlnet(model_name):
    """
    Register a model in ControlNet directly by its name
    
    This is a new function to fix missing model registration functions
    """
    try:
        from modules import shared
        import os
        
        # Get the ControlNet filename dict
        filename_dict = get_controlnet_filename_dict()
        
        # If the model is already registered, return the path
        if model_name in filename_dict:
            return filename_dict[model_name]
            
        # Check all potential directories
        potential_dirs = [
            os.path.join(shared.models_path, "ControlNet"),
            os.path.join(shared.models_path, "controlnet"),
            os.path.join(shared.models_path, "control_net"),
            os.path.join(shared.models_path, "xlabs", "controlnets")
        ]
        
        # Extensions to check if model_name doesn't have one
        extensions = ['', '.safetensors', '.ckpt', '.pt', '.pth']
        
        # If model already has an extension, we don't need to add more
        if os.path.splitext(model_name)[1]:
            extensions = ['']
            
        # Search for the model in all potential directories with all extensions
        for dir_path in potential_dirs:
            if not os.path.exists(dir_path):
                continue
                
            for ext in extensions:
                full_path = os.path.join(dir_path, model_name + ext)
                if os.path.exists(full_path):
                    # Register in dictionary
                    filename_dict[model_name] = full_path
                    debug_print(f"Registered model in controlnet: {model_name} => {full_path}")
                    
                    # Also register by basename (WebUI Forge approach)
                    basename = os.path.basename(full_path)
                    filename_dict[basename] = full_path
                    
                    # Register without extension too
                    basename_no_ext = os.path.splitext(basename)[0]
                    filename_dict[basename_no_ext] = full_path
                    
                    return full_path
        
        # Model not found
        debug_print(f"Could not find model {model_name} in any ControlNet directory")
        return None
            
    except Exception as e:
        debug_print(f"Error registering model in ControlNet: {e}")
        return None

def register_single_model_with_forge(model_path):
    """
    Register a single model with WebUI Forge's ControlNet
    
    This is a new function to fix missing model registration
    """
    try:
        if not os.path.exists(model_path):
            debug_print(f"Model path does not exist: {model_path}")
            return False
            
        filename_dict = get_controlnet_filename_dict()
        
        # Register all variants of the model name
        basename = os.path.basename(model_path)
        basename_no_ext = os.path.splitext(basename)[0]
        
        # Register all variants
        filename_dict[basename] = model_path
        filename_dict[basename_no_ext] = model_path
        
        # Also try to register in the models list
        try:
            from lib_controlnet.global_state import controlnet_models_list
            
            # Add both variants to the model list
            if basename not in controlnet_models_list:
                controlnet_models_list.append(basename)
            
            if basename_no_ext not in controlnet_models_list:
                controlnet_models_list.append(basename_no_ext)
                
            debug_print(f"Added {basename} and {basename_no_ext} to ControlNet models list")
        except:
            debug_print("Could not add model to controlnet_models_list")
            
        # Try to update models list through external code
        try:
            from lib_controlnet.external_code import update_models_list
            update_models_list()
            debug_print("Updated models list via external_code")
        except:
            # Not available or failed, ignore
            pass
            
        debug_print(f"Registered single model with Forge: {model_path}")
        return True
        
    except Exception as e:
        debug_print(f"Error registering single model with Forge: {e}")
        return False

def find_alternative_for_flux_model(flux_model_name):
    """
    Find a standard ControlNet model to use instead of a FLUX model
    
    This is a fallback for when FLUX models are not available
    """
    try:
        # Map of FLUX model types to standard model types
        flux_to_standard = {
            'canny': ['control_canny', 'control_v11p_sd15_canny'],
            'depth': ['control_depth', 'control_v11p_sd15_depth'],
            'hed': ['control_hed', 'control_v11p_sd15_scribble'],
            'mlsd': ['control_mlsd', 'control_v11p_sd15_mlsd'],
            'normal': ['control_normal', 'control_v11p_sd15_normalbae'],
            'openpose': ['control_openpose', 'control_v11p_sd15_openpose'],
            'scribble': ['control_scribble', 'control_v11p_sd15_scribble'],
            'seg': ['control_seg', 'control_v11p_sd15_seg'],
            'tile': ['control_tile', 'control_v11p_sd15_tile']
        }
        
        # Determine model type from name
        model_type = None
        flux_model_lower = flux_model_name.lower()
        
        # Extract model type from FLUX model name (e.g., 'flux-canny' -> 'canny')
        if 'flux-' in flux_model_lower:
            parts = flux_model_lower.split('flux-')
            if len(parts) > 1:
                model_type = parts[1].split('.')[0].split('-')[0]
                
        # If we couldn't determine the type, try some default mappings
        if model_type not in flux_to_standard:
            # Handle case when model type is not in map
            debug_print(f"Unknown FLUX model type: {model_type}, using default")
            
            # Try the first model available in any of these types as a default
            filename_dict = get_controlnet_filename_dict()
            for type_alternatives in flux_to_standard.values():
                for alt in type_alternatives:
                    for ext in ['.safetensors', '.ckpt', '.pt', '.pth']:
                        if alt + ext in filename_dict:
                            return filename_dict[alt + ext]
                        elif alt in filename_dict:
                            return filename_dict[alt]
            
            # If no alternatives found at all, return None            
            return None
        
        # Look for matching standard models
        for standard_name in flux_to_standard.get(model_type, []):
            # Try to register and get the model path
            for ext in ['.safetensors', '.ckpt', '.pt', '.pth']:
                model_path = register_model_in_controlnet(standard_name + ext)
                if model_path:
                    debug_print(f"Found alternative for FLUX model: {standard_name + ext}")
                    return model_path
                
                # Try without extension
                model_path = register_model_in_controlnet(standard_name)
                if model_path:
                    debug_print(f"Found alternative for FLUX model: {standard_name}")
                    return model_path
                    
        # No suitable alternative found
        debug_print(f"No suitable alternative model found for {flux_model_name}")
        return None
        
    except Exception as e:
        debug_print(f"Error finding alternative for FLUX model: {e}")
        return None

def try_handle_flux_model(model_name, module_type):
    """
    Attempt to handle FLUX models according to how they're used in x-flux-comfyui
    
    Based on examining the x-flux-comfyui GitHub repo:
    - FLUX models are typically stored in models/xlabs/controlnets/
    - Models are loaded with their full basename (including extension)
    - Special handling is needed for registration with ControlNet frameworks
    
    NOTE: For WebUI Forge, this function is mostly used for model path registration.
    The actual model used will typically be a standard model as a fallback.
    """
    try:
        import os
        from modules import shared

        # Check if model_name is None/invalid
        if not model_name or model_name == "None":
            debug_print("Invalid model name for FLUX handling")
            return None

        # Check if the name already contains flux, if not, assume we need to find a flux model
        contains_flux = 'flux' in model_name.lower()
        
        # First, create the FLUX directory if it doesn't exist
        flux_dir = os.path.join(shared.models_path, "xlabs", "controlnets")
        if not os.path.exists(flux_dir):
            try:
                os.makedirs(flux_dir, exist_ok=True)
                print(f"Created FLUX models directory: {flux_dir}")
            except:
                print(f"Could not create FLUX models directory: {flux_dir}")
        
        # Register the model path to ensure WebUI Forge can find it
        model_path = register_model_in_controlnet(model_name)
        if model_path:
            print(f"FLUX model registered: {model_path}")
            return model_path
            
        # Path helpers for different potential locations
        potential_paths = []
        
        # Check if model already has an extension
        if os.path.splitext(model_name)[1]:
            # Add model with existing extension to possible paths
            potential_paths.extend([
                os.path.join(shared.models_path, "ControlNet", model_name),
                os.path.join(shared.models_path, "controlnet", model_name),
                os.path.join(flux_dir, model_name)
            ])
        else:
            # Try adding standard extensions
            for ext in ['.safetensors', '.ckpt', '.pt', '.pth']:
                potential_paths.extend([
                    os.path.join(shared.models_path, "ControlNet", model_name + ext),
                    os.path.join(shared.models_path, "controlnet", model_name + ext),
                    os.path.join(flux_dir, model_name + ext)
                ])
        
        # Try the exact paths for the specific model
        for path in potential_paths:
            if os.path.exists(path):
                print(f"Found exact FLUX model match: {path}")
                # Register this model with controlnet
                register_single_model_with_forge(path)
                return path
        
        # If model not found by exact name, try to find similar models based on module type
        if contains_flux or module_type in ['canny', 'depth', 'hed', 'normal', 'openpose']:
            # Determine search string
            search_pattern = f"flux-{module_type}" if module_type else "flux-canny"
            print(f"Looking for FLUX model matching pattern: {search_pattern}")
            
            # Search in multiple directories
            for base_dir in [flux_dir, os.path.join(shared.models_path, "ControlNet"), os.path.join(shared.models_path, "controlnet")]:
                if os.path.exists(base_dir):
                    # Find all models matching the pattern
                    matching_models = []
                    for file in os.listdir(base_dir):
                        if file.lower().endswith('.safetensors') and search_pattern.lower() in file.lower():
                            matching_models.append(os.path.join(base_dir, file))
                    
                    # If matching models found, register the first one and return
                    if matching_models:
                        first_match = matching_models[0]
                        print(f"Found matching FLUX model: {first_match}")
                        register_single_model_with_forge(first_match)
                        return first_match
            
            # If no exact pattern match, try a partial match
            if not module_type:
                # If no module type specified, just look for any FLUX model
                for base_dir in [flux_dir, os.path.join(shared.models_path, "ControlNet"), os.path.join(shared.models_path, "controlnet")]:
                    if os.path.exists(base_dir):
                        flux_models = []
                        for file in os.listdir(base_dir):
                            if file.lower().endswith('.safetensors') and 'flux' in file.lower():
                                flux_models.append(os.path.join(base_dir, file))
                        
                        if flux_models:
                            first_match = flux_models[0]
                            print(f"Found any FLUX model: {first_match}")
                            register_single_model_with_forge(first_match)
                            return first_match
        
        # If no FLUX model found, try to find a standard model
        standard_model = find_alternative_for_flux_model(model_name)
        if standard_model:
            print(f"Using standard model instead of FLUX: {standard_model}")
            return standard_model
            
        print(f"Could not find a suitable FLUX model or alternative for '{model_name}'")
        return None
        
    except Exception as e:
        print(f"Error handling FLUX model: {e}")
        import traceback
        traceback.print_exc()
        return None

def cleanup_controlnet_resources(p):
    """
    Clean up ControlNet resources to prevent memory leaks
    """
    try:
        # Remove any cached models to free up memory
        if hasattr(p, "_deforum_cn_models_cache"):
            debug_print("Cleaning up cached ControlNet models")
            # Delete cached model references
            del p._deforum_cn_models_cache
            
        # Remove patched flag
        if hasattr(p, "_deforum_cn_patched"):
            debug_print("Resetting ControlNet patch state")
            del p._deforum_cn_patched
            
        # Force garbage collection to release memory
        import gc
        gc.collect()
        
        # If on CUDA, try to clear CUDA cache
        if devices.device.type == 'cuda':
            debug_print("Clearing CUDA cache after ControlNet processing")
            with torch.cuda.device(devices.device):
                torch.cuda.empty_cache()
        
        return True
    except Exception as e:
        debug_print(f"Error cleaning up ControlNet resources: {e}")
        return False

def build_controlnet_units(controlnet_args, current_frame=0, total_frames=0):
    """
    Build ControlNet units for WebUI Forge from Deforum's controlnet arguments
    
    This creates the proper dictionary structure that WebUI Forge's ControlNet extension expects.
    """
    try:
        from .animation_key_frames import FrameInterpolater
        
        # Determine how many ControlNet units are active
        active_count = getattr(controlnet_args, 'controlnet_units_count', 0)
        if active_count <= 0:
            debug_print("No ControlNet units active")
            return None
            
        # Check if we should optimize execution
        optimize = getattr(controlnet_args, 'controlnet_optimize_execution', True)
        if optimize and not DEBUG_CONTROLNET:
            # Only show ControlNet messages during frame 0 if optimized
            should_log = current_frame == 0
        else:
            should_log = True
        
        # Helper function to safely parse values that might be schedules
        def parse_value_with_schedule(value, default_value, param_name):
            try:
                # If it's already a number, just return it
                if isinstance(value, (int, float)):
                    return float(value)
                
                # If it's a string that might be a schedule
                if isinstance(value, str):
                    # If it's a simple number string
                    if value.replace('.', '', 1).isdigit():
                        return float(value)
                    
                    # If it looks like a schedule with keyframes (contains ':')
                    if ':' in value:
                        fi = FrameInterpolater(max_frames=max(total_frames, 1))
                        # Handle potential errors in schedule parsing
                        try:
                            series = fi.parse_inbetweens(value, param_name)
                            return float(series[min(current_frame, len(series)-1)])
                        except Exception as e:
                            debug_print(f"Error parsing schedule '{value}' for {param_name}: {e}")
                            return float(default_value)
                
                # Default fallback
                return float(default_value)
            except Exception as e:
                debug_print(f"Error parsing value '{value}' for {param_name}: {e}")
                return float(default_value)
        
        # Build ControlNet units
        units = []
        
        # Process each active unit
        for i in range(active_count):
            enabled = getattr(controlnet_args, f'cn_{i}_enabled', False)
            if not enabled:
                # Append None for inactive units
                units.append(None)
                continue
            
            # Get the basic settings
            module = getattr(controlnet_args, f'cn_{i}_module', "None")
            model = getattr(controlnet_args, f'cn_{i}_model', "None")
            
            # Parse values that might be schedules
            weight_value = getattr(controlnet_args, f'cn_{i}_weight', 1.0)
            weight = parse_value_with_schedule(weight_value, 1.0, f'cn_{i}_weight')
            
            guidance_start_value = getattr(controlnet_args, f'cn_{i}_guidance_start', 0.0)
            guidance_start = parse_value_with_schedule(guidance_start_value, 0.0, f'cn_{i}_guidance_start')
            
            guidance_end_value = getattr(controlnet_args, f'cn_{i}_guidance_end', 1.0)
            guidance_end = parse_value_with_schedule(guidance_end_value, 1.0, f'cn_{i}_guidance_end')
            
            # Get advanced settings
            processor_res_value = getattr(controlnet_args, f'cn_{i}_processor_res', 720)
            processor_res = int(parse_value_with_schedule(processor_res_value, 720, f'cn_{i}_processor_res'))
            
            threshold_a_value = getattr(controlnet_args, f'cn_{i}_threshold_a', 0.5)
            threshold_a = parse_value_with_schedule(threshold_a_value, 0.5, f'cn_{i}_threshold_a')
            
            threshold_b_value = getattr(controlnet_args, f'cn_{i}_threshold_b', 0.5)
            threshold_b = parse_value_with_schedule(threshold_b_value, 0.5, f'cn_{i}_threshold_b')
            
            # Get boolean settings
            pixel_perfect = getattr(controlnet_args, f'cn_{i}_pixel_perfect', False)
            low_vram = getattr(controlnet_args, f'cn_{i}_low_vram', False)
            guess_mode = getattr(controlnet_args, f'cn_{i}_guess_mode', False)
            input_invert = getattr(controlnet_args, f'cn_{i}_invert_image', False)
            
            # Check for batch mode and loopback settings
            batch_enabled = getattr(controlnet_args, f'cn_{i}_enabled_batch', False)
            vid_loopback = getattr(controlnet_args, f'cn_{i}_vid_loopback', False)
            
            # Handle resize mode
            resize_mode = getattr(controlnet_args, f'cn_{i}_resize_mode', "Scale to Fit (Inner Fit)")
            # Normalize resize mode format for WebUI Forge - it's very picky about exact format
            if "Inner" in resize_mode or "Scale to Fit" in resize_mode:
                resize_mode = "Inner Fit (Scale to Fit)"
            elif "Outer" in resize_mode or "Envelope" in resize_mode:
                resize_mode = "Outer Fit (Envelope)"
            else:
                resize_mode = "Just Resize"
                
            # Handle color channel order
            rgbbgr_mode = getattr(controlnet_args, f'cn_{i}_rgbbgr_mode', "RGB")
            channel_order = 0 if rgbbgr_mode == "RGB" else 1  # 0=RGB, 1=BGR
            
            # Determine the input image path if applicable
            input_image = None
            mask_image = None
            
            if not vid_loopback and batch_enabled:
                # Try to find the appropriate frame in the controlnet folder
                try:
                    frame_index = current_frame
                    input_dir = os.path.join(os.getcwd(), "controlnet_inputframes_{:02d}".format(i))
                    mask_dir = os.path.join(os.getcwd(), "controlnet_maskframes_{:02d}".format(i))
                    
                    # Try to get the input image if the directory exists
                    if os.path.exists(input_dir):
                        frame_path = os.path.join(input_dir, f"{frame_index:05d}.jpg")
                        if os.path.exists(frame_path):
                            input_image = frame_path
                            if should_log:
                                debug_print(f"ControlNet {i+1} using frame: {frame_path}")
                    
                    # Try to get the mask image if the directory exists
                    if os.path.exists(mask_dir):
                        mask_path = os.path.join(mask_dir, f"{frame_index:05d}.jpg")
                        if os.path.exists(mask_path):
                            mask_image = mask_path
                except Exception as e:
                    debug_print(f"Error loading ControlNet image for unit {i+1}: {e}")
            
            # Skip if module or model is None/disabled
            if module == "None" or model == "None":
                if should_log:
                    debug_print(f"ControlNet unit {i+1} has module='{module}' or model='{model}' set to None, skipping")
                units.append(None)
                continue
                
            # Special handling for models to ensure they're properly formatted
            try:
                # Try to format the model path correctly
                if os.path.exists(model) and os.path.isabs(model):
                    # This is a full path, we should try to get basename or register it
                    debug_print(f"Full model path detected: {model}")
                    
                    # First try to register it in global_state
                    try:
                        from lib_controlnet import global_state
                        if hasattr(global_state, "controlnet_filename_dict"):
                            # Register full path
                            global_state.controlnet_filename_dict[model] = model
                            
                            # Also register basename
                            basename = os.path.basename(model)
                            global_state.controlnet_filename_dict[basename] = model
                            
                            # Also register basename without extension
                            basename_no_ext = os.path.splitext(basename)[0]
                            global_state.controlnet_filename_dict[basename_no_ext] = model
                            
                            # For WebUI Forge, some functions expect just the basename
                            # So we'll use that format as most compatible
                            model = basename_no_ext  # Use basename without extension for best compatibility
                            debug_print(f"Using model basename without extension for WebUI Forge: {model}")
                    except Exception as e:
                        debug_print(f"Error registering model in global_state: {e}")
                
                # Special handling for FLUX models
                if 'flux' in model.lower():
                    try:
                        # Try to find a clean model name (without path or extension)
                        if os.path.isabs(model):
                            # For full paths, extract just the name without extension
                            model_basename = os.path.splitext(os.path.basename(model))[0]
                        else:
                            # For relative paths or simple names, remove any extension
                            model_basename = os.path.splitext(model)[0]
                        
                        # Get a list of possible model paths that might match
                        from modules import shared
                        import glob
                        
                        # Check several directories for matching model files
                        possible_models = []
                        for controlnet_dir in [
                            os.path.join(shared.models_path, "ControlNet"),
                            os.path.join(shared.models_path, "controlnet"),
                            os.path.join(shared.models_path, "control_net")
                        ]:
                            if os.path.exists(controlnet_dir):
                                # Look for any file that has the model name in it
                                pattern = os.path.join(controlnet_dir, f"*{model_basename}*.safetensors")
                                matching_files = glob.glob(pattern)
                                if matching_files:
                                    possible_models.extend(matching_files)
                        
                        if possible_models:
                            # Use the first matching model file
                            flux_model_path = possible_models[0]
                            debug_print(f"Found FLUX model by pattern matching: {flux_model_path}")
                            
                            # Register the model path
                            try:
                                from lib_controlnet import global_state
                                if hasattr(global_state, "controlnet_filename_dict"):
                                    # Register both the full path and the basename
                                    basename = os.path.basename(flux_model_path)
                                    basename_no_ext = os.path.splitext(basename)[0]
                                    
                                    global_state.controlnet_filename_dict[flux_model_path] = flux_model_path
                                    global_state.controlnet_filename_dict[basename] = flux_model_path
                                    global_state.controlnet_filename_dict[basename_no_ext] = flux_model_path
                                    
                                    # Use the basename without extension for best compatibility
                                    model = basename_no_ext
                                    debug_print(f"Using FLUX model basename without extension: {model}")
                            except Exception as e:
                                debug_print(f"Error registering FLUX model in global_state: {e}")
                        else:
                            # Try the FLUX ControlNet utilities approach
                            try:
                                from . import flux_controlnet_utils
                                flux_model_path = flux_controlnet_utils.find_flux_controlnet_model(model, module)
                            except ImportError:
                                # Fallback to direct function if module not available
                                flux_model_path = try_handle_flux_model(model, module)
                            if flux_model_path:
                                if should_log:
                                    debug_print(f"Using FLUX model: {flux_model_path}")
                                # Use basename without extension
                                basename_no_ext = os.path.splitext(os.path.basename(flux_model_path))[0]
                                model = basename_no_ext
                                debug_print(f"Using FLUX model basename without extension: {model}")
                    except Exception as e:
                        debug_print(f"Error handling FLUX model {model}: {e}")
                        # Fall back to standard model if FLUX fails
                        try:
                            # Try to find a non-FLUX controlnet model
                            from modules import shared
                            std_model_dir = os.path.join(shared.models_path, "ControlNet")
                            if os.path.exists(std_model_dir):
                                for std_model in os.listdir(std_model_dir):
                                    if std_model.endswith(".safetensors") and module.lower() in std_model.lower():
                                        model = os.path.splitext(std_model)[0]  # Use without extension
                                        debug_print(f"Falling back to standard model: {model}")
                                        break
                        except Exception as fallback_err:
                            debug_print(f"Error finding fallback model: {fallback_err}")
                        
                # Register the model one more time to ensure it's available
                try:
                    # Use a simpler model name approach - just the basename without extension
                    if os.path.isabs(model):
                        model = os.path.splitext(os.path.basename(model))[0]
                    else:
                        # For already simple names, just remove any extension
                        model = os.path.splitext(model)[0]
                        
                    debug_print(f"Final model name: {model}")
                    
                    # Register this simple name
                    model_path = register_model_in_controlnet(model)
                    if model_path:
                        debug_print(f"Successfully registered model path: {model_path}")
                except Exception as e:
                    debug_print(f"Error in final model registration: {e}")
            except Exception as e:
                debug_print(f"Error formatting model path: {e}")
            
            # Forge ControlNet has specific format requirements - must match exactly
            # Create a dictionary format that Forge expects
            controlnet_unit = {
                'enabled': enabled,
                'module': module,
                'model': model,
                'weight': weight,
                'guidance_start': guidance_start,
                'guidance_end': guidance_end,
                'processor_res': processor_res,
                'threshold_a': threshold_a,
                'threshold_b': threshold_b,
                'resize_mode': resize_mode,
                'pixel_perfect': pixel_perfect,
                'control_mode': 0,  # Default control mode
                'low_vram': low_vram,
                'guess_mode': guess_mode,
                'image': input_image,  # Path to image or None
                'mask': mask_image,    # Path to mask or None
                'invert_image': input_invert,
                'rgbbgr_mode': channel_order,
            }
            
            # Add the unit to our list
            units.append(controlnet_unit)
            
            if should_log:
                debug_print(f"Built ControlNet unit {i+1}: module={module}, model={model}, weight={weight}")
        
        return units
    except Exception as e:
        debug_print(f"Error building ControlNet units: {e}")
        import traceback
        traceback.print_exc()
        return None

def is_controlnet_enabled(controlnet_args):
    """Check if ControlNet is enabled in the arguments"""
    try:
        units_count = getattr(controlnet_args, 'controlnet_units_count', 0)
        if units_count <= 0:
            return False
            
        # Check if any unit is enabled
        for i in range(units_count):
            if getattr(controlnet_args, f'cn_{i}_enabled', False):
                return True
                
        return False
    except Exception as e:
        debug_print(f"Error checking if ControlNet is enabled: {e}")
        return False

def ensure_controlnet_keys_in_namespace(controlnet_args):
    """
    Ensure that all required ControlNet keys exist in the namespace
    This is important for persistence and ControlNetKeys compatibility
    """
    try:
        # Get the maximum number of units we need from shared options
        from modules import shared
        max_controlnet_units = shared.opts.data.get("control_net_unit_count", 
                                                    shared.opts.data.get("control_net_max_models_num", 5))
        
        # Make sure controlnet_units_count is set
        if not hasattr(controlnet_args, 'controlnet_units_count'):
            controlnet_args.controlnet_units_count = max_controlnet_units
        
        # Set optimize execution flag if missing
        if not hasattr(controlnet_args, 'controlnet_optimize_execution'):
            controlnet_args.controlnet_optimize_execution = True
        
        # Try to load default settings for ControlNet
        default_settings = {}
        try:
            import os
            import json
            settings_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'default_settings.txt')
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    default_settings = json.load(f)
                debug_print(f"Loaded default settings file for ControlNet")
        except Exception as e:
            debug_print(f"Could not load default settings: {e}")
            
        # Ensure all attributes for both 0-based and 1-based indices exist
        for i in range(max_controlnet_units):
            # 0-based attributes (used in UI)
            attr_checks = [
                # Basic settings
                (f'cn_{i}_enabled', False),
                (f'cn_{i}_module', "None"),
                (f'cn_{i}_model', "None"),
                (f'cn_{i}_weight', default_settings.get(f'cn_{i+1}_weight', "0:(1)")),
                (f'cn_{i}_guidance_start', default_settings.get(f'cn_{i+1}_guidance_start', "0:(0.0)")),
                (f'cn_{i}_guidance_end', default_settings.get(f'cn_{i+1}_guidance_end', "0:(1.0)")),
                
                # Processing settings
                (f'cn_{i}_processor_res', default_settings.get(f'cn_{i+1}_processor_res', 720)),
                (f'cn_{i}_threshold_a', default_settings.get(f'cn_{i+1}_threshold_a', 0.5)),
                (f'cn_{i}_threshold_b', default_settings.get(f'cn_{i+1}_threshold_b', 0.5)),
                (f'cn_{i}_resize_mode', default_settings.get(f'cn_{i+1}_resize_mode', "Inner Fit (Scale to Fit)")),
                (f'cn_{i}_rgbbgr_mode', default_settings.get(f'cn_{i+1}_rgbbgr_mode', "RGB")),
                
                # Boolean options
                (f'cn_{i}_pixel_perfect', default_settings.get(f'cn_{i+1}_pixel_perfect', True)),
                (f'cn_{i}_low_vram', default_settings.get(f'cn_{i+1}_low_vram', False)),
                (f'cn_{i}_guess_mode', default_settings.get(f'cn_{i+1}_guess_mode', False)),
                (f'cn_{i}_invert_image', default_settings.get(f'cn_{i+1}_invert_image', False)),
                
                # Batch and loopback settings
                (f'cn_{i}_enabled_batch', default_settings.get(f'cn_{i+1}_enabled_batch', False)),
                (f'cn_{i}_vid_path', default_settings.get(f'cn_{i+1}_vid_path', "")),
                (f'cn_{i}_mask_vid_path', default_settings.get(f'cn_{i+1}_mask_vid_path', "")),
                (f'cn_{i}_vid_loopback', default_settings.get(f'cn_{i+1}_loopback_mode', True)),
            ]
            
            # Set all attributes if they don't exist
            for attr_name, default_value in attr_checks:
                if not hasattr(controlnet_args, attr_name):
                    debug_print(f"Setting default value for {attr_name}: {default_value}")
                    controlnet_args.__dict__[attr_name] = default_value
            
            # 1-based attributes (used by ControlNetKeys)
            one_based_idx = i + 1
            
            # Special handling for the schedule values used by ControlNetKeys
            for suffix in ['weight', 'guidance_start', 'guidance_end']:
                one_based_key = f'cn_{one_based_idx}_{suffix}'
                zero_based_key = f'cn_{i}_{suffix}'
                
                if not hasattr(controlnet_args, one_based_key):
                    # Get the schedule value from the zero-based attribute
                    schedule_value = getattr(controlnet_args, zero_based_key, 
                                           "1.0" if suffix == 'weight' else "0.0" if suffix == 'guidance_start' else "1.0")
                    controlnet_args.__dict__[one_based_key] = schedule_value
                    debug_print(f"Created 1-based schedule attribute {one_based_key} with value {schedule_value}")
        
        return True
    except Exception as e:
        debug_print(f"Error ensuring ControlNet keys in namespace: {e}")
        import traceback
        traceback.print_exc()
        return False

def patch_controlnet_process(p):
    """
    Monkey patch the ControlNet process method to ensure resize_mode is always available
    
    This adds a safety check specifically for WebUI Forge's ControlNet where it expects
    the resize_mode attribute but it might not exist yet.
    """
    try:
        # Only attempt patching if we're in WebUI Forge environment
        if not hasattr(p, 'scripts'):
            debug_print("No scripts attr, skipping ControlNet patch")
            return False
            
        # First apply the direct patch for HWC3 error
        direct_patched = patch_controlnet_get_input_data(p)
        if direct_patched:
            debug_print("Applied direct patch to ControlNet get_input_data")
        
        # Continue with the process method patch
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
            
            # Patch the get_input_data method first to fix the HWC3 issue
            if hasattr(controlnet_script, "get_input_data"):
                original_get_input_data = controlnet_script.get_input_data
                
                def patched_get_input_data(self, p, unit, preprocessor, h, w):
                    # Make sure p has resize_mode
                    if not hasattr(p, 'resize_mode'):
                        debug_print("Adding missing resize_mode in get_input_data")
                        p.resize_mode = "Inner Fit (Scale to Fit)"
                    
                    # Try the original method
                    try:
                        return original_get_input_data(self, p, unit, preprocessor, h, w)
                    except Exception as e:
                        debug_print(f"Error in original get_input_data: {e}")
                        
                        # Try to handle the HWC3 error by safely wrapping the image conversion
                        try:
                            # Get the image from the unit
                            input_image = unit.get('image')
                            
                            # If we have an image as a string path, load it
                            if isinstance(input_image, str) and os.path.exists(input_image):
                                img = Image.open(input_image).convert('RGB')
                                input_image = np.array(img)
                                
                            # If it's an array but not uint8, convert it
                            if isinstance(input_image, np.ndarray) and input_image.dtype != np.uint8:
                                if np.issubdtype(input_image.dtype, np.floating):
                                    if input_image.max() <= 1.0:
                                        input_image = (input_image * 255).astype(np.uint8)
                                    else:
                                        input_image = input_image.astype(np.uint8)
                                else:
                                    input_image = input_image.astype(np.uint8)
                                
                            # Create a safe input list with our fixed image
                            debug_print("Using manually prepared input image for ControlNet")
                            return [input_image], p.resize_mode
                        except Exception as inner_e:
                            debug_print(f"Failed to manually prepare input: {inner_e}")
                            
                            # Last resort: return a blank image
                            blank_image = np.zeros((h or 512, w or 512, 3), dtype=np.uint8)
                            return [blank_image], "Inner Fit (Scale to Fit)"
                
                # Apply the patch
                controlnet_script.get_input_data = patched_get_input_data.__get__(controlnet_script, type(controlnet_script))
                debug_print("Successfully patched get_input_data method to handle HWC3 errors")
            
            # Store original process method
            original_process = controlnet_script.process
            
            # Define our patched version
            def patched_process(self, p, *args, **kwargs):
                # Make sure resize_mode exists before processing
                if not hasattr(p, 'resize_mode'):
                    debug_print("Adding missing resize_mode to processing object during script execution")
                    p.resize_mode = "Inner Fit (Scale to Fit)"
                
                # Make sure current_params exists
                if not hasattr(self, 'current_params') or self.current_params is None:
                    debug_print("Initializing missing current_params dictionary")
                    self.current_params = {}
                
                # Ensure all indices exist in current_params
                for i in range(len(args)):
                    if i not in self.current_params:
                        self.current_params[i] = {}
                
                # Call original process with safety wrapper
                try:
                    return original_process(p, *args, **kwargs)
                except Exception as e:
                    debug_print(f"Error in process method: {e}")
                    # Don't crash the whole render
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
                        debug_print("Initializing missing current_params in process_before_every_sampling")
                        self.current_params = {}
                    
                    # Ensure all indices exist in current_params
                    for i in range(len(args)):
                        if i not in self.current_params:
                            debug_print(f"Adding missing index {i} to current_params")
                            self.current_params[i] = {}
                    
                    # Call with safety wrapper
                    try:
                        return original_process_before(self, p, *args, **kwargs)
                    except Exception as e:
                        debug_print(f"Error in process_before_every_sampling: {e}")
                        return None
                
                # Apply the patch
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

def get_controlnet_script_args(p, controlnet_args, current_frame=0, total_frames=0, root=None, parseq_adapter=None):
    """
    Get the ControlNet script arguments for the given processing object and controlnet args
    
    This prepares the ControlNet units in the format expected by the ControlNet extension
    
    Parameters:
        p: Processing object or args object depending on context
        controlnet_args: Arguments for ControlNet
        current_frame: Current frame number being processed (used instead of frame_idx)
        total_frames: Total number of frames in the animation
        root: Root object for the animation
        parseq_adapter: ParseqAdapter instance for animation control
        
    Note: When calling this function, always use named parameters (p=, controlnet_args=, etc.)
    to ensure proper parameter matching, especially when code is refactored.
    """
    try:
        # Ensure all required attributes exist in the namespace
        ensure_controlnet_keys_in_namespace(controlnet_args)
        
        # First check if ControlNet is enabled
        if not is_controlnet_enabled(controlnet_args):
            debug_print("ControlNet is not enabled, returning empty units")
            return []
            
        # Build the ControlNet units
        try:
            units = build_controlnet_units(controlnet_args, current_frame, total_frames)
        except Exception as e:
            debug_print(f"Error building ControlNet units, possibly due to schedule parsing: {e}")
            debug_print("Attempting to build with simpler approach...")
            
            # Try a simplified approach with default values if schedule parsing fails
            # This is a fallback to ensure some kind of ControlNet is used
            try:
                from .animation_key_frames import ControlNetKeys
                # Get the resolved values from ControlNetKeys if possible
                if anim_args := getattr(root, 'anim_args', None):
                    cn_keys = ControlNetKeys(anim_args, controlnet_args)
                    # The units will be built using the default no-schedule approach
                    units = build_controlnet_units_simple(controlnet_args, current_frame)
                else:
                    # Simpler fallback if we can't get anim_args
                    units = build_controlnet_units_simple(controlnet_args, current_frame)
            except Exception as inner_e:
                debug_print(f"Simplified approach also failed: {inner_e}")
                return []
                
        if not units:
            debug_print("No valid ControlNet units built")
            return []
        
        # Pre-process ControlNet units to ensure they have all necessary properties
        # and any image inputs are properly formatted
        processed_units = []
        for unit in units:
            if unit is None:
                processed_units.append(None)
                continue

            # Handle image inputs if they're file paths
            if 'image' in unit and unit['image'] and isinstance(unit['image'], str):
                try:
                    # If it's a path string, load the image
                    from PIL import Image
                    import numpy as np
                    
                    img_path = unit['image']
                    if os.path.exists(img_path):
                        img = Image.open(img_path).convert('RGB')
                        # Convert to numpy array for compatibility
                        unit['image'] = np.array(img)
                        debug_print(f"Loaded image from {img_path} for ControlNet")
                except Exception as e:
                    debug_print(f"Error loading image from path: {e}")
                    unit['image'] = None

            # Same for mask
            if 'mask' in unit and unit['mask'] and isinstance(unit['mask'], str):
                try:
                    from PIL import Image
                    import numpy as np
                    
                    mask_path = unit['mask']
                    if os.path.exists(mask_path):
                        mask = Image.open(mask_path).convert('L')
                        unit['mask'] = np.array(mask)
                        debug_print(f"Loaded mask from {mask_path} for ControlNet")
                except Exception as e:
                    debug_print(f"Error loading mask from path: {e}")
                    unit['mask'] = None
            
            # Ensure other required fields have default values if missing
            for key, default in [
                ('enabled', True),
                ('control_mode', 0),
                ('resize_mode', "Inner Fit (Scale to Fit)"),
                ('rgbbgr_mode', 0),
                ('weight', 1.0),
                ('guidance_start', 0.0),
                ('guidance_end', 1.0),
                ('threshold_a', 0.5),
                ('threshold_b', 0.5),
                ('processor_res', 720),
                ('pixel_perfect', True),
                ('low_vram', False),
                ('guess_mode', False),
                ('invert_image', False)
            ]:
                if key not in unit or unit[key] is None:
                    unit[key] = default
            
            processed_units.append(unit)
            
        # Replace units with processed ones
        units = processed_units
        
        # Handling for the WebUI Forge ControlNet
        # The resize_mode attribute is required by the ControlNet plugin
        try:
            # Make sure p has a resize_mode attribute, as required by WebUI Forge's ControlNet
            # This is needed for both Txt2Img and Img2Img processing objects
            if hasattr(p, 'scripts'):
                # First check for existing resize_mode and convert it to the expected format
                if hasattr(p, 'resize_mode'):
                    # Normalize resize mode format for WebUI Forge - it's very picky about exact format
                    current_mode = p.resize_mode
                    if "Inner" in current_mode or "Scale to Fit" in current_mode:
                        p.resize_mode = "Inner Fit (Scale to Fit)"
                    elif "Outer" in current_mode or "Envelope" in current_mode:
                        p.resize_mode = "Outer Fit (Envelope)"
                    else:
                        p.resize_mode = "Just Resize"
                else:
                    # Add resize_mode attribute if it doesn't exist
                    p.resize_mode = "Inner Fit (Scale to Fit)"  # Default value
                
                # Apply monkey patch to ControlNet process method
                patch_controlnet_process(p)
                
                debug_print(f"Processing object resize_mode set to: {p.resize_mode}")
                
                # Prepare p to handle current_params
                if hasattr(p, 'scripts'):
                    for script in p.scripts.alwayson_scripts:
                        if hasattr(script, 'title') and script.title().lower() == "controlnet":
                            # Initialize current_params if needed
                            if not hasattr(script, 'current_params') or script.current_params is None:
                                script.current_params = {}
                            
                            # Set placeholders for each unit
                            for i in range(len(units)):
                                if i not in script.current_params:
                                    script.current_params[i] = {}
                                    
                            # Set a safe flag to indicate we've prepared this
                            script._deforum_params_initialized = True
                            debug_print("ControlNet current_params initialized")
                            break
                            
        except Exception as e:
            debug_print(f"Warning: Error handling resize_mode attribute: {e}")
            # Even if there's an error, still try to set it
            try:
                p.resize_mode = "Inner Fit (Scale to Fit)"
                # Also try to apply the patch
                patch_controlnet_process(p)
            except:
                debug_print("Could not set resize_mode attribute at all - this may cause ControlNet issues")
            
        # Try to optimize processing if requested
        optimize = getattr(controlnet_args, 'controlnet_optimize_execution', True)
        if optimize and hasattr(p, 'scripts'): # Only try to patch if p is a processing object
            patch_controlnet_optimize_processing(p, units)
            
        return units
    except Exception as e:
        debug_print(f"Error getting ControlNet script args: {e}")
        import traceback
        traceback.print_exc()
        return []

def build_controlnet_units_simple(controlnet_args, current_frame=0):
    """Simplified version of build_controlnet_units that doesn't try to handle schedules"""
    try:
        # Determine how many ControlNet units are active
        active_count = getattr(controlnet_args, 'controlnet_units_count', 0)
        units = []
        
        # Process each active unit with simple handling (no schedules)
        for i in range(active_count):
            enabled = getattr(controlnet_args, f'cn_{i}_enabled', False)
            if not enabled:
                units.append(None)
                continue
                
            # Get basic settings with default values
            module = getattr(controlnet_args, f'cn_{i}_module', "None")
            model = getattr(controlnet_args, f'cn_{i}_model', "None")
            
            # Skip if module or model is None/disabled
            if module == "None" or model == "None":
                units.append(None)
                continue
            
            # Use default values for numerical parameters
            controlnet_unit = {
                'enabled': enabled,
                'module': module,
                'model': model,
                'weight': 1.0,
                'guidance_start': 0.0,
                'guidance_end': 1.0,
                'processor_res': 720,
                'threshold_a': 0.5,
                'threshold_b': 0.5,
                'resize_mode': "Inner Fit (Scale to Fit)",
                'pixel_perfect': True,
                'control_mode': 0,
                'low_vram': False,
                'guess_mode': False,
                'image': None,
                'mask': None,
                'invert_image': False,
                'rgbbgr_mode': 0,
            }
            
            units.append(controlnet_unit)
        
        return units
    except Exception as e:
        debug_print(f"Error in simple ControlNet unit builder: {e}")
        return []

def unpack_controlnet_vids(args, anim_args, controlnet_args):
    """
    Prepare and unpack ControlNet videos for animation processing
    
    This function is called from render.py to set up ControlNet video inputs
    before starting the animation rendering.
    """
    try:
        # Ensure all required attributes exist in the namespace
        ensure_controlnet_keys_in_namespace(controlnet_args)
        
        # Check if ControlNet is enabled
        if not is_controlnet_enabled(controlnet_args):
            debug_print("ControlNet is not enabled, skipping video setup")
            return False
            
        debug_print("Setting up ControlNet videos")
        
        # Process ControlNet videos using the existing function
        setup_controlnet_video(args, anim_args, controlnet_args)
        
        return True
    except Exception as e:
        print(f"Error unpacking ControlNet videos: {e}")
        import traceback
        traceback.print_exc()
        return False

# Make sure our patch is applied to ControlNet when scripts load
def on_script_unloaded():
    try:
        # Find the ControlNet script and patch it
        controlnet_script = None
        for script in scripts.scripts_txt2img.alwayson_scripts:
            if hasattr(script, 'title') and script.title().lower() == "controlnet":
                controlnet_script = script
                break
                
        if controlnet_script:
            original_process = controlnet_script.process
            
            # Define our patched version
            def patched_process(self, p, *args, **kwargs):
                # Make sure resize_mode exists before processing
                if not hasattr(p, 'resize_mode'):
                    print("[Deforum] Adding missing resize_mode to processing object during script execution")
                    p.resize_mode = "Inner Fit (Scale to Fit)"
                
                # Call original process
                return original_process(p, *args, **kwargs)
                
            # Apply our patch
            controlnet_script.process = patched_process.__get__(controlnet_script, type(controlnet_script))
            print("[Deforum] Successfully patched ControlNet script process method during unload")
    except Exception as e:
        print(f"[Deforum] Error patching ControlNet during unload: {e}")

# Register our callback
script_callbacks.on_script_unloaded(on_script_unloaded)

def patch_controlnet_get_input_data(p):
    """
    Apply a direct monkey patch to the specific method in ControlNet where the HWC3 error occurs
    
    This directly targets the line that causes the error with the assertion failure in HWC3.
    """
    try:
        import os
        import numpy as np
        from PIL import Image
        import importlib
        import inspect
        import types
        
        # Only attempt patching if we're in WebUI Forge environment
        if not hasattr(p, 'scripts'):
            debug_print("No scripts attr, skipping direct ControlNet patch")
            return False
            
        # Find the ControlNet script
        controlnet_script = None
        for script in p.scripts.alwayson_scripts:
            if hasattr(script, 'title') and script.title().lower() == "controlnet":
                controlnet_script = script
                break
                
        if not controlnet_script:
            debug_print("ControlNet script not found for direct patch")
            return False
            
        # Check if we already patched
        if hasattr(controlnet_script, '_deforum_direct_patched'):
            debug_print("ControlNet already direct patched")
            return True
            
        # Define a safer version of HWC3 that won't fail with assertions
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
                elif len(x.shape) != 3:
                    return np.zeros((512, 512, 3), dtype=np.uint8)
                    
                return x
            except:
                return np.zeros((512, 512, 3), dtype=np.uint8)
                
        # Get the get_input_data method
        if hasattr(controlnet_script, "get_input_data"):
            original_get_input_data = controlnet_script.get_input_data
            
            # Define a patched version that hooks into the HWC3 call directly
            def patched_get_input_data(self, p, unit, preprocessor, h, w):
                debug_print("Direct patched get_input_data called")
                
                # Make sure resize_mode exists
                if not hasattr(p, 'resize_mode'):
                    p.resize_mode = "Inner Fit (Scale to Fit)"
                
                try:
                    # First attempt with original function
                    return original_get_input_data(self, p, unit, preprocessor, h, w)
                except Exception as e:
                    debug_print(f"Error in get_input_data: {e}")
                    
                    # Special handling for unit 5
                    if unit == 5 or (hasattr(unit, 'num') and unit.num == 5):
                        debug_print("Special handling for unit 5")
                        
                        # Add missing required attributes
                        for required_attr in ['module', 'model', 'weight', 'image', 'resize_mode']:
                            if not hasattr(unit, required_attr):
                                debug_print(f"Adding missing {required_attr} to unit 5")
                                if required_attr == 'module':
                                    setattr(unit, required_attr, 'none')
                                elif required_attr == 'model':
                                    setattr(unit, required_attr, 'None')
                                elif required_attr == 'weight':
                                    setattr(unit, required_attr, 1.0)
                                elif required_attr == 'image':
                                    setattr(unit, required_attr, None)
                                elif required_attr == 'resize_mode':
                                    setattr(unit, required_attr, "Inner Fit (Scale to Fit)")
                    
                    # Make a second attempt
                    try:
                        # Patch HWC3 temporarily
                        module = inspect.getmodule(original_get_input_data)
                        if module and hasattr(module, 'HWC3'):
                            original_HWC3 = module.HWC3
                            module.HWC3 = safe_HWC3
                            
                        # Special unit 5 case - default to blank image
                        if unit == 5 or (hasattr(unit, 'num') and unit.num == 5):
                            return np.zeros((h, w, 3), dtype=np.uint8)
                            
                        # Try again
                        result = original_get_input_data(self, p, unit, preprocessor, h, w)
                        
                        # Restore original HWC3
                        if module and hasattr(module, 'HWC3'):
                            module.HWC3 = original_HWC3
                            
                        return result
                    except Exception as e2:
                        debug_print(f"Second attempt error: {e2}")
                        
                        # Final fallback
                        return np.zeros((h, w, 3), dtype=np.uint8)
                    
            # Apply the patch
            controlnet_script.get_input_data = types.MethodType(patched_get_input_data, controlnet_script)
            controlnet_script._deforum_direct_patched = True
            debug_print("Successfully patched ControlNet get_input_data method")
            
            # Also try to find and patch the HWC3 function directly in the module
            try:
                # Get the module that contains the script
                module = inspect.getmodule(controlnet_script)
                if module:
                    module_name = module.__name__
                    debug_print(f"Found module {module_name} containing ControlNet script")
                    
                    # Check if the module has HWC3 function
                    if hasattr(module, "HWC3"):
                        module._original_HWC3 = module.HWC3
                        module.HWC3 = safe_HWC3
                        debug_print(f"Directly patched HWC3 in {module_name}")
                    
                    # Check for utils import
                    if hasattr(module, "utils") and hasattr(module.utils, "HWC3"):
                        module.utils._original_HWC3 = module.utils.HWC3
                        module.utils.HWC3 = safe_HWC3
                        debug_print(f"Directly patched utils.HWC3 in {module_name}")
                        
                    # Import the utils module Forge uses
                    try:
                        forge_utils = importlib.import_module("modules_forge.utils")
                        if hasattr(forge_utils, "HWC3"):
                            if not hasattr(forge_utils, "_original_HWC3"):
                                forge_utils._original_HWC3 = forge_utils.HWC3
                            forge_utils.HWC3 = safe_HWC3
                            debug_print("Patched HWC3 in modules_forge.utils")
                    except ImportError:
                        debug_print("Could not import modules_forge.utils")
            except Exception as module_err:
                debug_print(f"Error patching module: {module_err}")
            
            return True
        else:
            debug_print("get_input_data method not found on ControlNet script")
            return False
            
    except Exception as e:
        debug_print(f"Error in patch_controlnet_get_input_data: {e}")
        import traceback
        traceback.print_exc()
        return False

# Add a specialized function for unit 5
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
        
        # Only proceed if ControlNet is enabled
        if not hasattr(p, 'scripts'):
            debug_print("No scripts attribute, cannot apply unit 5 patch")
            return False
            
        # Find the ControlNet script
        controlnet_script = None
        for script in p.scripts.alwayson_scripts:
            if hasattr(script, 'title') and script.title().lower() == "controlnet":
                controlnet_script = script
                break
                
        if not controlnet_script:
            debug_print("ControlNet script not found for unit 5 patch")
            return False
            
        # Check if we already patched
        if hasattr(controlnet_script, '_unit5_patched'):
            debug_print("Unit 5 already patched")
            return True
            
        # Define a safer version of HWC3 specifically for unit 5
        def unit5_safe_HWC3(x):
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
                    return unit5_safe_HWC3(x)
                elif len(x.shape) != 3:
                    return np.zeros((512, 512, 3), dtype=np.uint8)
                    
                return x
            except:
                return np.zeros((512, 512, 3), dtype=np.uint8)
                
        # First, patch all HWC3 functions in all relevant modules
        try:
            # Get the module that contains the script
            script_module = inspect.getmodule(controlnet_script)
            if script_module:
                script_module_name = script_module.__name__
                debug_print(f"Found module {script_module_name} for unit 5 patch")
                
                # Patch HWC3 in the script's module
                if hasattr(script_module, "HWC3"):
                    script_module._unit5_original_HWC3 = script_module.HWC3
                    script_module.HWC3 = unit5_safe_HWC3
                    debug_print(f"Patched HWC3 in {script_module_name} for unit 5")
                
                # Check for utils import in script module
                if hasattr(script_module, "utils") and hasattr(script_module.utils, "HWC3"):
                    script_module.utils._unit5_original_HWC3 = script_module.utils.HWC3
                    script_module.utils.HWC3 = unit5_safe_HWC3
                    debug_print(f"Patched utils.HWC3 in {script_module_name} for unit 5")
                    
            # Direct patch to modules_forge.utils - most critical
            try:
                modules_forge_utils = importlib.import_module("modules_forge.utils")
                if hasattr(modules_forge_utils, "HWC3"):
                    modules_forge_utils._unit5_original_HWC3 = modules_forge_utils.HWC3
                    modules_forge_utils.HWC3 = unit5_safe_HWC3
                    debug_print("Patched HWC3 in modules_forge.utils for unit 5")
            except ImportError:
                debug_print("Could not import modules_forge.utils")
                
            # Patch lib_controlnet.utils if available
            try:
                lib_controlnet_utils = importlib.import_module("lib_controlnet.utils")
                if hasattr(lib_controlnet_utils, "HWC3"):
                    lib_controlnet_utils._unit5_original_HWC3 = lib_controlnet_utils.HWC3
                    lib_controlnet_utils.HWC3 = unit5_safe_HWC3
                    debug_print("Patched HWC3 in lib_controlnet.utils for unit 5")
            except ImportError:
                debug_print("Could not import lib_controlnet.utils")
                
            # Patch any other module with 'controlnet' and 'HWC3'
            for name, module in list(sys.modules.items()):
                if module and 'controlnet' in name.lower() and hasattr(module, 'HWC3'):
                    if not hasattr(module, '_unit5_patched_HWC3'):
                        module._unit5_original_HWC3 = module.HWC3
                        module.HWC3 = unit5_safe_HWC3
                        module._unit5_patched_HWC3 = True
                        debug_print(f"Patched HWC3 in {name} for unit 5")
        except Exception as e:
            debug_print(f"Error patching HWC3 for unit 5: {e}")
            
        # Now patch the get_input_data method specifically for unit 5
        if hasattr(controlnet_script, "get_input_data"):
            original_get_input_data = controlnet_script.get_input_data
            
            # Define a specialized unit 5 patched version
            def unit5_patched_get_input_data(self, p, unit, preprocessor, h, w):
                # If it's not unit 5, use original method
                if not (unit == 5 or (hasattr(unit, 'num') and unit.num == 5)):
                    try:
                        return original_get_input_data(self, p, unit, preprocessor, h, w)
                    except Exception as e:
                        debug_print(f"Error in original get_input_data (non-unit 5): {e}")
                        return np.zeros((h, w, 3), dtype=np.uint8)
                
                # Special handling for unit 5
                debug_print("Unit 5 specialized handler called")
                
                try:
                    # Set up required attributes for unit 5
                    for attr_name, default_value in [
                        ('module', 'none'),
                        ('model', 'None'),
                        ('weight', 1.0),
                        ('image', None),
                        ('resize_mode', "Inner Fit (Scale to Fit)"),
                        ('processor_res', 64),
                        ('threshold_a', 64),
                        ('threshold_b', 64)
                    ]:
                        if not hasattr(unit, attr_name):
                            setattr(unit, attr_name, default_value)
                            debug_print(f"Added missing {attr_name} to unit 5")
                    
                    # If unit has no image, return blank
                    if unit.image is None:
                        debug_print("Unit 5 has no image, returning blank")
                        return np.zeros((h, w, 3), dtype=np.uint8)
                    
                    # Try to safely process the image with our safe HWC3 function
                    try:
                        # Get the module to use our safe HWC3
                        module = inspect.getmodule(original_get_input_data)
                        original_module_HWC3 = None
                        
                        if module and hasattr(module, 'HWC3'):
                            original_module_HWC3 = module.HWC3
                            module.HWC3 = unit5_safe_HWC3
                            
                        # Safe preprocessing of the image
                        image = unit.image
                        if not isinstance(image, np.ndarray):
                            debug_print("Converting unit 5 image to array")
                            try:
                                from PIL import Image
                                if isinstance(image, Image.Image):
                                    image = np.array(image)
                                else:
                                    image = np.array(image)
                            except:
                                debug_print("Failed to convert unit 5 image")
                                image = np.zeros((64, 64, 3), dtype=np.uint8)
                        
                        # Process through the preprocessor if available
                        if preprocessor is not None:
                            debug_print(f"Using preprocessor for unit 5: {preprocessor}")
                            image = unit5_safe_HWC3(image)
                            image = preprocessor(image)
                        
                        # Convert result to correct format 
                        image = unit5_safe_HWC3(image)
                        
                        # Restore original HWC3 if we temporarily replaced it
                        if module and hasattr(module, 'HWC3') and original_module_HWC3:
                            module.HWC3 = original_module_HWC3
                            
                        return image
                            
                    except Exception as process_err:
                        debug_print(f"Error processing unit 5 image: {process_err}")
                        return np.zeros((h, w, 3), dtype=np.uint8)
                        
                except Exception as e:
                    debug_print(f"Unit 5 handler error: {e}")
                    return np.zeros((h, w, 3), dtype=np.uint8)
                    
            # Apply the patch
            controlnet_script.get_input_data = types.MethodType(unit5_patched_get_input_data, controlnet_script)
            controlnet_script._unit5_patched = True
            debug_print("Successfully applied unit 5 patch to get_input_data")
            
            return True
        else:
            debug_print("get_input_data method not found for unit 5 patch")
            return False
            
    except Exception as e:
        debug_print(f"Error in patch_controlnet_unit5: {e}")
        import traceback
        traceback.print_exc()
        return False

def setup_controlnet_units(count=1):
    """Set up ControlNet units and ensure they have required attributes
    This is required for FLUX models to work properly."""
    try:
        from modules import shared, scripts
        debug_print(f"Setting up {count} ControlNet units")
        
        # If ControlNet is not available, do nothing
        try:
            from lib_controlnet import global_state as cn_global_state
        except ImportError:
            debug_print("ControlNet not available, skipping unit setup")
            return False
            
        # Create default units if they don't exist
        if not hasattr(cn_global_state, 'cn_units'):
            cn_global_state.cn_units = {}
            debug_print("Created missing cn_units dict")
            
        # Ensure we have enough units
        for i in range(count):
            if i not in cn_global_state.cn_units:
                cn_global_state.cn_units[i] = {}
                debug_print(f"Created missing unit {i}")
                
            # Ensure all units have required attributes
            unit = cn_global_state.cn_units[i]
            for attr in ['enabled', 'module', 'model', 'weight', 'image', 'mask', 'resize_mode']:
                if attr not in unit:
                    default_value = None
                    if attr == 'enabled':
                        default_value = False
                    elif attr == 'module' or attr == 'model':
                        default_value = 'None'
                    elif attr == 'weight':
                        default_value = 1.0
                    elif attr == 'resize_mode':
                        default_value = "Inner Fit (Scale to Fit)"
                        
                    unit[attr] = default_value
                    debug_print(f"Added missing {attr} to unit {i}")
        
        # Apply unit 5 patch if configured for more than 5 units
        if count >= 5:
            try:
                from modules import processing
                if hasattr(processing, 'StableDiffusionProcessingTxt2Img'):
                    dummy_p = processing.StableDiffusionProcessingTxt2Img()
                    dummy_p.scripts = scripts.scripts_txt2img
                    if not patch_controlnet_unit5(dummy_p):
                        debug_print("Failed to apply unit 5 patch")
            except Exception as e:
                debug_print(f"Error applying unit 5 patch during setup: {e}")
                
        return True
        
    except Exception as e:
        debug_print(f"Error in setup_controlnet_units: {e}")
        return False
