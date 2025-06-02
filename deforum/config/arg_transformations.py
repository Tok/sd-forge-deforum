"""
Argument Transformations - Fixed for Missing Components
Contains argument processing, transformations, and component management logic
"""

import json
import os
import time
from types import SimpleNamespace
import pathlib
from .general_utils import substitute_placeholders, get_deforum_version, clean_gradio_path_strings
from ..utils import log_utils
from ..utils.color_constants import BOLD, CYAN, RESET_COLOR

# Conditional imports
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    from modules.processing import get_fixed_seed
    WEBUI_AVAILABLE = True
except ImportError:
    WEBUI_AVAILABLE = False
    
    def get_fixed_seed(seed):
        """Fallback seed function."""
        if seed == -1:
            import random
            return random.randint(0, 2**32 - 1)
        return seed

try:
    from .arg_defaults import RootArgs, DeforumAnimArgs, DeforumArgs, ParseqArgs, WanArgs, DeforumOutputArgs
    from .arg_validation import sanitize_strength, sanitize_seed
    ARG_MODULES_AVAILABLE = True
except ImportError:
    ARG_MODULES_AVAILABLE = False

try:
    from ..integrations.controlnet.core_integration import controlnet_component_names
    CONTROLNET_AVAILABLE = True
except ImportError:
    CONTROLNET_AVAILABLE = False
    
    def controlnet_component_names():
        """Fallback controlnet function."""
        return []

try:
    from .general_utils import substitute_placeholders
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    
    def substitute_placeholders(text, arg_list, base_path):
        """Fallback substitute function."""
        return text


def get_component_names():
    """Get all component names for UI binding - FIXED VERSION."""
    # Define essential components that should always exist
    essential_components = [
        # Batch mode components (CRITICAL FIX)
        'override_settings_with_file', 
        'custom_settings_file',
        
        # Basic generation components
        'seed', 'steps', 'sampler', 'scheduler', 'checkpoint', 'clip_skip',
        'W', 'H', 'strength', 'cfg_scale', 'distilled_cfg_scale', 'tiling',
        'restore_faces', 'seed_resize_from_w', 'seed_resize_from_h',
        'noise_multiplier', 'ddim_eta', 'ancestral_eta', 'overlay_mask',
        'mask_file', 'init_image', 'init_image_box', 'use_init', 'use_mask',
        'invert_mask', 'mask_brightness_adjust', 'mask_contrast_adjust',
        'mask_overlay_blur', 'fill', 'full_res_mask', 'full_res_mask_padding',
        'reroll_blank_frames', 'prompt', 'negative_prompt', 'subseed',
        'subseed_strength', 'seed_behavior', 'seed_iter_N', 'use_areas',
        'save_settings', 'save_sample', 'display_samples', 'save_sample_per_step',
        'show_sample_per_step', 'override_these_with_webui', 'batch_name',
        'filename_format', 'animation_mode', 'max_frames', 'border', 
        'angle', 'zoom', 'translation_x', 'translation_y', 'translation_z', 
        'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z',
        'flip_2d_perspective', 'perspective_flip_theta', 'perspective_flip_phi',
        'perspective_flip_gamma', 'perspective_flip_fv', 'noise_schedule',
        'strength_schedule', 'contrast_schedule', 'cfg_scale_schedule',
        'distilled_cfg_scale_schedule', 'steps_schedule', 'seed_schedule',
        'motion_preview_mode', 'motion_preview_length', 'motion_preview_step',
        
        # Animation prompt components
        'animation_prompts', 'animation_prompts_positive', 'animation_prompts_negative',
        
        # Components that are commonly missing but expected
        'optical_flow_cadence', 'cadence_flow_factor_schedule', 
        'optical_flow_redo_generation', 'redo_flow_factor_schedule',
        'diffusion_redo', 'enable_perspective_flip', 'depth_warp_msg_html',
        'color_force_grayscale', 'noise_type', 'perlin_octaves', 'perlin_persistence',
        'color_coherence', 'color_coherence_video_every_N_frames', 'color_coherence_image_path',
        'depth_algorithm', 'midas_weight', 'make_gif', 'ncnn_upscale_model', 'ncnn_upscale_factor',
        'skip_video_creation', 'frame_interp_slow_mo_amount', 'frame_interp_x_amount',
        'aspect_ratio_use_old_formula', 'aspect_ratio_schedule'
    ]
    
    # Add dynamic components if available
    try:
        if ARG_MODULES_AVAILABLE:
            essential_components.extend(DeforumAnimArgs().keys())
            essential_components.extend(DeforumArgs().keys()) 
            essential_components.extend(DeforumOutputArgs().keys())
            essential_components.extend(ParseqArgs().keys())
            essential_components.extend(WanArgs().keys())
    except:
        pass
    
    # Add ControlNet components if available
    try:
        if CONTROLNET_AVAILABLE:
            essential_components.extend(controlnet_component_names())
    except:
        pass
    
    # Remove duplicates while preserving order
    seen = set()
    unique_components = []
    for item in essential_components:
        if item not in seen:
            seen.add(item)
            unique_components.append(item)
    
    return unique_components


def get_settings_component_names():
    """Get settings component names."""
    return [name for name in get_component_names()]


def pack_args(args_dict, keys_function):
    """Pack arguments using specified keys function - with fallback for missing keys."""
    result = {}
    for name in keys_function():
        if name in args_dict:
            result[name] = args_dict[name]
        else:
            # Provide safe defaults for missing components
            if name in ['override_settings_with_file']:
                result[name] = False
            elif name in ['custom_settings_file']:
                result[name] = None
            else:
                result[name] = None
    return result


def create_namespace_from_dict(args_dict: dict, keys_function) -> SimpleNamespace:
    """Create SimpleNamespace from dictionary using keys function.
    
    Args:
        args_dict: Source dictionary
        keys_function: Function returning valid keys
        
    Returns:
        SimpleNamespace object with filtered arguments
    """
    filtered_args = pack_args(args_dict, keys_function)
    return SimpleNamespace(**filtered_args)


def process_animation_prompts(args_dict_main: dict, root: SimpleNamespace) -> None:
    """Process and transform animation prompts.
    
    Args:
        args_dict_main: Main arguments dictionary
        root: Root namespace to update
    """
    # Parse animation prompts JSON - with fallback
    try:
        animation_prompts_raw = args_dict_main.get('animation_prompts', '{"0": "a beautiful landscape"}')
        if isinstance(animation_prompts_raw, str):
            root.animation_prompts = json.loads(animation_prompts_raw)
        else:
            root.animation_prompts = animation_prompts_raw if animation_prompts_raw else {"0": "a beautiful landscape"}
    except (json.JSONDecodeError, TypeError):
        print("⚠️ Warning: Invalid animation prompts JSON, using default")
        root.animation_prompts = {"0": "a beautiful landscape"}
    
    # Get positive and negative prompts - with fallbacks
    positive_prompts = args_dict_main.get('animation_prompts_positive', '')
    negative_prompts = args_dict_main.get('animation_prompts_negative', '')
    
    # Clean negative prompts
    if negative_prompts:
        negative_prompts = negative_prompts.replace('--neg', '')
    
    # Create prompt keyframes
    root.prompt_keyframes = [key for key in root.animation_prompts.keys()]
    
    # Combine prompts with proper negative prompt formatting
    if positive_prompts or negative_prompts:
        root.animation_prompts = {
            key: f"{positive_prompts} {val} {'' if '--neg' in val else '--neg'} {negative_prompts}"
            for key, val in root.animation_prompts.items()
        }


def process_seed_settings(args: SimpleNamespace, root: SimpleNamespace) -> None:
    """Process seed settings and store raw seed.
    
    Args:
        args: Arguments namespace
        root: Root namespace
    """
    # Store raw seed before processing
    if hasattr(args, 'seed') and args.seed == -1:
        root.raw_seed = -1
    else:
        root.raw_seed = getattr(args, 'seed', -1)
    
    # Get fixed seed
    if hasattr(args, 'seed'):
        args.seed = get_fixed_seed(args.seed)
    
    # Update raw seed if it wasn't random
    if root.raw_seed != -1:
        root.raw_seed = args.seed


def setup_output_directory(args: SimpleNamespace, root: SimpleNamespace, p, current_arg_list: list) -> None:
    """Setup output directory and batch name processing.
    
    Args:
        args: Arguments namespace
        root: Root namespace
        p: Processing object
        current_arg_list: List of argument objects for substitution
    """
    # Store raw batch name
    root.raw_batch_name = getattr(args, 'batch_name', 'Deforum')
    
    # Process batch name with substitutions
    full_base_folder_path = os.path.join(os.getcwd(), p.outpath_samples)
    args.batch_name = substitute_placeholders(args.batch_name, current_arg_list, full_base_folder_path)
    
    # Setup output directory
    args.outdir = os.path.join(p.outpath_samples, str(args.batch_name))
    args.outdir = os.path.join(os.getcwd(), args.outdir)
    args.outdir = os.path.realpath(args.outdir)
    
    # Create directory
    os.makedirs(args.outdir, exist_ok=True)


def setup_default_image(args: SimpleNamespace, root: SimpleNamespace) -> None:
    """Setup default image for processing.
    
    Args:
        args: Arguments namespace
        root: Root namespace
    """
    if not PIL_AVAILABLE:
        return
        
    try:
        default_img_path = os.path.join(pathlib.Path(__file__).parent.absolute(), '114763196.jpg')
        if os.path.exists(default_img_path):
            default_img = Image.open(default_img_path)
            assert default_img is not None
            W = getattr(args, 'W', 512)
            H = getattr(args, 'H', 512)
            default_img = default_img.resize((W, H))
            root.default_img = default_img
    except Exception as e:
        print(f"⚠️ Warning: Could not load default image: {e}")


def process_init_image_settings(args: SimpleNamespace, anim_args: SimpleNamespace) -> None:
    """Process initial image settings.
    
    Args:
        args: Arguments namespace
        anim_args: Animation arguments namespace
    """
    use_init = getattr(args, 'use_init', False)
    hybrid_use_init_image = getattr(anim_args, 'hybrid_use_init_image', False)
    
    if not use_init and not hybrid_use_init_image:
        args.init_image = None
        args.init_image_box = None


def create_additional_substitutions() -> SimpleNamespace:
    """Create additional substitution variables.
    
    Returns:
        SimpleNamespace with date and time substitutions
    """
    return SimpleNamespace(
        date=time.strftime('%Y%m%d'),
        time=time.strftime('%H%M%S')
    )


def process_args(args_dict, index=0):
    """
    Process and transform arguments for Deforum execution - FIXED VERSION.
    Note: FreeU and Kohya HR Fix functionality has been removed.
    
    Returns:
        tuple: (args_loaded_ok, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args, wan_args)
    """
    try:
        # Create safe defaults for missing batch mode components
        if 'override_settings_with_file' not in args_dict:
            args_dict['override_settings_with_file'] = False
        if 'custom_settings_file' not in args_dict:
            args_dict['custom_settings_file'] = None
            
        # Extract basic arguments with fallbacks
        args = SimpleNamespace(**args_dict.get('args', {}))
        anim_args = SimpleNamespace(**args_dict.get('anim_args', {}))
        video_args = SimpleNamespace(**args_dict.get('video_args', {}))
        parseq_args = SimpleNamespace(**args_dict.get('parseq_args', {}))
        loop_args = SimpleNamespace(**args_dict.get('loop_args', {}))
        controlnet_args = SimpleNamespace(**args_dict.get('controlnet_args', {}))
        wan_args = SimpleNamespace(**args_dict.get('wan_args', {}))
        
        # Create root object
        root = SimpleNamespace()
        root.timestring = args_dict.get('timestring', time.strftime('%Y%m%d_%H%M%S'))
        root.animation_prompts = args_dict.get('animation_prompts', {"0": "a beautiful landscape"})
        root.models_path = args_dict.get('models_path', 'models')
        root.device = args_dict.get('device', 'cuda')
        root.half_precision = args_dict.get('half_precision', True)
        
        # Process paths and placeholders
        if hasattr(args, 'outdir') and args.outdir:
            args.outdir = substitute_placeholders(
                args.outdir, 
                [args, anim_args, video_args], 
                os.path.dirname(args.outdir)
            )
            args.outdir = clean_gradio_path_strings(args.outdir)
        
        # Set defaults for missing fields
        if not hasattr(args, 'outdir') or not args.outdir:
            args.outdir = os.path.join("outputs", "deforum")
            
        if not hasattr(anim_args, 'max_frames'):
            anim_args.max_frames = 100
        if not hasattr(anim_args, 'animation_mode'):
            anim_args.animation_mode = '2D'
        if not hasattr(video_args, 'fps'):
            video_args.fps = 15
        
        # Ensure output directory exists
        os.makedirs(args.outdir, exist_ok=True)
        
        log_utils.info(f"Arguments processed successfully for index {index}")
        
        return True, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args, wan_args
        
    except Exception as e:
        log_utils.error(f"Error processing arguments: {e}")
        return False, None, None, None, None, None, None, None, None


def validate_args(args, anim_args, video_args):
    """
    Validate argument consistency and requirements.
    Note: FreeU and Kohya HR Fix functionality has been removed.
    """
    errors = []
    
    # Check output directory
    if not hasattr(args, 'outdir') or not args.outdir:
        errors.append("Output directory is required")
    
    # Check frame count
    if not hasattr(anim_args, 'max_frames') or anim_args.max_frames <= 0:
        errors.append("Max frames must be greater than 0")
    
    # Check animation mode
    valid_modes = ['2D', '3D', 'Video Input', 'Interpolation', 'Wan Video']
    if not hasattr(anim_args, 'animation_mode') or anim_args.animation_mode not in valid_modes:
        errors.append(f"Animation mode must be one of: {valid_modes}")
    
    # Check FPS
    if not hasattr(video_args, 'fps') or video_args.fps <= 0:
        errors.append("FPS must be greater than 0")
    
    return errors


def create_default_args():
    """
    Create default argument objects.
    Note: FreeU and Kohya HR Fix functionality has been removed.
    """
    args = SimpleNamespace()
    args.outdir = ""
    args.seed = -1
    args.W = 512
    args.H = 512
    args.steps = 20
    args.cfg_scale = 7.0
    args.sampler = "euler"
    args.scheduler = "normal"
    
    anim_args = SimpleNamespace()
    anim_args.max_frames = 100
    anim_args.animation_mode = "2D"
    anim_args.border = "replicate"
    
    video_args = SimpleNamespace()
    video_args.fps = 15
    video_args.skip_video_creation = False
    
    parseq_args = SimpleNamespace()
    parseq_args.parseq_manifest = ""
    
    loop_args = SimpleNamespace()
    loop_args.use_looper = False
    
    controlnet_args = SimpleNamespace()
    
    wan_args = SimpleNamespace()
    
    return args, anim_args, video_args, parseq_args, loop_args, controlnet_args, wan_args
