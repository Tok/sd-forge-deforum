"""
Argument Transformations
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
    """Get all component names for UI binding."""
    return ['override_settings_with_file', 'custom_settings_file', *DeforumAnimArgs().keys(), 'animation_prompts',
            'animation_prompts_positive', 'animation_prompts_negative',
            *DeforumArgs().keys(), *DeforumOutputArgs().keys(), *ParseqArgs().keys(),
            *controlnet_component_names(), *WanArgs().keys()]


def get_settings_component_names():
    """Get settings component names."""
    return [name for name in get_component_names()]


def pack_args(args_dict, keys_function):
    """Pack arguments using specified keys function."""
    return {name: args_dict[name] for name in keys_function()}


def create_namespace_from_dict(args_dict: dict, keys_function) -> SimpleNamespace:
    """Create SimpleNamespace from dictionary using keys function.
    
    Args:
        args_dict: Source dictionary
        keys_function: Function returning valid keys
        
    Returns:
        SimpleNamespace object with filtered arguments
    """
    filtered_args = {name: args_dict[name] for name in keys_function() if name in args_dict}
    return SimpleNamespace(**filtered_args)


def process_animation_prompts(args_dict_main: dict, root: SimpleNamespace) -> None:
    """Process and transform animation prompts.
    
    Args:
        args_dict_main: Main arguments dictionary
        root: Root namespace to update
    """
    # Parse animation prompts JSON
    root.animation_prompts = json.loads(args_dict_main['animation_prompts'])
    
    # Get positive and negative prompts
    positive_prompts = args_dict_main['animation_prompts_positive']
    negative_prompts = args_dict_main['animation_prompts_negative']
    
    # Clean negative prompts
    negative_prompts = negative_prompts.replace('--neg', '')
    
    # Create prompt keyframes
    root.prompt_keyframes = [key for key in root.animation_prompts.keys()]
    
    # Combine prompts with proper negative prompt formatting
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
    if args.seed == -1:
        root.raw_seed = -1
    
    # Get fixed seed
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
    root.raw_batch_name = args.batch_name
    
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
    default_img_path = os.path.join(pathlib.Path(__file__).parent.absolute(), '114763196.jpg')
    default_img = Image.open(default_img_path)
    assert default_img is not None
    default_img = default_img.resize((args.W, args.H))
    root.default_img = default_img


def process_init_image_settings(args: SimpleNamespace, anim_args: SimpleNamespace) -> None:
    """Process initial image settings.
    
    Args:
        args: Arguments namespace
        anim_args: Animation arguments namespace
    """
    if not args.use_init and not anim_args.hybrid_use_init_image:
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
    Process and transform arguments for Deforum execution.
    Note: FreeU and Kohya HR Fix functionality has been removed.
    
    Returns:
        tuple: (args_loaded_ok, root, args, anim_args, video_args, parseq_args, loop_args, controlnet_args, wan_args)
    """
    try:
        # Extract basic arguments
        args = SimpleNamespace(**args_dict.get('args', {}))
        anim_args = SimpleNamespace(**args_dict.get('anim_args', {}))
        video_args = SimpleNamespace(**args_dict.get('video_args', {}))
        parseq_args = SimpleNamespace(**args_dict.get('parseq_args', {}))
        loop_args = SimpleNamespace(**args_dict.get('loop_args', {}))
        controlnet_args = SimpleNamespace(**args_dict.get('controlnet_args', {}))
        wan_args = SimpleNamespace(**args_dict.get('wan_args', {}))
        
        # Create root object
        root = SimpleNamespace()
        root.timestring = args_dict.get('timestring', '')
        root.animation_prompts = args_dict.get('animation_prompts', {})
        root.models_path = args_dict.get('models_path', '')
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
        
        # Validate required fields
        if not hasattr(args, 'outdir') or not args.outdir:
            log_utils.error("Output directory not specified")
            return False, None, None, None, None, None, None, None, None
        
        # Set defaults for missing fields
        if not hasattr(anim_args, 'max_frames'):
            anim_args.max_frames = 100
        if not hasattr(anim_args, 'animation_mode'):
            anim_args.animation_mode = '2D'
        if not hasattr(video_args, 'fps'):
            video_args.fps = 15
        
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
    if not args.outdir:
        errors.append("Output directory is required")
    
    # Check frame count
    if anim_args.max_frames <= 0:
        errors.append("Max frames must be greater than 0")
    
    # Check animation mode
    valid_modes = ['2D', '3D', 'Video Input', 'Interpolation']
    if anim_args.animation_mode not in valid_modes:
        errors.append(f"Animation mode must be one of: {valid_modes}")
    
    # Check FPS
    if video_args.fps <= 0:
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