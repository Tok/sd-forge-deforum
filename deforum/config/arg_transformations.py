"""
Argument Transformations
Contains argument processing, transformations, and component management logic
"""

import json
import os
import time
from types import SimpleNamespace
import pathlib

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
    from .deforum_controlnet import controlnet_component_names
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


def process_args(args_dict_main, run_id):
    """Process arguments from main dictionary.
    
    Args:
        args_dict_main: Main arguments dictionary
        run_id: Run identifier
        
    Returns:
        Tuple of processed argument objects
    """
    from .settings import load_args
    
    # Extract settings
    override_settings_with_file = args_dict_main['override_settings_with_file']
    custom_settings_file = args_dict_main['custom_settings_file']
    p = args_dict_main['p']

    # Create immutable argument objects using functional approach
    root = SimpleNamespace(**RootArgs())
    args = create_namespace_from_dict(args_dict_main, DeforumArgs)
    anim_args = create_namespace_from_dict(args_dict_main, DeforumAnimArgs)
    video_args = create_namespace_from_dict(args_dict_main, DeforumOutputArgs)
    parseq_args = create_namespace_from_dict(args_dict_main, ParseqArgs)
    controlnet_args = create_namespace_from_dict(args_dict_main, controlnet_component_names)
    wan_args = create_namespace_from_dict(args_dict_main, WanArgs)

    # Process animation prompts
    process_animation_prompts(args_dict_main, root)

    # Load settings from file if requested
    args_loaded_ok = True
    if override_settings_with_file:
        args_loaded_ok = load_args(
            args_dict_main, args, anim_args, parseq_args, 
            controlnet_args, video_args, custom_settings_file, root, run_id
        )

    # Process seed settings
    process_seed_settings(args, root)
    
    # Set timestring
    root.timestring = time.strftime('%Y%m%d%H%M%S')
    
    # Sanitize strength
    args.strength = sanitize_strength(args.strength)
    
    # Setup prompt information
    args.prompts = json.loads(args_dict_main['animation_prompts'])
    args.positive_prompts = args_dict_main['animation_prompts_positive']
    args.negative_prompts = args_dict_main['animation_prompts_negative']

    # Process init image settings
    process_init_image_settings(args, anim_args)

    # Create substitution variables
    additional_substitutions = create_additional_substitutions()
    current_arg_list = [args, anim_args, video_args, parseq_args, root, additional_substitutions]
    
    # Setup output directory
    setup_output_directory(args, root, p, current_arg_list)
    
    # Setup default image
    setup_default_image(args, root)

    # Create placeholder namespaces for backward compatibility
    freeu_args = SimpleNamespace()
    kohya_hrfix_args = SimpleNamespace()
    loop_args = SimpleNamespace()

    return (args_loaded_ok, root, args, anim_args, video_args, parseq_args, 
            loop_args, controlnet_args, freeu_args, kohya_hrfix_args, wan_args) 