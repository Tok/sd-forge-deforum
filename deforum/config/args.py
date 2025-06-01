"""
Args Module
Legacy argument handling for backward compatibility during refactoring
"""

from types import SimpleNamespace
from .arguments import (
    DeforumArgs, DeforumAnimArgs, DeforumOutputArgs, 
    ParseqArgs, WanArgs, RootArgs
)
from .data_models import UIDefaults


def set_arg_lists():
    """Create argument namespaces for UI components.
    
    This function provides backward compatibility during the refactoring process
    by converting functional argument dictionaries to SimpleNamespace objects.
    
    Returns:
        Tuple of SimpleNamespace objects for UI components
    """
    # Create immutable UI defaults using functional args system
    ui_defaults = UIDefaults.create_defaults()
    
    # Convert to individual namespace objects for backward compatibility during transition
    d = SimpleNamespace(**ui_defaults.deforum_args)
    da = SimpleNamespace(**ui_defaults.animation_args) 
    dp = SimpleNamespace(**ui_defaults.parseq_args)
    dv = SimpleNamespace(**ui_defaults.video_args)
    dr = SimpleNamespace(**ui_defaults.root_args)
    dw = SimpleNamespace(**ui_defaults.wan_args)
    
    # Create placeholder for removed LoopArgs with proper UI element format
    loop_args_dict = {
        "use_looper": {
            "label": "Use guided images mode",
            "type": "checkbox",
            "value": False,
            "info": "Enable guided images (loop) mode for keyframe-based animation"
        },
        "init_images": {
            "label": "Images to keyframe", 
            "type": "textbox",
            "value": "{}",
            "info": "JSON format of images to use as keyframes"
        },
        "image_strength_schedule": {
            "label": "Image strength schedule",
            "type": "textbox", 
            "value": "0: (0.0)",
            "info": "Schedule for image strength over time"
        },
        "image_keyframe_strength_schedule": {
            "label": "Image keyframe strength schedule",
            "type": "textbox",
            "value": "0: (0.0)", 
            "info": "Schedule for keyframe strength over time"
        },
        "blendFactorMax": {
            "label": "Blend factor max",
            "type": "textbox",
            "value": "0: (0.35)",
            "info": "Maximum blend factor for guided images"
        },
        "blendFactorSlope": {
            "label": "Blend factor slope", 
            "type": "textbox",
            "value": "0: (0.25)",
            "info": "Slope for blend factor changes"
        },
        "tweening_frames_schedule": {
            "label": "Tweening frames schedule",
            "type": "textbox",
            "value": "0: (20)",
            "info": "Schedule for tweening frame counts"
        },
        "color_correction_factor": {
            "label": "Color correction factor",
            "type": "textbox", 
            "value": "0: (0.075)",
            "info": "Factor for color correction between frames"
        }
    }
    dloopArgs = SimpleNamespace(**loop_args_dict)
    
    return d, da, dp, dv, dr, dw, dloopArgs


def get_component_names():
    """Get list of all UI component names for gradio integration.
    
    Returns:
        List of component names for UI setup
    """
    return [
        'override_settings_with_file', 'custom_settings_file', 
        *DeforumAnimArgs().keys(), 
        'animation_prompts', 'animation_prompts_positive', 'animation_prompts_negative',
        *DeforumArgs().keys(), 
        *DeforumOutputArgs().keys(), 
        *ParseqArgs().keys(),
        # Note: controlnet and WAN components handled separately in their modules
    ] 