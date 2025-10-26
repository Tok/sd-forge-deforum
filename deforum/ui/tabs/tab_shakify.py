"""Camera Shakify tab for Deforum UI.

Provides controls for integrating realistic camera shake effects into renders
using data sourced from EatTheFuture's 'Camera Shakify' Blender plugin.
"""

import gradio as gr
from deforum.utils.system.logging import emoji as emoji_utils
from deforum.utils.ui.builders import create_row


def get_tab_shakify(da, skip_tabitem=False):
    """Create the Camera Shakify tab.

    Args:
        da: DeforumAnimArgs namespace
        skip_tabitem: If True, don't create TabItem wrapper (default: False)

    Returns:
        dict: Component dictionary for event binding
    """
    # Controls first - most important
    shake_name = create_row(da.shake_name)
    shake_intensity = create_row(da.shake_intensity)
    shake_speed = create_row(da.shake_speed)

    # Explanation after controls
    with gr.Accordion(f"{emoji_utils.info} About Camera Shakify", open=True):
        gr.Markdown("""
        ## Camera Shakify
        **Integrate dynamic camera shake effects** into your renders with data sourced from EatTheFuture's 'Camera Shakify' Blender plugin.

        This feature enhances the realism of your animations by simulating natural camera movements, adding a layer of depth and engagement to your visuals.

        **Available Shake Patterns:**
        - EARTHQUAKE - Violent, chaotic shaking
        - FILM_GRAIN - Subtle analog film vibration
        - GENTLE_HANDHELD - Natural handheld camera movement
        - INVESTIGATION - Detective-style documentary camera work
        - MOVING_HANDHELD - Active walking/running camera movement
        - PANIC - Frantic, disoriented shaking
        - ROLLING_SHUTTER - Digital camera sensor distortion
        - And more...

        **How It Works:**
        - Shake patterns are layered on top of your scheduled movement (translation, rotation, zoom)
        - Intensity controls the magnitude of shake
        - Speed controls how fast the shake pattern plays
        """)

    return {k: v for k, v in {**locals(), **vars()}.items()}
