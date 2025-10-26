"""Distribution & Render Mode tab for Deforum UI.

Main workflow control for switching between rendering modes and configuring
keyframe distribution options including Wan FLF2V integration.
"""

import gradio as gr
from modules.ui_components import FormRow
from deforum.utils.system.logging import emoji as emoji_utils
from deforum.utils.ui.builders import create_row


def create_keyframe_distribution_info_tab():
    """Create informational content about keyframe distribution.

    This is defined in ui_elements.py and will be imported later
    when we refactor that file.
    """
    # Import here to avoid circular dependency during refactoring
    from deforum.ui.ui_elements import create_keyframe_distribution_info_tab as _create_info
    return _create_info()


def get_tab_distribution(da):
    """Create the Distribution & Render Mode tab.

    Args:
        da: DeforumAnimArgs namespace

    Returns:
        dict: Component dictionary for event binding
    """
    with gr.TabItem(f"{emoji_utils.distribution()} Distribution", elem_id='distribution_tab'):
        keyframe_distribution = create_row(da.keyframe_distribution)

        # Wan FLF2V Integration
        with gr.Accordion(f"{emoji_utils.movie_camera()} Wan FLF2V Tween Mode (Experimental)", open=False):
            gr.Markdown("""
            **Use Wan AI video interpolation instead of depth-based tweening.**

            **When to use:**
            - Calm sections with few tween frames (< 20 frames between keyframes)
            - When depth warping creates artifacts
            - When you want cinematic AI-generated motion

            **How it works:**
            1. Flux generates keyframes as normal
            2. Wan FLF2V interpolates smooth video between keyframes
            3. No depth estimation needed

            **⚠️ Requirements:**
            - **MUST use FLF2V-specific model:** Wan2.1-FLF2V-14B
            - **TI2V models (e.g., Wan2.2-TI2V-5B) will NOT work** - they extend first frame instead
            - Works best with keyframe distribution mode
            - VRAM: ~15-18GB (less than standalone Wan T2V)
            - Download: `huggingface-cli download Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers --local-dir models/Deforum/wan/Wan2.1-FLF2V-14B`

            **For longer sections (> 81 frames):**
            - Automatically uses FLF2V chaining mode
            - Generates depth-tween intermediate keyframes
            - Chains FLF2V between them for smooth motion
            """)
            enable_wan_flf2v = create_row(da.enable_wan_flf2v)
            wan_flf2v_chunk_size = create_row(da.wan_flf2v_chunk_size)

            gr.Markdown("**Per-Keyframe Type Control (Advanced):**")
            keyframe_type_schedule = create_row(da.keyframe_type_schedule)

            with FormRow():
                auto_assign_keyframe_types_btn = gr.Button(
                    "🤖 Auto-Assign Types",
                    variant="secondary",
                    size="sm",
                    elem_id="auto_assign_keyframe_types_btn"
                )
                gr.Markdown("*Analyzes tween distances and suggests optimal types based on chunk size*")

        # Informational section at bottom
        gr.Markdown("""
        ---
        ## Keyframe Distribution & Render Mode

        **This is the main control for switching between rendering modes:**
        - **Keyframes Only:** Modern render core (recommended for Flux + Wan)
        - **Cadence:** Traditional rendering with fixed frame intervals

        **Wan FLF2V Integration** is available when using Keyframes Only mode.
        """)

        create_keyframe_distribution_info_tab()

    return {k: v for k, v in {**locals(), **vars()}.items()}
