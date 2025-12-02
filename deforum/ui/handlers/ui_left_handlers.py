"""Event handlers and UI builders for Deforum left side UI.

Extracted from ui_left.py to reduce complexity.
"""

import gradio as gr
from typing import Tuple, Dict, Any
from deforum.utils.system.logging import get_logger, emoji_if_enabled

logger = get_logger()


# UI Builder Functions

def build_flux_blocker_ui(d) -> Dict[str, Any]:
    """Build minimal UI when Flux is not available.

    Args:
        d: Default args namespace

    Returns:
        Dict with minimal component set for compatibility
    """
    from deforum.utils.system.flux_check import get_flux_setup_message

    with gr.Row(variant='compact'):
        show_info_on_ui = gr.Checkbox(
            label="Show more info",
            value=d.show_info_on_ui,
            interactive=True,
            visible=False
        )

    gr.HTML(value=get_flux_setup_message())

    return {'show_info_on_ui': show_info_on_ui}


def build_top_level_controls(d, da, dv):
    """Build top-level UI controls (render mode, reset, fps, steps, etc.).

    Args:
        d: Default args
        da: Default anim args
        dv: Default video args

    Returns:
        Dict of created components
    """
    from .ui_elements import create_gr_elem

    components = {}

    # Show info checkbox
    with gr.Row(variant='compact'):
        components['show_info_on_ui'] = gr.Checkbox(
            label="Show more info",
            value=d.show_info_on_ui,
            interactive=True
        )

    # Render mode + Reset button
    with gr.Row(variant='compact'):
        components['render_mode'] = create_gr_elem(da.render_mode)
        with gr.Column(scale=0, min_width=60):
            components['reset_to_defaults_btn'] = gr.Button(
                value="Reset",
                variant="secondary",
                size="sm"
            )

    # Reset confirmation modal
    with gr.Row(visible=False) as reset_confirm_row:
        with gr.Column(scale=1):
            warning_emoji = emoji_if_enabled("⚠️") or "WARNING"
            gr.Markdown(f"{warning_emoji} **Reset to Mode Defaults?**\n\nThis will reset Steps, Scheduler, CFG Scale, Strength, FPS, and Cadence to optimized defaults for this render mode. Your prompts and keyframes will NOT be changed.")
            with gr.Row():
                components['reset_confirm_yes'] = gr.Button(
                    "Yes, Reset to Defaults",
                    variant="primary",
                    size="sm",
                    elem_classes=["slopcore-button"]
                )
                components['reset_confirm_no'] = gr.Button(
                    "Cancel",
                    variant="secondary",
                    size="sm"
                )

    components['reset_confirm_row'] = reset_confirm_row

    # Progress indicator
    components['reset_progress'] = gr.Textbox(
        label="Generation Status",
        value="",
        interactive=False,
        visible=False,
        lines=4
    )

    # FPS and Steps
    with gr.Row(variant='compact'):
        components['fps'] = create_gr_elem(dv.fps)
        components['steps'] = create_gr_elem(d.steps)

    # Reverse generation
    with gr.Row(variant='compact'):
        components['reverse_generation'] = create_gr_elem(da.reverse_generation)

    # Fractional strength
    with gr.Row(variant='compact'):
        components['enable_fractional_strength'] = gr.Checkbox(
            label="Fractional Strength (1% precision)",
            value=True,
            info="Enables true 1% strength resolution via fractional t_enc (Forge monkey patch). "
                 "Affects both slider precision AND actual sampling. "
                 "OFF: discrete 1/steps resolution. ON: continuous 1% precision."
        )

    return components


def build_cadence_controls():
    """Build cadence/pseudo-cadence control columns.

    Returns:
        Dict with cadence_column, pseudo_cadence_column, cadence, pseudo_cadence_display
    """
    from .ui_elements import create_gr_elem
    from deforum.config.args import DeforumAnimArgs

    da = DeforumAnimArgs()
    components = {}

    with gr.Row(variant='compact'):
        with gr.Column(scale=1, visible=True) as cadence_column:
            components['cadence'] = create_gr_elem(da['diffusion_cadence'])
        with gr.Column(scale=1, visible=False) as pseudo_cadence_column:
            components['pseudo_cadence_display'] = gr.Textbox(
                label="Calculated Pseudo-Cadence",
                value="Will be calculated on render",
                interactive=False,
                info="Average frames between diffusions (calculated from keyframes)"
            )

    components['cadence_column'] = cadence_column
    components['pseudo_cadence_column'] = pseudo_cadence_column

    return components


def build_strength_sliders():
    """Build strength slider controls (normal and keyframe).

    Returns:
        Dict with strength slider components and columns
    """
    components = {}

    with gr.Row(variant='compact'):
        # Normal strength
        with gr.Column(scale=1, visible=True) as normal_strength_column:
            components['normal_strength_slider'] = gr.Slider(
                label="Normal Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.85,
                info="Fractional: 1% precision (0.01) via log-linear interpolation. Slider sets constant '0:(value)'. Edit textbox for complex schedules."
            )

        # Keyframe strength
        with gr.Column(scale=1, visible=True) as keyframe_strength_column:
            components['keyframe_strength_slider'] = gr.Slider(
                label="Keyframe Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.20,
                info="Fractional: 1% precision (0.01) via log-linear interpolation. Slider sets constant '0:(value)'. Edit textbox for complex schedules."
            )

    components['normal_strength_column'] = normal_strength_column
    components['keyframe_strength_column'] = keyframe_strength_column

    return components


def build_hidden_legacy_fields(da):
    """Build hidden legacy compatibility fields.

    Args:
        da: Default anim args

    Returns:
        Dict with hidden legacy components
    """
    from .ui_elements import create_gr_elem

    components = {}

    # Hidden animation_mode for backwards compatibility
    components['animation_mode'] = create_gr_elem(da.animation_mode, visible=False)

    # Hidden keyframe_distribution
    components['keyframe_distribution'] = create_gr_elem(da.keyframe_distribution, visible=False)

    return components


# Event Handler Functions


def handle_render_mode_change(mode: str, fractional_enabled: bool) -> list:
    """Update UI components when render mode changes.

    Args:
        mode: Selected render mode string
        fractional_enabled: Whether fractional strength is enabled

    Returns:
        List of gr.update() for: tab_depth, tab_shakify, tab_wan, cadence, pseudo_cadence,
                                 fps, steps, sliders, strength_columns, animation_mode
    """
    from deforum.rendering.data.render_mode import RenderMode

    render_mode_enum = RenderMode.from_string(mode)
    config = render_mode_enum.config

    # Determine tab visibility
    show_3d_tabs = render_mode_enum.should_show_3d_tabs()
    show_wan_tab = render_mode_enum.should_show_wan_tab()

    # Determine cadence/pseudo-cadence visibility
    show_real_cadence = render_mode_enum.should_show_cadence_slider()
    show_pseudo_cadence = config.shows_pseudo_cadence

    # Determine strength slider visibility
    show_normal_strength = render_mode_enum.should_show_normal_strength()
    show_keyframe_strength = render_mode_enum.should_show_keyframe_strength()

    # Calculate strength resolution (slider step size)
    if fractional_enabled:
        strength_step = 0.01
        strength_info = "Fractional: 1% precision (0.01). Slider sets constant '0:(value)'. Edit textbox for complex schedules."
    else:
        strength_step = 1.0 / config.default_steps
        strength_info = f"Discrete: 1/{config.default_steps} = {strength_step:.4f}. Slider sets constant '0:(value)'. Edit textbox for complex schedules."

    # Mode-specific steps info text
    steps_info_map = {
        RenderMode.CLASSIC_3D: "Sampling steps for all diffusions (every cadence frames)",
        RenderMode.NEW_3D: "Sampling steps for all diffusions (keyframes + cadence frames)",
        RenderMode.KEYFRAMES_ONLY: "Sampling steps for keyframe diffusions only",
        RenderMode.FLUX_WAN: "Sampling steps for Flux keyframe generation (Wan FLF2V steps in Wan Models tab)",
    }
    steps_info = steps_info_map.get(render_mode_enum, "Sampling steps for diffusion")

    # Update legacy animation_mode for backward compatibility
    legacy_mode = render_mode_enum.to_legacy_animation_mode()

    return [
        gr.update(visible=show_3d_tabs),           # tab_depth
        gr.update(visible=show_3d_tabs),           # tab_shakify
        gr.update(visible=show_wan_tab),           # tab_wan
        gr.update(visible=show_real_cadence),      # cadence_column
        gr.update(visible=show_pseudo_cadence),    # pseudo_cadence_column
        gr.update(value=config.default_fps),       # fps
        gr.update(value=config.default_steps, info=steps_info),  # steps
        gr.update(step=strength_step, info=strength_info),  # normal_strength_slider
        gr.update(step=strength_step, info=strength_info),  # keyframe_strength_slider
        gr.update(visible=show_normal_strength),   # normal_strength_column
        gr.update(visible=show_keyframe_strength), # keyframe_strength_column
        gr.update(value=legacy_mode)               # animation_mode (hidden)
    ]


def on_reset_to_defaults_click(render_mode_val: str) -> Tuple:
    """Handle Reset to Mode Defaults button click.

    Generates AI-powered defaults for the selected render mode:
    1. Loads static settings from JSON
    2. Generates audio with Meta MusicGen
    3. Detects audio events with BPM-aware sensitivity
    4. Generates prompts with Qwen
    5. Syncs prompts to audio events
    6. Saves to batch directory

    Note: Due to Gradio limitations, defaults are saved to a batch directory.
    User needs to manually load the settings file after generation.

    Args:
        render_mode_val: Selected render mode

    Returns:
        Tuple of gr.update() for (status, prompts, audio, fps, max_frames, rotation, translation_z, zoom)
    """
    from deforum.config.defaults_generator import generate_mode_defaults
    import json
    from pathlib import Path
    from datetime import datetime

    try:
        # Get current model for model-specific defaults
        try:
            from modules import shared
            current_model = (shared.sd_model.sd_checkpoint_info.model_name
                           if hasattr(shared, 'sd_model')
                           else "Flux\\flux1-dev-bnb-nf4-v2.safetensors")
        except:
            current_model = "Flux\\flux1-dev-bnb-nf4-v2.safetensors"  # Fallback

        # Create batch directory with timestamp
        timestring = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_safe = render_mode_val.replace(' ', '')
        batch_name = f"Deforum_Defaults_{mode_safe}_{timestring}"
        batch_dir = Path("output") / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)

        # Generate defaults
        logger.info(f"Generating defaults for {render_mode_val} with model {current_model}")
        logger.info(f"Batch directory: {batch_dir}")

        defaults = generate_mode_defaults(
            render_mode=render_mode_val,
            current_model=current_model,
            batch_dir=batch_dir,
            progress_callback=None
        )

        # Save settings to batch directory
        settings_file = batch_dir / "deforum_settings.txt"
        with open(settings_file, 'w') as f:
            json.dump(defaults, f, indent=4)

        check_emoji = emoji_if_enabled("✓") or "[OK]"
        success_msg = (
            f"{check_emoji} Defaults loaded successfully!\n"
            f"Generated {len(defaults.get('prompts', {}))} prompts synced to audio.\n"
            f"Batch: {batch_name}\n"
            f"Settings: {settings_file}"
        )

        logger.info(f"{emoji_if_enabled('✓')} Reset to defaults complete for {render_mode_val}")
        logger.info(f"Batch created: {batch_dir}")

        # Return updates for key components
        import json as json_module
        return (
            gr.update(value=success_msg, visible=True),  # Status message
            gr.update(value=json_module.dumps(defaults.get('prompts', {}), indent=4)),  # Prompts
            gr.update(value=defaults.get('soundtrack_path', '')),  # Audio path
            gr.update(value=defaults.get('fps', 60)),  # FPS
            gr.update(value=defaults.get('max_frames', 120)),  # Max frames
            gr.update(value=defaults.get('rotation_3d_y', '0: (0)')),  # Camera rotation
            gr.update(value=defaults.get('translation_z', '0: (0)')),  # Camera Z
            gr.update(value=defaults.get('zoom', '0: (1.0)')),  # Zoom
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        x_emoji = emoji_if_enabled("✗") or "[ERROR]"
        error_msg = f"{x_emoji} Error generating defaults:\n{str(e)}\n\nCheck console for details."
        logger.error(f"Reset to defaults failed: {e}")
        # Return error message + no updates for other components
        return (
            gr.update(value=error_msg, visible=True),  # Status message
            gr.update(),  # Prompts - no change
            gr.update(),  # Audio path - no change
            gr.update(),  # FPS - no change
            gr.update(),  # Max frames - no change
            gr.update(),  # Camera rotation - no change
            gr.update(),  # Camera Z - no change
            gr.update(),  # Zoom - no change
        )


def update_slider_step_size(steps_value: int, fractional_enabled: bool) -> list:
    """Update slider step size based on steps and fractional checkbox state.

    Args:
        steps_value: Number of sampling steps
        fractional_enabled: Whether fractional interpolation checkbox is enabled

    Returns:
        List of gr.update() for [normal_strength_slider, keyframe_strength_slider]
    """
    if fractional_enabled:
        # Fractional interpolation: 1% precision regardless of steps
        step = 0.01
        info_text = "Fractional: 1% precision (0.01). Slider sets constant '0:(value)'. Edit textbox for complex schedules."
    else:
        # Discrete: precision = 1/steps
        step = 1.0 / max(1, steps_value)
        info_text = f"Discrete: 1/{steps_value} = {step:.4f}. Slider sets constant '0:(value)'. Edit textbox for complex schedules."

    return [
        gr.update(step=step, info=info_text),  # normal_strength_slider
        gr.update(step=step, info=info_text),  # keyframe_strength_slider
    ]


def slider_to_textbox(slider_value: float) -> str:
    """Convert slider value to schedule textbox format.

    Args:
        slider_value: Slider value (0.0-1.0)

    Returns:
        Formatted schedule string
    """
    return f"0: ({slider_value:.2f})"
