# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

from types import SimpleNamespace
import gradio as gr
from deforum.config.defaults import get_gradio_html
from deforum.ui.gradio_funcs import change_css, handle_change_functions
from deforum.config.args import DeforumArgs, DeforumAnimArgs, ParseqArgs, AudioSyncArgs, DeforumOutputArgs, RootArgs, LoopArgs, WanArgs
from deforum.utils.system.logging import emoji as emoji_utils
# TEMPORARILY DISABLED: ControlNet support disabled until Flux-specific reimplementation
# from .deforum_controlnet import setup_controlnet_ui
from deforum.ui.ui_elements import (get_tab_run, get_tab_keyframes, get_tab_prompts, get_tab_init,
                          get_tab_output, get_tab_masking)
from deforum.ui.tabs.tab_camera_path import get_tab_camera_path
from deforum.ui.tabs.tab_zero_hitl import get_tab_zero_hitl
from deforum.ui.handlers.audio_prompt_generator import generate_prompts_with_ai
from deforum.ui.handlers.audio_sync import synchronize_prompts_to_audio
from deforum.ui.handlers.ui_left_handlers import handle_render_mode_change
from deforum.utils.system.logging import get_logger, emoji_if_enabled

# Initialize logger
logger = get_logger()


def set_arg_lists():
    # convert dicts to NameSpaces for easy working (args.param instead of args['param']
    d = SimpleNamespace(**DeforumArgs())  # default args
    da = SimpleNamespace(**DeforumAnimArgs())  # default anim args
    dp = SimpleNamespace(**ParseqArgs())  # default parseq ars
    dau = SimpleNamespace(**AudioSyncArgs())  # default audio sync args
    dv = SimpleNamespace(**DeforumOutputArgs())  # default video args
    dr = SimpleNamespace(**RootArgs())  # ROOT args
    dw = SimpleNamespace(**WanArgs())  # Wan args
    dloopArgs = SimpleNamespace(**LoopArgs())  # Guided imgs args
    return d, da, dp, dau, dv, dr, dw, dloopArgs

def wan_generate_video():
    """
    Simple placeholder function for Wan video generation button
    Returns a status message indicating the feature is integrated but needs models
    """
    try:
        logger.info("Wan video generation button clicked!", emoji='movie_camera')
        
        # Try to discover models to check if setup is complete
        try:
            from deforum.integrations.wan.wan_simple_integration import WanSimpleIntegration
            integration = WanSimpleIntegration()
            models = integration.discover_models()
            
            # Theme-aware emoji symbols
            check = emoji_utils.maybe_check()
            cross = emoji_utils.maybe_cross()
            warning = emoji_utils.maybe_warning()
            bulb = emoji_utils.bulb()
            folder = emoji_utils.folder()

            if models:
                return f"""{check} Wan integration is working!

Found {len(models)} model(s):
{chr(10).join([f"• {model['name']} ({model['size']})" for model in models[:3]])}

{bulb} Next steps:
1. Ensure your prompts are configured in the Prompts tab
2. Set your desired FPS in the Output tab
3. Choose animation mode 'Flux + Interpolation' in the Keyframes tab
4. Click the main Generate button in Deforum

{folder} Models found in: {models[0]['path']}"""
            else:
                return f"""{cross} No Wan models found!

{bulb} SETUP REQUIRED:
1. Download a Wan model:
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/Deforum/wan

2. Or place your Wan models in:
   • models/Deforum/wan/
   • models/Wan/
   • HuggingFace cache (automatic)

3. Restart the WebUI after downloading

The auto-discovery will find your models automatically!"""

        except ImportError as e:
            return f"""{warning} Wan integration partially loaded

The Wan tab is integrated but some dependencies may be missing.

Error: {str(e)}

{bulb} To complete setup:
1. Download Wan models as instructed above
2. Ensure all Wan dependencies are installed
3. Check the console for any import errors"""

        except Exception as e:
            return f"""{cross} Wan integration error: {str(e)}

{bulb} Troubleshooting:
1. Check that Wan models are downloaded and placed correctly
2. Verify all dependencies are installed
3. Check console output for detailed error messages
4. Try restarting the WebUI"""

    except Exception as e:
        logger.error(f"Wan button error: {e}", emoji='off')
        cross = emoji_utils.maybe_cross()
        return f"{cross} Error: {str(e)}"

def setup_deforum_left_side_ui():
    d, da, dp, dau, dv, dr, dw, dloopArgs = set_arg_lists()

    # FLUX AVAILABILITY CHECK - All Deforum modes require Flux
    from deforum.utils.system.flux_check import should_show_flux_blocker, get_flux_setup_message

    if should_show_flux_blocker():
        # Show blocker message and minimal UI
        with gr.Row(variant='compact'):
            show_info_on_ui = gr.Checkbox(label="Show more info", value=d.show_info_on_ui, interactive=True, visible=False)

        gr.HTML(value=get_flux_setup_message())

        # Return minimal component set for compatibility
        return {
            'show_info_on_ui': show_info_on_ui,
        }

    # Normal UI setup continues if Flux is available
    # show button to hide/ show gradio's info texts for each element in the UI
    with gr.Row(variant='compact'):
        show_info_on_ui = gr.Checkbox(label="Show more info", value=d.show_info_on_ui, interactive=True)

    # Top-level essential settings before tabs - NEW RENDER MODE SYSTEM
    with gr.Row(variant='compact'):
        from .ui_elements import create_gr_elem
        render_mode = create_gr_elem(da.render_mode)
        with gr.Column(scale=0, min_width=60):
            reset_to_defaults_btn = gr.Button(
                value="Reset",
                variant="secondary",
                size="sm"
            )

    # Confirmation modal for reset to defaults
    with gr.Row(visible=False) as reset_confirm_row:
        with gr.Column(scale=1):
            warning_emoji = emoji_if_enabled("⚠️") or "WARNING"
            gr.Markdown(f"{warning_emoji} **Reset to Mode Defaults?**\n\nThis will generate new AI defaults. Make sure you've saved your current settings if needed.")
            with gr.Row():
                reset_confirm_yes = gr.Button(
                    "Yes, Generate Defaults",
                    variant="primary",
                    size="sm",
                    elem_classes=["slopcore-button"]
                )
                reset_confirm_no = gr.Button("Cancel", variant="secondary", size="sm")

    # Progress indicator for defaults generation
    reset_progress = gr.Textbox(
        label="Generation Status",
        value="",
        interactive=False,
        visible=False,
        lines=4
    )

    with gr.Row(variant='compact'):
        fps = create_gr_elem(dv.fps)
        steps = create_gr_elem(d.steps)

    # Reverse generation checkbox - top-level
    with gr.Row(variant='compact'):
        reverse_generation = create_gr_elem(da.reverse_generation)

    # Fractional strength - top-level (UI + backend)
    with gr.Row(variant='compact'):
        enable_fractional_strength = gr.Checkbox(
            label="Fractional Strength (1% precision)",
            value=True,
            info="Enables true 1% strength resolution via fractional t_enc (Forge monkey patch). "
                 "Affects both slider precision AND actual sampling. "
                 "OFF: discrete 1/steps resolution. ON: continuous 1% precision."
        )

    # Mode-dependent controls row - cadence OR pseudo-cadence (mutually exclusive)
    with gr.Row(variant='compact'):
        with gr.Column(scale=1, visible=True) as cadence_column:
            cadence = create_gr_elem(da.diffusion_cadence)
        with gr.Column(scale=1, visible=False) as pseudo_cadence_column:
            pseudo_cadence_display = gr.Textbox(
                label="Calculated Pseudo-Cadence",
                value="Will be calculated on render",
                interactive=False,
                info="Average frames between diffusions (calculated from keyframes)"
            )

    # Strength schedules - smart sliders + textboxes
    # Sliders for quick constant values, textboxes for complex schedules
    with gr.Row(variant='compact'):
        # Normal strength (Classic 3D, New 3D)
        with gr.Column(scale=1, visible=True) as normal_strength_column:
            normal_strength_slider = gr.Slider(
                label="Normal Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,  # Initialize with 1% (fractional default enabled)
                value=0.85,
                info="Fractional: 1% precision (0.01) via log-linear interpolation. Slider sets constant '0:(value)'. Edit textbox for complex schedules."
            )
            normal_strength = create_gr_elem(da.strength_schedule)

        # Keyframe strength (New 3D, Keyframes Only, Flux + Interpolation)
        with gr.Column(scale=1, visible=True) as keyframe_strength_column:
            keyframe_strength_slider = gr.Slider(
                label="Keyframe Strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,  # Initialize with 1% (fractional default enabled)
                value=0.50,
                info="Fractional: 1% precision (0.01) via log-linear interpolation. Slider sets constant '0:(value)'. Edit textbox for complex schedules."
            )
            keyframe_strength = create_gr_elem(da.keyframe_strength_schedule)

    # Keep legacy animation_mode hidden for backwards compatibility
    animation_mode = gr.Radio(
        label="Animation mode (Legacy - Hidden)",
        choices=['3D', 'Interpolation', 'Flux + Interpolation'],
        value="3D",
        visible=False
    )

    # Hidden keyframe_distribution (render mode automatically controls distribution now)
    keyframe_distribution = gr.Dropdown(
        label="Keyframe Distribution (Auto-Controlled by Render Mode)",
        choices=["Off", "Keyframes Only", "Redistributed Cadence"],
        value="Keyframes Only",
        visible=False
    )

    with gr.Blocks():
        with gr.Tabs() as main_tabs:
            # Get main tab contents in new workflow order:
            # Init tab now contains Zero-HITL as first subtab
            tab_init_params = get_tab_init(d, da, dp, dau, dv)  # 1. Init - all modes (includes Zero-HITL, audio sync)
            from .ui_elements import get_tab_shakify, get_tab_depth_warping
            tab_prompts_params = get_tab_prompts(da, dw, dv)  # 2. Prompts - all modes (now includes audio/timing)
            tab_keyframes_params = get_tab_keyframes(d, da, dloopArgs)  # 3. Keyframes - all modes
            tab_camera_path_params = get_tab_camera_path(da)  # 4. Camera Path - after audio sync & keyframes

            # Mode-specific tabs (with visibility control):
            # 3D mode only tabs:
            with gr.TabItem(f"{emoji_utils.hole()} 3D Depth", visible=True) as tab_depth:
                tab_depth_params = get_tab_depth_warping(da, skip_tabitem=True)  # 5. 3D Depth - 3D mode only
            with gr.TabItem(f"{emoji_utils.bicycle()} Shakify", visible=True) as tab_shakify:
                tab_shakify_params = get_tab_shakify(da, skip_tabitem=True)  # 6. Shakify - 3D mode only
            with gr.TabItem(f"{emoji_utils.masking()} Masking", visible=True) as tab_masking:
                tab_masking_params = get_tab_masking(d, da, skip_tabitem=True)  # 7. Masking - all modes

            # Flux + Interpolation mode tab:
            from .ui_elements import get_tab_wan
            with gr.TabItem(f"{emoji_utils.wan_video()} Interpolation", visible=True) as tab_wan:
                tab_wan_params = get_tab_wan(dw, da, skip_tabitem=True)  # 8. Interpolation - Flux + Interpolation mode

            # Always visible tabs:
            tab_run_params = get_tab_run(d, da)  # 8. Run - all modes
            tab_output_params = get_tab_output(da, dv)  # 9. Output - all modes

            # TEMPORARILY DISABLED: ControlNet support disabled until Flux-specific reimplementation
            controlnet_dict = {}  # Empty dict for backwards compatibility

            # add returned gradio elements from main tabs to locals()
            # Note: Zero-HITL components now come from tab_init_params
            for key, value in {**tab_run_params, **tab_keyframes_params, **tab_prompts_params, **tab_camera_path_params, **tab_shakify_params, **tab_masking_params, **tab_depth_params, **tab_init_params, **controlnet_dict, **tab_wan_params, **tab_output_params}.items():
                locals()[key] = value

            # WORKAROUND: Explicitly unpack audio AI components as actual local variables
            # (Python doesn't support creating locals via locals()[key]=value, so we must assign them explicitly)
            audio_ai_generate_button = tab_init_params.get('audio_ai_generate_button')
            audio_ai_generation_mode = tab_init_params.get('audio_ai_generation_mode')
            audio_ai_intensity = tab_init_params.get('audio_ai_intensity')
            audio_ai_style = tab_init_params.get('audio_ai_style')
            audio_ai_prompt_theme = tab_init_params.get('audio_ai_prompt_theme')
            audio_ai_prompt_count = tab_init_params.get('audio_ai_prompt_count')
            audio_ai_start_prompt = tab_init_params.get('audio_ai_start_prompt')
            audio_ai_end_prompt = tab_init_params.get('audio_ai_end_prompt')
            audio_sync_prompts = tab_init_params.get('audio_sync_prompts')

            # Explicitly unpack components needed for Reset to Defaults button
            soundtrack_path = tab_init_params.get('soundtrack_path')
            max_frames = tab_keyframes_params.get('max_frames')
            rotation_3d_y = tab_keyframes_params.get('rotation_3d_y')
            translation_z = tab_keyframes_params.get('translation_z')
            zoom = tab_keyframes_params.get('zoom')

            # Also unpack Wan FLF2V component from depth tab
            enable_wan_flf2v = tab_depth_params.get('enable_wan_flf2v')

            # Unpack Wan components from wan tab
            wan_generate_button = tab_wan_params.get('wan_generate_button')
            wan_generation_status = tab_wan_params.get('wan_generation_status')

            # Add top-level settings to locals()
            locals()['render_mode'] = render_mode
            locals()['animation_mode'] = animation_mode
            locals()['keyframe_distribution'] = keyframe_distribution
            locals()['fps'] = fps
            locals()['steps'] = steps
            locals()['cadence'] = cadence
            locals()['diffusion_cadence'] = cadence  # Alias for backward compatibility with gradio_funcs
            locals()['cadence_column'] = cadence_column
            locals()['pseudo_cadence_display'] = pseudo_cadence_display
            locals()['pseudo_cadence_column'] = pseudo_cadence_column
            locals()['strength_schedule'] = normal_strength
            locals()['keyframe_strength_schedule'] = keyframe_strength
            locals()['normal_strength_slider'] = normal_strength_slider
            locals()['keyframe_strength_slider'] = keyframe_strength_slider
            locals()['normal_strength_column'] = normal_strength_column
            locals()['keyframe_strength_column'] = keyframe_strength_column

            # Store tab references for visibility control
            locals()['tab_depth'] = tab_depth
            locals()['tab_shakify'] = tab_shakify
            locals()['tab_wan'] = tab_wan

            # ====== AUDIO SYNC BUTTON WIRING (moved inside tabs context) ======
            logger.debug("Button wiring section reached - checking for audio components...")
            logger.debug(f"   audio_upload in locals: {'audio_upload' in locals()}")
            logger.debug(f"   audio_ai_generate_button in locals: {'audio_ai_generate_button' in locals()}")

            # Wire up audio upload to use actual FPS and update max_frames
            if 'audio_upload' in locals() and 'soundtrack_path' in locals():
                def handle_audio_upload_with_fps(audio_filepath, current_fps):
                    """Save uploaded audio, calculate duration, and suggest max_frames."""
                    if audio_filepath is None:
                        return None, "File", "", gr.update()

                    import os
                    import shutil
                    from pathlib import Path

                    # Create outputs/audio directory
                    output_dir = Path("outputs/audio")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Save uploaded file
                    filename = Path(audio_filepath).name
                    dest_path = output_dir / filename
                    shutil.copy2(audio_filepath, dest_path)
                    abs_path = str(dest_path.absolute())

                    # Calculate audio duration and suggested max_frames
                    try:
                        import librosa
                        duration = librosa.get_duration(path=abs_path)
                        fps_val = current_fps if current_fps and current_fps > 0 else 24
                        suggested_max_frames = int(duration * fps_val)
                        info_text = f"Duration: {duration:.2f}s | Suggested max_frames @ {fps_val} FPS: {suggested_max_frames}"

                        # Auto-update max_frames
                        max_frames_update = gr.update(value=suggested_max_frames)
                    except Exception as e:
                        info_text = f"Could not calculate duration: {str(e)}"
                        max_frames_update = gr.update()

                    return abs_path, "File", info_text, max_frames_update

                # Re-wire the upload event with actual FPS and max_frames
                if 'max_frames' in locals():
                    locals()['audio_upload'].upload(
                        fn=handle_audio_upload_with_fps,
                        inputs=[locals()['audio_upload'], fps],
                        outputs=[
                            locals()['soundtrack_path'],
                            locals()['add_soundtrack'],
                            locals()['audio_info_display'],
                            locals()['max_frames']
                        ]
                    )

            # Wire up audio sync button to synchronize prompts with audio events
            # Get buttons from tab_init_params (they're in Init tab, not Prompts tab!)
            # Get animation_prompts from tab_prompts_params (it's in Prompts tab)
            logger.debug("Attempting to retrieve audio sync components...")
            logger.debug(f"   tab_init_params type: {type(tab_init_params)}")
            logger.debug(f"   tab_prompts_params type: {type(tab_prompts_params)}")

            audio_sync_button = tab_init_params.get('audio_sync_button')
            audio_sync_fewer_button = tab_init_params.get('audio_sync_fewer_button')
            audio_sync_more_button = tab_init_params.get('audio_sync_more_button')
            audio_sync_status = tab_init_params.get('audio_sync_status')
            audio_sync_timeline = tab_init_params.get('audio_sync_timeline')
            audio_target_keyframe_count = tab_init_params.get('audio_target_keyframe_count')
            # New structured info displays
            audio_sync_keyframe_count_display = tab_init_params.get('audio_sync_keyframe_count_display')
            audio_sync_pseudo_cadence_display = tab_init_params.get('audio_sync_pseudo_cadence_display')
            audio_sync_bpm_display = tab_init_params.get('audio_sync_bpm_display')
            audio_sync_duration_display = tab_init_params.get('audio_sync_duration_display')
            animation_prompts = tab_prompts_params.get('animation_prompts')

            logger.debug(f"   Retrieved audio_sync_button: {audio_sync_button is not None}")
            logger.debug(f"   Retrieved audio_sync_status: {audio_sync_status is not None}")
            logger.debug(f"   Retrieved audio_sync_timeline: {audio_sync_timeline is not None}")
            logger.debug(f"   Retrieved audio_target_keyframe_count: {audio_target_keyframe_count is not None}")
            logger.debug(f"   Retrieved audio_sync_keyframe_count_display: {audio_sync_keyframe_count_display is not None}")
            logger.debug(f"   Retrieved audio_sync_pseudo_cadence_display: {audio_sync_pseudo_cadence_display is not None}")
            logger.debug(f"   Retrieved audio_sync_bpm_display: {audio_sync_bpm_display is not None}")
            logger.debug(f"   Retrieved audio_sync_duration_display: {audio_sync_duration_display is not None}")
            logger.debug(f"   Retrieved animation_prompts: {animation_prompts is not None}")

            if audio_sync_button and audio_sync_status:

                # Get all required inputs
                audio_sync_inputs = []
                required_components = [
                    'soundtrack_path',
                    'audio_sync_prompts',
                    'audio_prompt_distribution_mode',
                    'audio_target_keyframe_count',
                    'audio_detection_method',
                    'audio_frequency_band',
                    'audio_sensitivity',
                    'audio_intensity_threshold',
                    'audio_min_spacing_frames',
                    'fps'
                ]

                # Collect components from various tab dicts
                local_scope = locals()
                for comp_name in required_components:
                    found = False
                    # Check direct locals first
                    if comp_name in local_scope:
                        audio_sync_inputs.append(local_scope[comp_name])
                        found = True
                    # Check tab_init_params for soundtrack_path and fps
                    elif comp_name in tab_init_params:
                        audio_sync_inputs.append(tab_init_params[comp_name])
                        found = True
                    # Check tab_prompts_params for audio sync components
                    elif comp_name in tab_prompts_params:
                        audio_sync_inputs.append(tab_prompts_params[comp_name])
                        found = True

                    if not found:
                        logger.warning(f"Audio sync component '{comp_name}' not found")

                logger.debug(f"{emoji_if_enabled('🔍')} DEBUG Audio sync wiring check:")
                logger.debug(f"   Inputs collected: {len(audio_sync_inputs)}/{len(required_components)}")
                logger.debug(f"   audio_sync_button: {audio_sync_button is not None}")
                logger.debug(f"   audio_sync_fewer_button: {audio_sync_fewer_button is not None}")
                logger.debug(f"   audio_sync_more_button: {audio_sync_more_button is not None}")
                logger.debug(f"   audio_sync_status: {audio_sync_status is not None}")
                logger.debug(f"   animation_prompts: {animation_prompts is not None}")

                if len(audio_sync_inputs) == len(required_components):
                    # Buttons already retrieved above, just check they all exist
                    required_outputs = all([
                        audio_sync_button, audio_sync_fewer_button, audio_sync_more_button,
                        audio_sync_status, audio_sync_timeline, audio_target_keyframe_count,
                        audio_sync_keyframe_count_display, audio_sync_pseudo_cadence_display,
                        audio_sync_bpm_display, audio_sync_duration_display, animation_prompts
                    ])

                    if required_outputs:
                        # Define common outputs list for all buttons
                        audio_sync_outputs = [
                            animation_prompts,
                            audio_target_keyframe_count,
                            audio_sync_keyframe_count_display,
                            audio_sync_pseudo_cadence_display,
                            audio_sync_bpm_display,
                            audio_sync_duration_display,
                            audio_sync_timeline,
                            audio_sync_status
                        ]

                        # Main sync button (0% adjustment)
                        audio_sync_button.click(
                            fn=synchronize_prompts_to_audio,
                            inputs=audio_sync_inputs,
                            outputs=audio_sync_outputs
                        )

                        # Fewer keyframes button
                        def fewer_keyframes_wrapper(*args):
                            return synchronize_prompts_to_audio(*args, keyframe_adjustment=-5)

                        audio_sync_fewer_button.click(
                            fn=fewer_keyframes_wrapper,
                            inputs=audio_sync_inputs,
                            outputs=audio_sync_outputs
                        )

                        # More keyframes button
                        def more_keyframes_wrapper(*args):
                            return synchronize_prompts_to_audio(*args, keyframe_adjustment=5)

                        audio_sync_more_button.click(
                            fn=more_keyframes_wrapper,
                            inputs=audio_sync_inputs,
                            outputs=audio_sync_outputs
                        )

                        logger.debug("Audio sync buttons connected successfully (main, fewer, more)", emoji='sound')
                    else:
                        warning = emoji_utils.maybe_warning()
                        logger.error(f"{warning} Could not connect audio sync buttons: missing button/output components")
                        logger.info(f"   Condition checks: audio_sync_button={audio_sync_button is not None}, fewer={audio_sync_fewer_button is not None}, more={audio_sync_more_button is not None}, status={audio_sync_status is not None}, prompts={animation_prompts is not None}")
                else:
                    warning = emoji_utils.maybe_warning()
                    logger.error(f"{warning} Could not connect audio sync button: missing input components ({len(audio_sync_inputs)}/{len(required_components)})")
                    logger.info(f"   Missing components: {[comp for comp in required_components if comp not in [c for c in audio_sync_inputs]]}")

            # Wire up AI prompt generation button (using extracted handler)
            if audio_ai_generate_button is not None:
                # Wire up the button with visibility toggle for start/end mode
                def toggle_start_end_visibility(mode):
                    """Show/hide start and end prompt fields based on generation mode."""
                    is_start_end = mode == "start-to-end"
                    return gr.update(visible=is_start_end), gr.update(visible=is_start_end)

                # Wire up AI prompt generation button using explicitly assigned variables
                # (bypassing locals() check which doesn't work reliably with explicit assignments)

                # Wire up mode change to show/hide start/end prompts
                audio_ai_generation_mode.change(
                    fn=toggle_start_end_visibility,
                    inputs=[audio_ai_generation_mode],
                    outputs=[audio_ai_start_prompt, audio_ai_end_prompt]
                )
                logger.debug(f"   {emoji_if_enabled('✓')} Mode change visibility toggle wired")

                # Wire up generate button
                audio_ai_generate_button.click(
                    fn=generate_prompts_with_ai,
                    inputs=[
                        audio_ai_generation_mode,
                        audio_ai_intensity,
                        audio_ai_style,
                        audio_ai_prompt_theme,
                        audio_ai_prompt_count,
                        audio_ai_start_prompt,
                        audio_ai_end_prompt
                    ],
                    outputs=[audio_sync_prompts]
                )
                logger.debug(f"{emoji_if_enabled('✨')} AI prompt generation button connected successfully")

            # ===== ZERO-HITL BUTTON WIRING =====
            # Wire up "🔥 SLOP IT! 🔥" button
            if 'btn_slop_it' in locals() and 'zero_hitl_duration' in locals():
                from deforum.ui.handlers.zero_hitl_handler import (
                    handle_slop_it_click,
                    handle_view_settings_click,
                    handle_open_output_click
                )

                # Main SLOP IT button
                locals()['btn_slop_it'].click(
                    fn=handle_slop_it_click,
                    inputs=[
                        locals()['zero_hitl_duration'],
                        locals()['zero_hitl_theme'],
                        locals()['zero_hitl_seed']
                    ],
                    outputs=[
                        locals()['zero_hitl_status'],
                        locals()['zero_hitl_log'],
                        locals()['zero_hitl_generated_settings']
                    ]
                )

                # View settings button
                locals()['btn_view_settings'].click(
                    fn=handle_view_settings_click,
                    inputs=[locals()['zero_hitl_generated_settings']],
                    outputs=[locals()['zero_hitl_log']]
                )

                # Open output folder button
                locals()['btn_open_output'].click(
                    fn=handle_open_output_click,
                    inputs=[],
                    outputs=[locals()['zero_hitl_status']]
                )

                logger.debug(f"{emoji_if_enabled('🔥')} Zero-HITL buttons wired successfully")



    # Mode change handler - updates all UI based on selected render mode
    # Connect render mode change handler (imported from handlers module)
    render_mode.change(
        fn=handle_render_mode_change,
        inputs=[render_mode, enable_fractional_strength],
        outputs=[
            tab_depth,
            tab_shakify,
            tab_wan,
            cadence_column,
            pseudo_cadence_column,
            fps,
            steps,
            normal_strength_slider,
            keyframe_strength_slider,
            normal_strength_column,
            keyframe_strength_column,
            animation_mode
        ]
    )

    # Import event handlers from helpers module
    from deforum.ui.handlers.ui_left_handlers import (
        on_reset_to_defaults_click,
        update_slider_step_size,
        slider_to_textbox
    )

    # Connect steps and fractional checkbox changes to update slider step size
    steps.change(
        fn=update_slider_step_size,
        inputs=[steps, enable_fractional_strength],
        outputs=[normal_strength_slider, keyframe_strength_slider]
    )

    enable_fractional_strength.change(
        fn=update_slider_step_size,
        inputs=[steps, enable_fractional_strength],
        outputs=[normal_strength_slider, keyframe_strength_slider]
    )

    # Connect sliders to textboxes (ONE-WAY: slider -> textbox only)
    # This prevents infinite loops. Textbox is the source of truth.
    normal_strength_slider.change(
        fn=slider_to_textbox,
        inputs=[normal_strength_slider],
        outputs=[normal_strength]
    )

    keyframe_strength_slider.change(
        fn=slider_to_textbox,
        inputs=[keyframe_strength_slider],
        outputs=[keyframe_strength]
    )

    # Connect Reset to Defaults button - shows confirmation
    reset_to_defaults_btn.click(
        fn=lambda: gr.update(visible=True),
        inputs=[],
        outputs=[reset_confirm_row]
    )

    # Connect confirmation buttons
    reset_confirm_yes.click(
        fn=on_reset_to_defaults_click,
        inputs=[render_mode],
        outputs=[
            reset_progress,
            animation_prompts,  # Update prompts JSON
            soundtrack_path,    # Update audio path
            fps,                # Update FPS
            max_frames,         # Update frame count
            rotation_3d_y,      # Update camera path
            translation_z,      # Update camera path
            zoom,               # Update camera path
        ]
    ).then(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[reset_confirm_row]
    )

    reset_confirm_no.click(
        fn=lambda: gr.update(visible=False),
        inputs=[],
        outputs=[reset_confirm_row]
    )

    # Gradio's Change functions - hiding and renaming elements based on other elements
    show_info_on_ui.change(fn=change_css, inputs=show_info_on_ui, outputs=[gr.HTML()])
    handle_change_functions(locals())

    # Set up Wan Generate button if it exists - with better error handling
    if 'wan_generate_button' in locals() and 'wan_generation_status' in locals():
        try:
            logger.debug(f"{emoji_if_enabled('🔗')} Connecting Wan generate button...")
            
            # Import the real Wan generation function from ui_elements
            from .ui_elements import wan_generate_video as wan_generate_video_main
            
            # Get all component values to pass to the Wan generation function
            from deforum.config.args import get_component_names
            component_names = get_component_names()
            
            # Create list of all UI components in the correct order
            # Use dummy_component for None values to prevent Gradio errors
            dummy_component = gr.Button(visible=False)
            component_inputs = []
            missing_components = []
            for name in component_names:
                if name in locals():
                    component = locals()[name]
                    # Replace None with dummy_component
                    component_inputs.append(component if component is not None else dummy_component)
                else:
                    missing_components.append(name)
                    logger.warning(f"Component '{name}' not found in locals()")
                    component_inputs.append(dummy_component)  # Add dummy for missing components
            
            logger.debug(f"Found {len(component_inputs)} UI components for Wan generation", emoji='distribution')
            if missing_components:
                warning = emoji_utils.maybe_warning()
                logger.warning(f"{warning} Missing {len(missing_components)} components: {missing_components[:5]}...")
            
            # Create a wrapper function with better error handling
            def wan_generate_wrapper(*args):
                try:
                    logger.info(f"Wan generate button clicked! Received {len(args)} arguments", emoji='movie_camera')
                    logger.info("Calling wan_generate_video_main...", emoji='refresh')
                    result = wan_generate_video_main(*args)
                    logger.debug(f"{emoji_if_enabled('✅')} Wan generation completed: {str(result)[:100]}...")
                    return result
                except Exception as e:
                    cross = emoji_utils.maybe_cross()
                    error_msg = f"{cross} Wan generation error: {str(e)}"
                    logger.info(error_msg)
                    import traceback
                    traceback.print_exc()
                    return error_msg

            # Only connect if button and status exist
            if wan_generate_button is not None and wan_generation_status is not None:
                wan_generate_button.click(
                    fn=wan_generate_wrapper,
                    inputs=component_inputs,  # Pass all UI component values
                    outputs=[wan_generation_status]
                )
                logger.debug(f"{emoji_if_enabled('✅')} Wan generate button connected successfully")
            else:
                logger.warning(f"Wan generate button or status not found - skipping connection")
        except Exception as e:
            logger.error(f"Failed to connect Wan generate button: {e}")
            import traceback
            traceback.print_exc()

    # Set up Wan Prompt Enhancement button with proper wan_enhanced_prompts access
    if 'enhance_prompts_btn' in locals() and 'wan_enhanced_prompts' in locals():
        try:
            logger.debug(f"{emoji_if_enabled('🔗')} Connecting Wan prompt enhancement button...")
            
            from .ui_elements import enhance_prompts_handler
            
            # Check if enhancement_progress component exists for progress feedback
            enhancement_progress_available = 'enhancement_progress' in locals()
            
            # Connect the enhance button with current prompts as first parameter
            if enhancement_progress_available:
                # Connect with progress feedback
                locals()['enhance_prompts_btn'].click(
                    fn=enhance_prompts_handler,
                    inputs=[
                        locals()['wan_enhanced_prompts'],  # current_prompts - first parameter
                        locals()['wan_qwen_model'], 
                        locals()['wan_qwen_language'],
                        locals()['wan_qwen_auto_download']
                    ],
                    outputs=[locals()['wan_enhanced_prompts'], locals()['enhancement_progress']]
                )
                logger.debug(f"{emoji_if_enabled('✅')} Wan prompt enhancement button connected successfully with progress feedback")
            else:
                # Fallback connection without progress feedback
                def enhance_wrapper(*args):
                    result = enhance_prompts_handler(*args)
                    if isinstance(result, tuple):
                        return result[0]  # Return only the enhanced prompts
                    return result
                
                locals()['enhance_prompts_btn'].click(
                    fn=enhance_wrapper,
                    inputs=[
                        locals()['wan_enhanced_prompts'],  # current_prompts - first parameter
                        locals()['wan_qwen_model'], 
                        locals()['wan_qwen_language'],
                        locals()['wan_qwen_auto_download']
                    ],
                    outputs=[locals()['wan_enhanced_prompts']]
                )
                logger.debug(f"{emoji_if_enabled('✅')} Wan prompt enhancement button connected successfully (without progress feedback)")
        except Exception as e:
            logger.error(f"Failed to connect Wan prompt enhancement button: {e}")
            import traceback
            traceback.print_exc()

    # Set up Auto-Assign Keyframe Types button
    if 'auto_assign_keyframe_types_btn' in locals() and 'keyframe_type_schedule' in locals():
        try:
            logger.debug(f"{emoji_if_enabled('🔗')} Connecting auto-assign keyframe types button...")

            from .ui_elements import auto_assign_keyframe_types_handler

            # Check if animation_prompts and wan_flf2v_chunk_size are available
            if 'animation_prompts' in locals() and 'wan_flf2v_chunk_size' in locals():
                locals()['auto_assign_keyframe_types_btn'].click(
                    fn=auto_assign_keyframe_types_handler,
                    inputs=[
                        locals()['animation_prompts'],
                        locals()['wan_flf2v_chunk_size']
                    ],
                    outputs=[locals()['keyframe_type_schedule']]
                )
                logger.debug(f"{emoji_if_enabled('✅')} Auto-assign keyframe types button connected successfully")
            else:
                logger.warning("animation_prompts or wan_flf2v_chunk_size not found in locals()")
        except Exception as e:
            logger.error(f"Failed to connect auto-assign keyframe types button: {e}")
            import traceback
            traceback.print_exc()

    # Set up movement component references for analyze_movement_handler
    try:
        from .ui_elements import analyze_movement_handler, enhance_prompts_handler
        
        # Store references to movement schedule components to get actual schedule strings
        movement_components = {}
        movement_component_names = [
            'translation_x', 'translation_y', 'translation_z',
            'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z',
            'zoom', 'angle', 'max_frames',
            # Add Camera Shakify components
            'shake_name', 'shake_intensity', 'shake_speed'
        ]
        
        # Get actual schedule values from the UI components (these are the schedule strings)
        for comp_name in movement_component_names:
            if comp_name in locals():
                component = locals()[comp_name]
                # Get the actual value from the component
                if comp_name == 'max_frames':
                    # max_frames is a number, not a schedule
                    movement_components[comp_name] = getattr(component, 'value', 100)
                elif comp_name in ['shake_name', 'shake_intensity', 'shake_speed']:
                    # Camera Shakify settings
                    if comp_name == 'shake_name':
                        movement_components[comp_name] = getattr(component, 'value', "None")
                    elif comp_name == 'shake_intensity':
                        movement_components[comp_name] = getattr(component, 'value', 1.0)
                    elif comp_name == 'shake_speed':
                        movement_components[comp_name] = getattr(component, 'value', 1.0)
                else:
                    # These are schedule strings used by Deforum's animation system
                    movement_components[comp_name] = getattr(component, 'value', f"0:(0)")
            else:
                # Fallback defaults (same as Deforum defaults)
                if comp_name == 'max_frames':
                    movement_components[comp_name] = 100
                elif comp_name == 'zoom':
                    movement_components[comp_name] = "0:(1.0)"  # Default zoom schedule
                elif comp_name == 'shake_name':
                    movement_components[comp_name] = "None"     # Default shake disabled
                elif comp_name == 'shake_intensity':
                    movement_components[comp_name] = 1.0       # Default shake intensity
                elif comp_name == 'shake_speed':
                    movement_components[comp_name] = 1.0       # Default shake speed
                else:
                    movement_components[comp_name] = "0:(0)"    # Default movement schedule
        
        # Store the movement components dictionary for the handler
        analyze_movement_handler._movement_components = movement_components
        
        # Store reference to wan_enhanced_prompts component for updating prompts with movement
        if 'wan_enhanced_prompts' in locals():
            enhance_prompts_handler._wan_enhanced_prompts_component = locals()['wan_enhanced_prompts']
        
        # Store reference to wan_movement_description component
        if 'wan_movement_description' in locals():
            analyze_movement_handler._wan_movement_description_component = locals()['wan_movement_description']
        
        logger.debug(f"{emoji_if_enabled('✅')} Movement schedule references set up for {len(movement_components)} Deforum schedules")
        logger.info(f"Sample schedules: translation_x='{movement_components.get('translation_x', 'N/A')[:30]}...', zoom='{movement_components.get('zoom', 'N/A')}'", emoji='distribution')
        
    except Exception as e:
        logger.error(f"Failed to set up movement schedule references: {e}")
        import traceback
        traceback.print_exc()

    # Set up Wan prompt template loading buttons
    if 'load_wan_prompts_btn' in locals() and 'wan_enhanced_prompts' in locals():
        try:
            logger.debug(f"{emoji_if_enabled('🔗')} Connecting Wan prompt loading button...")
            
            from .ui_elements import load_wan_prompts_handler
            
            locals()['load_wan_prompts_btn'].click(
                fn=load_wan_prompts_handler,
                inputs=[],
                outputs=[locals()['wan_enhanced_prompts']]
            )
            logger.debug(f"{emoji_if_enabled('✅')} Wan prompt loading button connected")
        except Exception as e:
            logger.error(f"Failed to connect Wan prompt button: {e}")
    
    if 'load_deforum_prompts_btn' in locals() and 'wan_enhanced_prompts' in locals():
        try:
            logger.debug(f"{emoji_if_enabled('🔗')} Connecting Deforum prompts loading button...")
            
            from .ui_elements import load_deforum_prompts_handler
            
            locals()['load_deforum_prompts_btn'].click(
                fn=load_deforum_prompts_handler,
                inputs=[],
                outputs=[locals()['wan_enhanced_prompts']]
            )
            logger.debug(f"{emoji_if_enabled('✅')} Deforum prompts loading button connected")
        except Exception as e:
            logger.error(f"Failed to connect Deforum prompts button: {e}")
    
    # Set up load Deforum to Wan and load defaults buttons
    if 'load_deforum_to_wan_btn' in locals() and 'wan_enhanced_prompts' in locals():
        try:
            logger.debug(f"{emoji_if_enabled('🔗')} Connecting Load Deforum to Wan button...")
            
            from .ui_elements import load_deforum_to_wan_prompts_handler, enhance_prompts_handler
            
            # Store animation_prompts reference for deforum-to-wan loading
            if 'animation_prompts' in locals():
                enhance_prompts_handler._animation_prompts_component = locals()['animation_prompts']
            
            locals()['load_deforum_to_wan_btn'].click(
                fn=load_deforum_to_wan_prompts_handler,
                inputs=[],
                outputs=[locals()['wan_enhanced_prompts']]
            )
            logger.debug(f"{emoji_if_enabled('✅')} Load Deforum to Wan button connected")
        except Exception as e:
            logger.error(f"Failed to connect Load Deforum to Wan button: {e}")
    
    if 'load_wan_defaults_btn' in locals() and 'wan_enhanced_prompts' in locals():
        try:
            logger.debug(f"{emoji_if_enabled('🔗')} Connecting Load Wan Defaults button...")
            
            from .ui_elements import load_wan_defaults_handler
            
            locals()['load_wan_defaults_btn'].click(
                fn=load_wan_defaults_handler,
                inputs=[],
                outputs=[locals()['wan_enhanced_prompts']]
            )
            logger.debug(f"{emoji_if_enabled('✅')} Load Wan Defaults button connected")
        except Exception as e:
            logger.error(f"Failed to connect Load Wan Defaults button: {e}")

    # Set up Wan Model Validation buttons
    try:
        from deforum.integrations.wan.wan_model_validator import WanModelValidator
        
        # Validation functions
        def wan_validate_models():
            """Validate Wan models with HuggingFace checksums when possible"""
            # Theme-aware emoji symbols
            check = emoji_utils.maybe_check()
            cross = emoji_utils.maybe_cross()
            warning = emoji_utils.maybe_warning()
            alert = emoji_utils.maybe_alert()
            lock = emoji_utils.lock()
            folder = emoji_utils.folder()
            chart_increasing = emoji_utils.chart_increasing()
            bulb = emoji_utils.bulb()
            try:
                validator = WanModelValidator()
                models = validator.discover_models()

                if not models:
                    return f"{cross} No Wan models found for validation."

                results = []
                results.append(f"{lock} WAN MODEL VALIDATION WITH OFFICIAL CHECKSUMS")
                results.append("=" * 55)

                valid_models = 0
                total_models = len(models)

                for model in models:
                    results.append(f"\n{folder} {model['name']} ({model['size_formatted']}):")
                    
                    from pathlib import Path
                    model_path = Path(model['path'])
                    
                    # Try HuggingFace checksum validation first
                    hf_validation = validator.validate_against_huggingface_checksums(model_path)
                    
                    if hf_validation['checked_files']:
                        # HuggingFace validation was possible
                        if hf_validation['valid']:
                            valid_count = len([f for f in hf_validation['checked_files'].values() if f['status'] == 'valid'])
                            total_count = len(hf_validation['checked_files'])
                            results.append(f"   {check} VALID - {valid_count}/{total_count} files verified with official checksums")
                            valid_models += 1
                        else:
                            results.append(f"   {cross} INVALID - Checksum verification failed")
                            for error in hf_validation['errors']:
                                results.append(f"      {alert} {error}")
                    else:
                        # Fall back to basic validation if HuggingFace validation not possible
                        results.append(f"   {warning} Official checksums not available, using basic validation...")
                        validation_result = validator.validate_model_integrity(model_path)

                        if validation_result['valid']:
                            results.append(f"   {check} VALID (basic structure check)")
                            valid_models += 1
                        else:
                            results.append(f"   {cross} INVALID")
                            for error in validation_result['errors']:
                                results.append(f"      {alert} {error}")

                    if hf_validation['warnings']:
                        for warn_msg in hf_validation['warnings']:
                            results.append(f"   {warning} {warn_msg}")

                summary = f"{chart_increasing} SUMMARY: {valid_models}/{total_models} models valid"
                if valid_models == total_models:
                    summary = f"{check} {summary} - All models verified!"
                else:
                    summary = f"{warning} {summary} - Some models have issues"

                results.append(f"\n{summary}")
                results.append(f"{bulb} Using official HuggingFace checksums for maximum reliability")

                return "\n".join(results)

            except Exception as e:
                return f"{cross} Validation error: {str(e)}"

        def cleanup_invalid_models():
            """Clean up invalid models with confirmation"""
            # Theme-aware emoji symbols
            check = emoji_utils.maybe_check()
            cross = emoji_utils.maybe_cross()
            warning = emoji_utils.maybe_warning()
            magnifying_glass = emoji_utils.magnifying_glass()
            trash = emoji_utils.trash()
            try:
                validator = WanModelValidator()
                models = validator.discover_models()

                if not models:
                    return f"{cross} No models found to validate."

                # Find invalid models
                invalid_models = []
                results = []
                results.append(f"{magnifying_glass} Checking all models for corruption...")
                results.append("=" * 50)
                
                for model in models:
                    from pathlib import Path
                    model_path = Path(model['path'])
                    
                    # Use HuggingFace validation if possible, otherwise basic validation
                    hf_validation = validator.validate_against_huggingface_checksums(model_path)
                    
                    if hf_validation['checked_files']:
                        # Use HuggingFace validation result
                        is_valid = hf_validation['valid']
                        errors = hf_validation['errors']
                    else:
                        # Fall back to basic validation
                        validation_result = validator.validate_model_integrity(model_path)
                        is_valid = validation_result['valid']
                        errors = validation_result['errors']
                    
                    if not is_valid:
                        invalid_models.append({
                            'name': model['name'],
                            'path': model['path'],
                            'size': model['size_formatted'],
                            'errors': errors
                        })
                
                if not invalid_models:
                    return f"{check} All models passed validation! No cleanup needed."

                results.append(f"\n{warning} Found {len(invalid_models)} invalid model(s):")
                for i, invalid in enumerate(invalid_models, 1):
                    results.append(f"\n{i}. {invalid['name']} ({invalid['size']})")
                    results.append(f"   Issues: {', '.join(invalid['errors'])}")

                results.append(f"\n{trash} Use the 'Clean Up Invalid Models' button to remove these automatically.")
                results.append(f"{warning} This action will permanently delete the invalid model directories!")

                return "\n".join(results)

            except Exception as e:
                return f"{cross} Cleanup scan error: {str(e)}"
        
        def compute_model_checksums():
            """Compute checksums for all model files"""
            # Theme-aware emoji symbols
            check = emoji_utils.maybe_check()
            cross = emoji_utils.maybe_cross()
            lock = emoji_utils.lock()
            folder = emoji_utils.folder()
            save = emoji_utils.save()
            try:
                validator = WanModelValidator()
                models = validator.discover_models()

                if not models:
                    return f"{cross} No models found to checksum.", {}

                results = []
                results.append(f"{lock} Computing checksums for all models...")
                results.append("=" * 60)

                checksums = {}

                for model in models:
                    results.append(f"\n{folder} {model['name']}:")
                    model_checksums = {}

                    from pathlib import Path
                    model_path = Path(model['path'])

                    # Compute checksums for important files
                    important_files = [
                        "diffusion_pytorch_model.safetensors",
                        "diffusion_pytorch_model-00001-of-00007.safetensors",
                        "models_t5_umt5-xxl-enc-bf16.pth",
                        "Wan2.1_VAE.pth",
                        "config.json"
                    ]

                    for file_name in important_files:
                        file_path = model_path / file_name
                        if file_path.exists():
                            file_hash = validator.compute_file_hash(file_path)
                            if file_hash:
                                model_checksums[file_name] = file_hash
                                results.append(f"   {check} {file_name}: {file_hash[:16]}...")
                            else:
                                results.append(f"   {cross} {file_name}: Failed to compute hash")

                    checksums[model['name']] = model_checksums

                results.append(f"\n{check} Checksum computation complete!")
                results.append(f"{save} Full checksums available in the Model Details output below.")

                return "\n".join(results), checksums

            except Exception as e:
                return f"{cross} Checksum error: {str(e)}", {}
        
        def full_integrity_check():
            """Comprehensive integrity check with HuggingFace checksum validation"""
            # Theme-aware emoji symbols
            check = emoji_utils.maybe_check()
            cross = emoji_utils.maybe_cross()
            warning = emoji_utils.maybe_warning()
            alert = emoji_utils.maybe_alert()
            magnifying_glass = emoji_utils.magnifying_glass()
            folder = emoji_utils.folder()
            bulb = emoji_utils.bulb()
            save = emoji_utils.save()
            try:
                validator = WanModelValidator()
                models = validator.discover_models()

                if not models:
                    return f"{cross} No models found for integrity check.", {}

                results = []
                results.append(f"{magnifying_glass} COMPREHENSIVE INTEGRITY CHECK WITH OFFICIAL CHECKSUMS")
                results.append("=" * 70)

                all_details = {}
                overall_status = f"{check} ALL GOOD"
                
                for model in models:
                    results.append(f"\n{folder} {model['name']} ({model['size_formatted']}):")
                    results.append("-" * 50)

                    from pathlib import Path
                    model_path = Path(model['path'])

                    # Run HuggingFace checksum validation first
                    hf_validation = validator.validate_against_huggingface_checksums(model_path)

                    model_details = {
                        'path': model['path'],
                        'size': model['size_formatted'],
                        'type': model['type'],
                        'hf_checksum_validation': hf_validation,
                        'basic_validation': None
                    }

                    # Report HuggingFace validation results
                    if hf_validation['valid']:
                        checked_count = len(hf_validation['checked_files'])
                        valid_count = sum(1 for f in hf_validation['checked_files'].values() if f['status'] == 'valid')
                        results.append(f"   {check} HuggingFace Checksum Validation: {valid_count}/{checked_count} files verified")

                        for file_name, file_info in hf_validation['checked_files'].items():
                            if file_info['status'] == 'valid':
                                results.append(f"      {check} {file_name}: Official checksum verified")
                            else:
                                results.append(f"      {cross} {file_name}: Checksum mismatch")
                                overall_status = f"{warning} CHECKSUM ISSUES FOUND"
                    else:
                        results.append(f"   {cross} HuggingFace Checksum Validation: FAILED")
                        overall_status = f"{warning} CHECKSUM ISSUES FOUND"
                        for error in hf_validation['errors']:
                            results.append(f"      {alert} {error}")

                    if hf_validation['warnings']:
                        for warn_msg in hf_validation['warnings']:
                            results.append(f"      {warning} {warn_msg}")
                    
                    # Only run basic validation if HuggingFace validation had issues
                    if not hf_validation['valid'] or hf_validation['warnings']:
                        basic_validation = validator.validate_model_integrity(model_path)
                        model_details['basic_validation'] = basic_validation

                        if basic_validation['valid']:
                            results.append(f"   {check} Basic Structure Validation: PASS")
                        else:
                            results.append(f"   {cross} Basic Structure Validation: FAIL")
                            for error in basic_validation['errors']:
                                results.append(f"      {alert} {error}")

                    all_details[model['name']] = model_details

                target = emoji_utils.target()
                results.insert(1, f"{target} OVERALL STATUS: {overall_status}")
                results.append(f"\n{bulb} **Using Official HuggingFace Checksums for Maximum Reliability**")
                results.append(f"{save} Detailed results saved to Model Details output.")

                return "\n".join(results), all_details

            except Exception as e:
                return f"{cross} Integrity check error: {str(e)}", {}
        
        # Connect validation buttons if they exist
        validation_buttons = [
            ('validate_models_btn', wan_validate_models),
            ('cleanup_invalid_btn', cleanup_invalid_models),
            ('compute_checksums_btn', compute_model_checksums),
            ('verify_integrity_btn', full_integrity_check)
        ]
        
        for button_name, callback_fn in validation_buttons:
            if button_name in locals():
                # Determine output based on function return signature
                if button_name in ['compute_checksums_btn', 'verify_integrity_btn']:
                    # These functions return tuple (text, dict)
                    locals()[button_name].click(
                        fn=callback_fn,
                        inputs=[],
                        outputs=[locals()['validation_output'], locals()['model_details_output']]
                    )
                else:
                    # These functions return just text
                    locals()[button_name].click(
                        fn=callback_fn,
                        inputs=[],
                        outputs=[locals()['validation_output']]
                    )
                logger.debug(f"{emoji_if_enabled('✅')} Connected {button_name}")
            
        logger.debug(f"{emoji_if_enabled('✅')} All Wan model validation buttons connected")

    except ImportError:
        warning = emoji_utils.maybe_warning()
        logger.warning(f"{warning} WanModelValidator not available - validation buttons will not work")
    except Exception as e:
        warning = emoji_utils.maybe_warning()
        logger.error(f"{warning} Failed to set up validation buttons: {e}")

    # Camera Path button wiring moved to ui_right.py (after camera_path_plot is created)

    # Merge all tab component dicts into main locals() for component access
    result = locals().copy()

    # Flatten all tab dicts so components are accessible at top level
    tab_dicts = ['tab_zero_hitl_params', 'tab_camera_path_params', 'tab_init_params', 'tab_prompts_params',
                 'tab_keyframes_params', 'tab_depth_params', 'tab_shakify_params',
                 'tab_masking_params', 'tab_wan_params', 'tab_run_params', 'tab_output_params']

    for tab_dict_name in tab_dicts:
        if tab_dict_name in result and isinstance(result[tab_dict_name], dict):
            # Merge components from this tab dict into result
            result.update(result[tab_dict_name])

    return result
