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

import os
from deforum.config.args import DeforumOutputArgs, get_component_names, get_settings_component_names
from modules.shared import opts, state
from modules.ui import create_output_panel, wrap_gradio_call
from modules.util import open_folder
from modules.call_queue import wrap_gradio_gpu_call
from deforum.orchestration.run_deforum import run_deforum
from deforum.config.settings import save_settings, load_all_settings, load_settings_from_video, get_default_settings_path, update_settings_path
from deforum.utils.general import get_deforum_version, get_commit_date
from deforum.ui.ui_left import setup_deforum_left_side_ui
from scripts.deforum_extend_paths import deforum_sys_extend
import gradio as gr
from deforum.utils.system.logging import get_logger, emoji_if_enabled

# Initialize logger
logger = get_logger()


def get_latest_frames():
    """Poll for latest frame and depth map preview files during generation.

    Returns None, None if:
    - No generation in progress (checked via preview file age)
    - Backend is disconnected
    - No output directories exist
    """
    import glob
    from pathlib import Path
    import time

    try:
        deforum_outdir = os.path.join(os.getcwd(), 'outputs', 'deforum')

        # Find most recent directory (Deforum_TIMESTAMP pattern)
        subdirs = [d for d in glob.glob(os.path.join(deforum_outdir, "Deforum_*")) if os.path.isdir(d)]
        if not subdirs:
            return None, None

        latest_dir = max(subdirs, key=os.path.getmtime)

        # Look for fixed preview filenames
        frame_preview = os.path.join(latest_dir, "frame-preview.png")
        depth_preview = os.path.join(latest_dir, "depth-raft-preview.png")

        # Check if preview files are fresh (modified within last 5 seconds)
        # This prevents showing stale previews and stops polling when generation ends
        current_time = time.time()
        frame_is_fresh = (
            os.path.exists(frame_preview) and
            (current_time - os.path.getmtime(frame_preview)) < 5.0
        )
        depth_is_fresh = (
            os.path.exists(depth_preview) and
            (current_time - os.path.getmtime(depth_preview)) < 5.0
        )

        # Return paths only if files are fresh
        latest_frame = frame_preview if frame_is_fresh else None
        latest_depth = depth_preview if depth_is_fresh else None

        return latest_frame, latest_depth

    except Exception:
        # Silently handle errors (e.g., backend disconnected, filesystem issues)
        # This prevents error spam in the UI
        return None, None

def on_ui_tabs():
    # extend paths using sys.path.extend so we can access all of our files and folders
    deforum_sys_extend()
    # set text above generate button
    style = '"text-align:center;font-weight:bold;padding:8px 0;min-height:60px;display:block"'
    extension_url = "https://github.com/Tok/sd-forge-deforum"
    link = f"<a href='{extension_url}' target='_blank'>Zirteqs Fluxabled Fork</a>"
    extension_name = f"{link} of the Deforum Extension for WebUI Forge"

    commit_info = f"Git commit: {get_deforum_version()}"
    i1_store_backup = f"<div style={style}>{extension_name}<br>Version: {get_commit_date()} | {commit_info}</div>"
    i1_store = i1_store_backup

    # Slopcore gradient aesthetic for Generate button and hide unwanted buttons
    slopcore_css = """
    /* Slopcore gradient for Generate button and audio sync buttons */
    /* NUCLEAR OPTION: Override Gradio 4 default button styles with maximum specificity */
    /* Target by ID with maximum specificity */
    #deforum_generate,
    #deforum_generate *,
    #deforum_generate.primary,
    #deforum_generate.secondary,
    button#deforum_generate,
    #deforum_generate > button,
    #deforum_generate button,
    [id*="deforum_generate"] button,
    [id="deforum_generate"],
    #audio_sync_button,
    #audio_sync_button *,
    button#audio_sync_button,
    #audio_sync_button > button,
    [id*="audio_sync_button"] button,
    #audio_sync_button button,
    #audio_sync_fewer_button,
    #audio_sync_fewer_button *,
    button#audio_sync_fewer_button,
    #audio_sync_fewer_button > button,
    [id*="audio_sync_fewer_button"] button,
    #audio_sync_fewer_button button,
    #audio_sync_more_button,
    #audio_sync_more_button *,
    button#audio_sync_more_button,
    #audio_sync_more_button > button,
    [id*="audio_sync_more_button"] button,
    #audio_sync_more_button button,
    #audio_ai_generate_button,
    #audio_ai_generate_button *,
    button#audio_ai_generate_button,
    #audio_ai_generate_button > button,
    [id*="audio_ai_generate_button"] button,
    #audio_ai_generate_button button,
    .slopcore-button,
    .slopcore-button *,
    .slopcore-button button,
    div.slopcore-button button,
    button.slopcore-button,
    *[class*="slopcore-button"],
    *[class*="slopcore-button"] button,
    .generate-box-generating,
    .generate-box-interrupting {
        background: linear-gradient(135deg, #5606ff 0%, #17a7fe 100%) !important;
        background-image: linear-gradient(135deg, #5606ff 0%, #17a7fe 100%) !important;
        background-color: #5606ff !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
        box-shadow: 0 4px 6px rgba(86, 6, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    /* Hover states with universal selectors */
    #deforum_generate:hover,
    #deforum_generate *:hover,
    button#deforum_generate:hover,
    #deforum_generate > button:hover,
    [id*="deforum_generate"] button:hover,
    #audio_sync_button:hover,
    #audio_sync_button *:hover,
    button#audio_sync_button:hover,
    #audio_sync_button > button:hover,
    [id*="audio_sync_button"] button:hover,
    #audio_sync_button button:hover,
    #audio_sync_fewer_button:hover,
    #audio_sync_fewer_button *:hover,
    button#audio_sync_fewer_button:hover,
    #audio_sync_fewer_button > button:hover,
    [id*="audio_sync_fewer_button"] button:hover,
    #audio_sync_fewer_button button:hover,
    #audio_sync_more_button:hover,
    #audio_sync_more_button *:hover,
    button#audio_sync_more_button:hover,
    #audio_sync_more_button > button:hover,
    [id*="audio_sync_more_button"] button:hover,
    #audio_sync_more_button button:hover,
    #audio_ai_generate_button:hover,
    #audio_ai_generate_button *:hover,
    button#audio_ai_generate_button:hover,
    #audio_ai_generate_button > button:hover,
    [id*="audio_ai_generate_button"] button:hover,
    #audio_ai_generate_button button:hover,
    .slopcore-button:hover,
    .slopcore-button *:hover,
    .slopcore-button button:hover,
    div.slopcore-button button:hover,
    button.slopcore-button:hover,
    *[class*="slopcore-button"]:hover,
    *[class*="slopcore-button"] button:hover {
        background: linear-gradient(135deg, #17a7fe 0%, #5606ff 100%) !important;
        background-image: linear-gradient(135deg, #17a7fe 0%, #5606ff 100%) !important;
        background-color: #17a7fe !important;
        box-shadow: 0 6px 12px rgba(86, 6, 255, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Hide ALL unwanted buttons in deforum results - keep only folder button */
    #deforum_results button[id*="save"],
    #deforum_results button[id*="send"],
    #save_deforum,
    #save_zip_deforum,
    #deforum_send_to_img2img,
    #deforum_send_to_inpaint,
    #deforum_send_to_extras,
    [id^="save_"][id$="_deforum"],
    [id^="deforum_send_"] {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    /* Fixed layout for generate box - prevent buttons from collapsing */
    #deforum_generate_box {
        display: grid !important;
        grid-template-columns: 200px 1fr !important;
        gap: 10px !important;
    }

    /* Buttons column - stack vertically */
    #deforum_generate_box > div:first-child {
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
        min-width: 200px !important;
        max-width: 200px !important;
    }

    /* Depth gallery column */
    #deforum_generate_box > div:last-child {
        min-height: 200px !important;
    }

    /* Style both preview images consistently with rounded corners */
    #deforum_live_preview, #deforum_depth_preview {
        max-height: 200px !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    #deforum_live_preview .image-container,
    #deforum_depth_preview .image-container {
        border-radius: 8px !important;
        overflow: hidden !important;
    }

    #deforum_live_preview img,
    #deforum_depth_preview img {
        border-radius: 8px !important;
        display: block !important;
    }
    """

    with gr.Blocks(analytics_enabled=False, css=slopcore_css) as deforum_interface:
        components = {}
        dummy_component = gr.Button(visible=False)
        with gr.Row(elem_id='deforum_progress_row', equal_height=False, variant='compact'):
            with gr.Column(scale=1.618, variant='panel'):  # Golden ratio - more space for controls
                # setting the left side of the ui:
                components = setup_deforum_left_side_ui()
            with gr.Column(scale=1, variant='compact'):  # Right side preview column
                with gr.Row(variant='compact'):
                    i1 = gr.HTML(i1_store, elem_id='deforum_header')
                id_part = 'deforum'

                # Use Deforum-specific output directory (hidden - only for folder button access)
                deforum_outdir = os.path.join(os.getcwd(), 'outputs', 'deforum')
                os.makedirs(deforum_outdir, exist_ok=True)

                # Create hidden output panel (we only need it for the folder button reference)
                with gr.Row(visible=False):
                    res = create_output_panel("deforum", deforum_outdir)
                    generation_info = res.generation_info
                    html_info= res.html_log
                    deforum_gallery = res.gallery

                # Live preview - show latest frame during generation
                live_preview_image = gr.Image(
                    label="Frame Preview",
                    show_label=True,
                    elem_id="deforum_live_preview",
                    type="filepath",
                    interactive=False,
                    visible=True,
                    height=200
                )

                # Buttons and Depth Preview side by side
                with gr.Row(elem_id=f"{id_part}_generate_box", variant='compact'):
                    # Left: Buttons stacked vertically
                    with gr.Column(scale=1, min_width=200):
                        skip = gr.Button('Pause/Resume', elem_id=f"{id_part}_skip", visible=False)
                        interrupt = gr.Button('Interrupt', elem_id=f"{id_part}_interrupt", visible=True)
                        interrupting = gr.Button('Interrupting...', elem_id=f"{id_part}_interrupting", elem_classes="generate-box-interrupting", tooltip="Interrupting generation...")
                        submit = gr.Button('Generate', elem_id=f"{id_part}_generate", variant='primary')

                        skip.click(
                            fn=lambda: state.skip(),
                            inputs=[],
                            outputs=[],
                        )

                        interrupt.click(
                            fn=lambda: state.interrupt(),
                            inputs=[],
                            outputs=[],
                        )

                        interrupting.click(
                            fn=lambda: state.interrupt(),
                            inputs=[],
                            outputs=[],
                        )

                    # Right: Depth preview (compact, beside buttons)
                    with gr.Column(scale=2):
                        depth_preview_image = gr.Image(
                            label="Depth & Flow Preview",
                            show_label=True,
                            elem_id="deforum_depth_preview",
                            type="filepath",
                            interactive=False,
                            visible=False,
                            height=200
                        )

                        components['depth_preview_image'] = depth_preview_image

                with gr.Row(variant='compact'):
                    settings_path = gr.Textbox(get_default_settings_path(), elem_id='deforum_settings_path', label="Settings File", info="Settings are automatically loaded on startup. Path can be relative to webui folder OR full/absolute.", lines=3, max_lines=3)
                with gr.Row(variant='compact'):
                    save_settings_btn = gr.Button('Save Settings', elem_id='deforum_save_settings_btn')
                    load_settings_btn = gr.Button('Load All Settings', elem_id='deforum_load_settings_btn')
                    open_folder_btn = gr.Button('📂 Open Output Directory', elem_id='deforum_open_folder_btn')

                with gr.Row(variant='compact'):
                    video_upload = gr.File(
                        label="Load Settings from Video (ComfyUI-style metadata extraction)",
                        file_types=[".mp4", ".mov", ".avi", ".webm", ".mkv"],
                        type="filepath",
                        elem_id="deforum_video_upload",
                        file_count="single"
                    )

                # Camera Path Visualization (real-time display)
                with gr.Row(variant='compact'):
                    camera_path_plot = gr.Plot(
                        label="Camera Path Visualization (Real-time)",
                        show_label=True,
                        elem_id="deforum_camera_path_viz",
                        visible=True
                    )

                components['camera_path_plot'] = camera_path_plot

        # Live preview polling - updates every 500ms
        # Smart polling: only shows fresh previews (< 5 sec old), silently handles errors
        live_preview_timer = gr.Timer(value=0.5, active=True)
        live_preview_timer.tick(
            fn=get_latest_frames,
            inputs=[],
            outputs=[live_preview_image, depth_preview_image]
        )

        # Check if Flux blocker is active (minimal component set)
        is_flux_blocker_active = len(components) < 10  # Minimal set has only ~2 components

        if is_flux_blocker_active:
            logger.info("Flux blocker active - Deforum UI will show setup instructions")
            # In blocker mode, just use what components we have
            component_list = [components.get(name, dummy_component) for name in ['show_info_on_ui']]
        else:
            # Normal mode - get all components (use .get() with dummy_component fallback for None values)
            component_names_needed = get_component_names()
            component_list = [components.get(name, dummy_component) or dummy_component for name in component_names_needed]

        submit.click(
                    fn=wrap_gradio_gpu_call(run_deforum),
                    _js="submit_deforum",
                    inputs=[dummy_component, dummy_component] + component_list,
                    outputs=[
                         deforum_gallery,
                         components.get("resume_timestring", dummy_component),
                         generation_info,
                         html_info
                    ],
                )
        
        settings_component_list = [components.get(name, dummy_component) or dummy_component for name in get_settings_component_names()]
        video_settings_component_list = [components.get(name, dummy_component) or dummy_component for name in list(DeforumOutputArgs().keys())]

        save_settings_btn.click(
            fn=wrap_gradio_call(save_settings),
            inputs=[settings_path] + settings_component_list + video_settings_component_list,
            outputs=[],
        )
        
        # Create a path update function
        def path_updating_load_settings(*args):
            path = args[0]
            settings_path.value = path
            return load_all_settings(*args)
            
        load_settings_btn.click(
            fn=wrap_gradio_call(path_updating_load_settings),
            inputs=[settings_path] + settings_component_list,
            outputs=settings_component_list,
        )

        # Video upload for metadata extraction
        video_upload.upload(
            fn=wrap_gradio_call(load_settings_from_video),
            inputs=[video_upload] + settings_component_list,
            outputs=settings_component_list,
        )

        # Open output folder button
        open_folder_btn.click(
            fn=lambda: open_folder(deforum_outdir),
            inputs=[],
            outputs=[],
        )

        # Wire up Camera Path buttons and schedule visualization
        try:
            from deforum.ui.handlers.camera_path_generator import (
                handle_generate_preset,
                handle_generate_custom
            )
            from deforum.utils.schedule_visualizer import visualize_schedules

            btn_generate_preset = components.get('btn_generate_preset')
            btn_randomize_preset = components.get('btn_randomize_preset')
            btn_generate_custom = components.get('btn_generate_custom')

            # Get schedule textboxes
            tx = components.get('translation_x')
            ty = components.get('translation_y')
            tz = components.get('translation_z')
            rx = components.get('rotation_3d_x')
            ry = components.get('rotation_3d_y')
            rz = components.get('rotation_3d_z')
            animation_prompts = components.get('animation_prompts')

            if btn_generate_preset and camera_path_plot and tx:
                # Wire up preset generation button
                btn_generate_preset.click(
                    fn=handle_generate_preset,
                    inputs=[
                        components.get('preset_type'),
                        components.get('preset_radius'),
                        components.get('preset_height'),
                        components.get('preset_rotation_factor'),
                        components.get('preset_num_frames'),
                        components.get('preset_closed_loop'),
                        components.get('preset_randomize'),
                        components.get('preset_random_seed'),
                        tx, ty, tz, rx, ry, rz
                    ],
                    outputs=[
                        components.get('preset_status'),
                        tx, ty, tz, rx, ry, rz  # Only update schedules
                    ]
                )
                logger.debug("✅ Camera Path preset button wired to right panel plot")

            if btn_randomize_preset and camera_path_plot and tx:
                # Wire up randomize button
                def randomize_preset_wrapper(*args):
                    """Randomize by using current params but with random seed"""
                    args_list = list(args)
                    args_list[7] = -1  # preset_random_seed index - force new randomization
                    if args_list[6] == 0:  # preset_randomize
                        args_list[6] = 0.5
                    return handle_generate_preset(*args_list)

                btn_randomize_preset.click(
                    fn=randomize_preset_wrapper,
                    inputs=[
                        components.get('preset_type'),
                        components.get('preset_radius'),
                        components.get('preset_height'),
                        components.get('preset_rotation_factor'),
                        components.get('preset_num_frames'),
                        components.get('preset_closed_loop'),
                        components.get('preset_randomize'),
                        components.get('preset_random_seed'),
                        tx, ty, tz, rx, ry, rz
                    ],
                    outputs=[
                        components.get('preset_status'),
                        tx, ty, tz, rx, ry, rz  # Only update schedules
                    ]
                )
                logger.debug("✅ Camera Path randomize button wired")

            if btn_generate_custom and camera_path_plot and tx:
                # Wire up custom spline generation button
                btn_generate_custom.click(
                    fn=handle_generate_custom,
                    inputs=[
                        components.get('num_control_points'),
                        components.get('spline_type'),
                        components.get('spline_smoothness'),
                        components.get('look_at_curve'),
                        components.get('custom_num_frames'),
                        components.get('custom_closed_loop'),
                        components.get('control_point_pattern'),
                        components.get('control_pattern_scale'),
                        tx, ty, tz, rx, ry, rz
                    ],
                    outputs=[
                        components.get('custom_status'),
                        tx, ty, tz, rx, ry, rz  # Only update schedules
                    ]
                )
                logger.debug("✅ Camera Path custom button wired")

            # Wire schedule textboxes to update visualization whenever they change
            def update_viz_from_schedules(tx_val, ty_val, tz_val, rx_val, ry_val, rz_val, prompts_val="", max_frames=333):
                """Update visualization from schedule textbox values."""
                fig, stats = visualize_schedules(tx_val, ty_val, tz_val, rx_val, ry_val, rz_val, max_frames, prompts_val)
                return fig

            # Each schedule textbox triggers visualization update
            if tx and camera_path_plot:
                schedule_inputs = [tx, ty, tz, rx, ry, rz]
                if animation_prompts:
                    schedule_inputs.append(animation_prompts)

                for schedule_box in [tx, ty, tz, rx, ry, rz]:
                    if schedule_box:
                        schedule_box.change(
                            fn=update_viz_from_schedules,
                            inputs=schedule_inputs,
                            outputs=[camera_path_plot]
                        )

                # Also trigger on prompt changes (for keyframe markers)
                if animation_prompts:
                    animation_prompts.change(
                        fn=update_viz_from_schedules,
                        inputs=schedule_inputs,
                        outputs=[camera_path_plot]
                    )

                # Trigger visualization on settings load
                load_settings_btn.click(
                    fn=update_viz_from_schedules,
                    inputs=schedule_inputs,
                    outputs=[camera_path_plot]
                )

                # Also trigger on video upload (metadata extraction)
                video_upload.upload(
                    fn=update_viz_from_schedules,
                    inputs=schedule_inputs,
                    outputs=[camera_path_plot]
                )

                logger.debug("✅ Schedule textboxes wired to visualization")

        except Exception as e:
            logger.error(f"Failed to wire Camera Path buttons to right panel: {e}")
            import traceback
            traceback.print_exc()

        # Depth preview visibility toggle based on animation_mode
        def update_depth_preview_visibility(save_depth, anim_mode):
            # Show depth preview in 3D mode (depth maps are always generated for warping)
            should_show = anim_mode == '3D'
            return gr.update(visible=should_show)

        # Only bind events if components exist (skip in blocker mode)
        if 'save_depth_maps' in components and 'animation_mode' in components:
            components['save_depth_maps'].change(
                fn=update_depth_preview_visibility,
                inputs=[components['save_depth_maps'], components['animation_mode']],
                outputs=[depth_preview_image]
            )

            components['animation_mode'].change(
                fn=update_depth_preview_visibility,
                inputs=[components['save_depth_maps'], components['animation_mode']],
                outputs=[depth_preview_image]
            )

            # Also update visibility when settings are loaded
            load_settings_btn.click(
                fn=update_depth_preview_visibility,
                inputs=[components['save_depth_maps'], components['animation_mode']],
                outputs=[depth_preview_image]
            )

            # And when video is uploaded (metadata extraction)
            video_upload.upload(
                fn=update_depth_preview_visibility,
                inputs=[components['save_depth_maps'], components['animation_mode']],
                outputs=[depth_preview_image]
            )

    # handle settings loading on UI launch
    def trigger_load_general_settings():
        logger.info("Loading general settings...")
        
        # First check if deforum_settings.txt exists in webui root
        import os
        from modules import paths_internal
        webui_root_settings = os.path.join(paths_internal.script_path, "deforum_settings.txt")
        
        # Determine the settings file to load
        if os.path.isfile(webui_root_settings):
            # Use the settings file from webui root if it exists
            settings_file_path = webui_root_settings
            logger.info(f"Loading existing settings from webui root: {settings_file_path}")
        else:
            # Fall back to default settings provided by the extension
            settings_file_path = get_default_settings_path()
            logger.info(f"No settings found in webui root, using default settings from: {settings_file_path}")
        
        # Update the settings path field with the path
        settings_path.value = settings_file_path
        
        # Now call load_all_settings with ui_launch=True to update all components
        wrapped_fn = wrap_gradio_call(lambda *args, **kwargs: load_all_settings(*args, ui_launch=True, **kwargs))
        inputs = [settings_file_path] + [component.value for component in settings_component_list]
        outputs = settings_component_list
        updated_values = wrapped_fn(*inputs, *outputs)[0]
        
        # Update all the component values
        settings_component_name_to_obj = {name: component for name, component in zip(get_settings_component_names(), settings_component_list)}
        for key, value in updated_values.items():
            if key in settings_component_name_to_obj:
                settings_component_name_to_obj[key].value = value['value']

        # Update depth preview visibility based on loaded settings (skip in blocker mode)
        if 'animation_mode' in components:
            anim_mode = components['animation_mode'].value
            should_show = anim_mode == '3D'
            depth_preview_image.visible = should_show
            logger.info(f"Depth preview gallery: visible={should_show} (anim_mode={anim_mode})")

        # Update camera path visualization on startup
        try:
            from deforum.utils.schedule_visualizer import visualize_schedules
            tx = components.get('translation_x')
            ty = components.get('translation_y')
            tz = components.get('translation_z')
            rx = components.get('rotation_3d_x')
            ry = components.get('rotation_3d_y')
            rz = components.get('rotation_3d_z')
            prompts = components.get('animation_prompts')

            if tx and ty and tz and rx and ry and rz and camera_path_plot:
                fig, _ = visualize_schedules(
                    tx.value or "",
                    ty.value or "",
                    tz.value or "",
                    rx.value or "",
                    ry.value or "",
                    rz.value or "",
                    333,
                    prompts.value if prompts else ""
                )
                camera_path_plot.value = fig
                logger.info(f"{emoji_if_enabled('✅')} Camera path visualization initialized on startup")
        except Exception as e:
            logger.warning(f"Failed to initialize camera path visualization: {e}")

    # Always load settings on startup - either from persistent settings path (if enabled),
    # from webui root, or from the extension's default settings
    trigger_load_general_settings()
        
    return [(deforum_interface, "Deforum", "deforum_interface")]