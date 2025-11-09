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
from deforum.utils.system.logging import emoji
from deforum.config.settings import (
    save_settings,
    load_all_settings,
    load_settings_from_video,
    get_default_settings_path,
    update_settings_path,
)
from deforum.utils.general import get_deforum_version, get_commit_date
from deforum.ui.ui_left import setup_deforum_left_side_ui
from scripts.deforum_extend_paths import deforum_sys_extend
import gradio as gr
from deforum.utils.system.logging import get_logger, emoji_if_enabled

# Initialize logger
logger = get_logger()


# State tracking to prevent UI flashing when no previews available
_last_preview_state = {"frame": None, "depth": None}


def get_latest_frames():
    """Poll for latest frame and depth map preview files during generation.

    Uses state tracking to prevent UI updates when previews haven't changed.
    This eliminates blinking/flashing when backend disconnects or no generation active.

    Returns gr.skip() for both outputs if state unchanged to avoid UI updates.
    """
    import glob
    from pathlib import Path
    import time
    import gradio as gr

    try:
        deforum_outdir = os.path.join(os.getcwd(), "outputs", "deforum")

        # Find most recent directory (Deforum_TIMESTAMP pattern)
        subdirs = [
            d for d in glob.glob(os.path.join(deforum_outdir, "Deforum_*")) if os.path.isdir(d)
        ]
        if not subdirs:
            # No output directories - check if state unchanged
            if _last_preview_state["frame"] is None and _last_preview_state["depth"] is None:
                return gr.skip(), gr.skip()  # Skip update if already showing nothing
            _last_preview_state["frame"] = None
            _last_preview_state["depth"] = None
            return None, None

        latest_dir = max(subdirs, key=os.path.getmtime)

        # Look for fixed preview filenames
        frame_preview = os.path.join(latest_dir, "frame-preview.png")
        depth_preview = os.path.join(latest_dir, "depth-raft-preview.png")

        # Check if preview files are fresh (modified within last 5 seconds)
        # This prevents showing stale previews and stops polling when generation ends
        current_time = time.time()
        frame_is_fresh = (
            os.path.exists(frame_preview) and (current_time - os.path.getmtime(frame_preview)) < 5.0
        )
        depth_is_fresh = (
            os.path.exists(depth_preview) and (current_time - os.path.getmtime(depth_preview)) < 5.0
        )

        # Determine new preview paths
        latest_frame = frame_preview if frame_is_fresh else None
        latest_depth = depth_preview if depth_is_fresh else None

        # Check if state changed - skip update if unchanged to prevent flashing
        if (
            latest_frame == _last_preview_state["frame"]
            and latest_depth == _last_preview_state["depth"]
        ):
            return gr.skip(), gr.skip()

        # Update state
        _last_preview_state["frame"] = latest_frame
        _last_preview_state["depth"] = latest_depth

        return latest_frame, latest_depth

    except Exception:
        # Silently handle errors (e.g., backend disconnected, filesystem issues)
        # Only update UI if state actually changes
        if _last_preview_state["frame"] is None and _last_preview_state["depth"] is None:
            return gr.skip(), gr.skip()  # Skip update if already showing nothing
        _last_preview_state["frame"] = None
        _last_preview_state["depth"] = None
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
    i1_store_backup = (
        f"<div style={style}>{extension_name}<br>Version: {get_commit_date()} | {commit_info}</div>"
    )
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
        with gr.Row(elem_id="deforum_progress_row", equal_height=False, variant="compact"):
            with gr.Column(scale=1.618, variant="panel"):  # Golden ratio - more space for controls
                # setting the left side of the ui:
                components = setup_deforum_left_side_ui()
            with gr.Column(scale=1, variant="compact"):  # Right side preview column
                with gr.Row(variant="compact"):
                    i1 = gr.HTML(i1_store, elem_id="deforum_header")
                id_part = "deforum"

                # Use Deforum-specific output directory
                deforum_outdir = os.path.join(os.getcwd(), "outputs", "deforum")
                os.makedirs(deforum_outdir, exist_ok=True)

                # Create output panel with gallery for viewing finished videos
                with gr.Row():
                    res = create_output_panel("deforum", deforum_outdir)
                    generation_info = res.generation_info
                    html_info = res.html_log
                    deforum_gallery = res.gallery

                # Live preview - show latest frame during generation
                live_preview_image = gr.Image(
                    label="Frame Preview",
                    show_label=True,
                    elem_id="deforum_live_preview",
                    type="filepath",
                    interactive=False,
                    visible=True,
                    height=200,
                )

                # Buttons and Depth Preview side by side
                with gr.Row(elem_id=f"{id_part}_generate_box", variant="compact"):
                    # Left: Buttons stacked vertically
                    with gr.Column(scale=1, min_width=200):
                        skip = gr.Button("Pause/Resume", elem_id=f"{id_part}_skip", visible=False)
                        interrupt = gr.Button(
                            "Interrupt", elem_id=f"{id_part}_interrupt", visible=True
                        )
                        interrupting = gr.Button(
                            "Interrupting...",
                            elem_id=f"{id_part}_interrupting",
                            elem_classes="generate-box-interrupting",
                            tooltip="Interrupting generation...",
                        )
                        submit = gr.Button(
                            "Generate", elem_id=f"{id_part}_generate", variant="primary"
                        )

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
                            height=200,
                        )

                        components["depth_preview_image"] = depth_preview_image

                with gr.Row(variant="compact"):
                    settings_path = gr.Textbox(
                        get_default_settings_path(),
                        elem_id="deforum_settings_path",
                        label="Settings File",
                        info="Settings are automatically loaded on startup. Path can be relative to webui folder OR full/absolute.",
                        lines=3,
                        max_lines=3,
                    )
                with gr.Row(variant="compact"):
                    save_settings_btn = gr.Button(
                        "Save Settings", elem_id="deforum_save_settings_btn"
                    )
                    load_settings_btn = gr.Button(
                        "Load All Settings", elem_id="deforum_load_settings_btn"
                    )
                    folder_emoji = emoji.open_folder()
                    if folder_emoji:
                        folder_emoji += " "
                    open_folder_btn = gr.Button(
                        f"{folder_emoji}Open Output Directory", elem_id="deforum_open_folder_btn"
                    )

                # Camera Path Visualization (real-time display)
                with gr.Row(variant="compact"):
                    camera_emoji = emoji.movie_camera()
                    if camera_emoji:
                        camera_emoji += " "
                    gr.Markdown(f"### {camera_emoji}Camera Path (3D Spline)")
                    show_shakify_in_camera_path = gr.Checkbox(
                        value=False,
                        label="Show Shakify Preview",
                        info="Add subtle camera shake overlay to visualization",
                        scale=0,
                    )
                with gr.Row(variant="compact"):
                    camera_path_plot = gr.Plot(
                        label="Camera Path Visualization (Real-time)",
                        show_label=False,
                        elem_id="deforum_camera_path_viz",
                        visible=True,
                    )

                # Frame Overlap Simulator (shows preservation/novelty metrics)
                with gr.Row(variant="compact"):
                    frame_emoji = emoji.purple_square()
                    if frame_emoji:
                        frame_emoji += " "
                    gr.Markdown(f"### {frame_emoji}Frame Overlap Simulator (Worm Trail)")
                    show_shakify_in_overlap = gr.Checkbox(
                        value=False,
                        label="Show Shakify Preview",
                        info="Add subtle camera shake overlay to visualization",
                        scale=0,
                    )
                with gr.Row(variant="compact"):
                    frame_overlap_simulator = gr.HTML(
                        value='<div style="padding: 20px; color: #C8C8DC;">Loading frame overlap simulator...</div>',
                        label="Frame Overlap Simulator",
                        show_label=False,
                        elem_id="deforum_frame_overlap_sim",
                        visible=True,
                    )

                # Path Analysis & Optimization (depth warping suitability)
                with gr.Row(variant="compact"):
                    analysis_emoji = emoji.distribution()
                    if analysis_emoji:
                        analysis_emoji += " "
                    gr.Markdown(f"### {analysis_emoji}Path Analysis & Optimization")
                with gr.Row(variant="compact"):
                    analyze_emoji = emoji.distribution()
                    if analyze_emoji:
                        analyze_emoji += " "
                    optimize_emoji = emoji.gear()
                    if optimize_emoji:
                        optimize_emoji += " "
                    analyze_path_btn = gr.Button(
                        f"{analyze_emoji}Analyze Camera Path",
                        elem_id="deforum_analyze_path_btn",
                        variant="secondary",
                    )
                    optimize_path_btn = gr.Button(
                        f"{optimize_emoji}Auto-optimize for Depth Warping",
                        elem_id="deforum_optimize_path_btn",
                        variant="primary",
                    )
                with gr.Row(variant="compact"):
                    path_analysis_output = gr.Markdown(
                        value="",
                        label="Analysis Results",
                        elem_id="deforum_path_analysis_output",
                        visible=True,
                    )

                components["camera_path_plot"] = camera_path_plot
                components["frame_overlap_simulator"] = frame_overlap_simulator
                components["show_shakify_in_camera_path"] = show_shakify_in_camera_path
                components["show_shakify_in_overlap"] = show_shakify_in_overlap
                components["analyze_path_btn"] = analyze_path_btn
                components["optimize_path_btn"] = optimize_path_btn
                components["path_analysis_output"] = path_analysis_output

        # Camera Path visualization - load on UI startup (independent of tab selection)
        if camera_path_plot:
            from deforum.utils.schedule_visualizer import visualize_schedules

            def update_camera_path_visualization(
                tx,
                ty,
                tz,
                rx,
                ry,
                rz,
                prompts,
                shake_name_val,
                shake_intensity_val,
                shake_speed_val,
                apply_shakify_toggle,
            ):
                """Update camera path visualization with optional shakify overlay."""
                try:
                    # Shakify params with defaults
                    shake_name = shake_name_val if shake_name_val else "None"
                    shake_intensity = float(shake_intensity_val) if shake_intensity_val else 1.0
                    shake_speed = float(shake_speed_val) if shake_speed_val else 1.0

                    fig, _ = visualize_schedules(
                        tx or "",
                        ty or "",
                        tz or "",
                        rx or "",
                        ry or "",
                        rz or "",
                        333,  # max_frames default
                        prompts or "",
                        shake_name=shake_name,
                        shake_intensity=shake_intensity,
                        shake_speed=shake_speed,
                        target_fps=60,
                        apply_shakify=apply_shakify_toggle,  # Toggle control
                    )
                    return fig
                except Exception as e:
                    logger.warning(f"Failed to update camera path visualization: {e}")
                    return None

            # Load visualization on UI startup (not on tab selection)
            deforum_interface.load(
                fn=update_camera_path_visualization,
                inputs=[
                    components.get("translation_x"),
                    components.get("translation_y"),
                    components.get("translation_z"),
                    components.get("rotation_3d_x"),
                    components.get("rotation_3d_y"),
                    components.get("rotation_3d_z"),
                    components.get("animation_prompts"),
                    components.get("shake_name"),
                    components.get("shake_intensity"),
                    components.get("shake_speed"),
                    components.get("show_shakify_in_camera_path"),
                ],
                outputs=[camera_path_plot],
            )

        # Frame Overlap Simulator - load on UI startup (independent of tab selection)
        if frame_overlap_simulator:
            from deforum.ui.handlers.frame_overlap_handler import update_frame_overlap_visualization
            from deforum.ui.handlers.camera_path_generator import generate_preset_path

            def update_overlap_viz(
                tx,
                ty,
                tz,
                rx,
                ry,
                rz,
                width_val,
                height_val,
                shake_name_val,
                shake_intensity_val,
                shake_speed_val,
                apply_shakify_toggle,
            ):
                """Update frame overlap visualization with optional shakify overlay."""
                # Guard against empty inputs during UI initialization
                if tx is None and ty is None and tz is None:
                    return '<div style="padding: 20px; color: #C8C8DC;">Loading frame overlap simulator...</div>'

                # Get max_frames from motion settings if available, otherwise default to 333
                max_frames = 333
                width = int(width_val) if width_val else 1920
                height = int(height_val) if height_val else 1080

                # Shakify params with defaults (only apply if toggle is on)
                if apply_shakify_toggle:
                    shake_name = shake_name_val if shake_name_val else "None"
                    shake_intensity = float(shake_intensity_val) if shake_intensity_val else 1.0
                    shake_speed = float(shake_speed_val) if shake_speed_val else 1.0
                else:
                    shake_name = "None"  # Disable shakify
                    shake_intensity = 1.0
                    shake_speed = 1.0

                return update_frame_overlap_visualization(
                    translation_x=tx or "",
                    translation_y=ty or "",
                    translation_z=tz or "",
                    rotation_3d_x=rx or "",
                    rotation_3d_y=ry or "",
                    rotation_3d_z=rz or "",
                    max_frames=max_frames,
                    width=width,
                    height=height,
                    shake_name=shake_name,
                    shake_intensity=shake_intensity,
                    shake_speed=shake_speed,
                    target_fps=60,
                )

            def init_overlap_viz_with_preset():
                """Initialize frame overlap visualization with default 'rotate-around' preset."""
                # Generate default rotate-around path with smaller radius to reduce rotation
                _, schedules, _ = generate_preset_path(
                    preset_type="rotate-around",
                    radius=30.0,  # Reduced from 100 to minimize rotation
                    height=0.0,
                    num_frames=333,
                    closed_loop=True,
                    speed_multiplier=0.5,  # Slower movement
                    speed_randomization=0.0,
                )

                # Generate visualization with preset schedules (NO shakify on init)
                return update_frame_overlap_visualization(
                    translation_x=schedules.get("translation_x", "0:(0)"),
                    translation_y=schedules.get("translation_y", "0:(0)"),
                    translation_z=schedules.get("translation_z", "0:(0)"),
                    rotation_3d_x=schedules.get("rotation_3d_x", "0:(0)"),
                    rotation_3d_y=schedules.get("rotation_3d_y", "0:(0)"),
                    rotation_3d_z=schedules.get("rotation_3d_z", "0:(0)"),
                    max_frames=333,
                    width=1920,
                    height=1080,
                    shake_name="None",  # Explicitly disable shakify on init
                    shake_intensity=1.0,
                    shake_speed=1.0,
                    target_fps=60,
                )

            # Load visualization on UI startup with default preset
            deforum_interface.load(
                fn=init_overlap_viz_with_preset, inputs=[], outputs=[frame_overlap_simulator]
            )

            # Update visualization when schedules change
            schedule_components = [
                components.get("translation_x"),
                components.get("translation_y"),
                components.get("translation_z"),
                components.get("rotation_3d_x"),
                components.get("rotation_3d_y"),
                components.get("rotation_3d_z"),
            ]

            # Also add shakify components to trigger updates
            shakify_components = [
                components.get("shake_name"),
                components.get("shake_intensity"),
                components.get("shake_speed"),
            ]

            # Collect all inputs for visualization update
            viz_inputs = [
                components.get("translation_x"),
                components.get("translation_y"),
                components.get("translation_z"),
                components.get("rotation_3d_x"),
                components.get("rotation_3d_y"),
                components.get("rotation_3d_z"),
                components.get("W"),
                components.get("H"),
                components.get("shake_name"),
                components.get("shake_intensity"),
                components.get("shake_speed"),
                components.get("show_shakify_in_overlap"),  # Toggle control
            ]

            # Wire up change handlers for all schedule fields
            for schedule_component in schedule_components:
                if schedule_component:
                    schedule_component.change(
                        fn=update_overlap_viz, inputs=viz_inputs, outputs=[frame_overlap_simulator]
                    )

            # Wire up change handlers for shakify controls (including toggle)
            shakify_all_components = shakify_components + [
                components.get("show_shakify_in_overlap")
            ]
            for shakify_component in shakify_all_components:
                if shakify_component:
                    shakify_component.change(
                        fn=update_overlap_viz, inputs=viz_inputs, outputs=[frame_overlap_simulator]
                    )

        # Path Analysis & Optimization handlers
        if components.get("analyze_path_btn") and components.get("optimize_path_btn"):
            from deforum.utils.camera_path_optimizer import (
                analyze_camera_path,
                generate_optimization_report,
                auto_optimize_for_depth_warping,
            )

            def handle_analyze_path(tx, ty, tz, rx, ry, rz, width_val, height_val):
                """Analyze camera path and show preservation metrics."""
                try:
                    max_frames = 333
                    width = int(width_val) if width_val else 1920
                    height = int(height_val) if height_val else 1080

                    metrics, analysis = analyze_camera_path(
                        translation_x=tx or "0:(0)",
                        translation_y=ty or "0:(0)",
                        translation_z=tz or "0:(0)",
                        rotation_3d_x=rx or "0:(0)",
                        rotation_3d_y=ry or "0:(0)",
                        rotation_3d_z=rz or "0:(0)",
                        max_frames=max_frames,
                        width=width,
                        height=height,
                    )

                    report = generate_optimization_report(analysis)
                    return report
                except Exception as e:
                    return f"❌ Analysis failed: {str(e)}"

            def handle_optimize_path(tx, ty, tz, rx, ry, rz, width_val, height_val):
                """Auto-optimize translation schedules for depth warping."""
                try:
                    max_frames = 333
                    width = int(width_val) if width_val else 1920
                    height = int(height_val) if height_val else 1080

                    optimized_tx, optimized_ty, optimized_tz, status = (
                        auto_optimize_for_depth_warping(
                            translation_x=tx or "0:(0)",
                            translation_y=ty or "0:(0)",
                            translation_z=tz or "0:(0)",
                            rotation_3d_x=rx or "0:(0)",
                            rotation_3d_y=ry or "0:(0)",
                            rotation_3d_z=rz or "0:(0)",
                            max_frames=max_frames,
                            width=width,
                            height=height,
                        )
                    )

                    return optimized_tx, optimized_ty, optimized_tz, status
                except Exception as e:
                    return tx, ty, tz, f"❌ Optimization failed: {str(e)}"

            # Wire up analyze button
            components["analyze_path_btn"].click(
                fn=handle_analyze_path,
                inputs=[
                    components.get("translation_x"),
                    components.get("translation_y"),
                    components.get("translation_z"),
                    components.get("rotation_3d_x"),
                    components.get("rotation_3d_y"),
                    components.get("rotation_3d_z"),
                    components.get("W"),
                    components.get("H"),
                ],
                outputs=[components["path_analysis_output"]],
            )

            # Wire up optimize button (updates schedules AND shows report)
            components["optimize_path_btn"].click(
                fn=handle_optimize_path,
                inputs=[
                    components.get("translation_x"),
                    components.get("translation_y"),
                    components.get("translation_z"),
                    components.get("rotation_3d_x"),
                    components.get("rotation_3d_y"),
                    components.get("rotation_3d_z"),
                    components.get("W"),
                    components.get("H"),
                ],
                outputs=[
                    components.get("translation_x"),
                    components.get("translation_y"),
                    components.get("translation_z"),
                    components["path_analysis_output"],
                ],
            )

        # Live preview polling - updates every 500ms
        # Smart polling: only shows fresh previews (< 5 sec old), silently handles errors
        live_preview_timer = gr.Timer(value=0.5, active=True)
        live_preview_timer.tick(
            fn=get_latest_frames, inputs=[], outputs=[live_preview_image, depth_preview_image]
        )

        # Check if Flux blocker is active (minimal component set)
        is_flux_blocker_active = len(components) < 10  # Minimal set has only ~2 components

        if is_flux_blocker_active:
            logger.info("Flux blocker active - Deforum UI will show setup instructions")
            # In blocker mode, just use what components we have
            component_list = [components.get(name, dummy_component) for name in ["show_info_on_ui"]]
        else:
            # Normal mode - get all components (use .get() with dummy_component fallback for None values)
            component_names_needed = get_component_names()
            component_list = [
                components.get(name, dummy_component) or dummy_component
                for name in component_names_needed
            ]

        submit.click(
            fn=wrap_gradio_gpu_call(run_deforum),
            _js="submit_deforum",
            inputs=[dummy_component, dummy_component] + component_list,
            outputs=[
                deforum_gallery,
                components.get("resume_timestring", dummy_component),
                generation_info,
                html_info,
            ],
        )

        settings_component_list = [
            components.get(name, dummy_component) or dummy_component
            for name in get_settings_component_names()
        ]
        video_settings_component_list = [
            components.get(name, dummy_component) or dummy_component
            for name in list(DeforumOutputArgs().keys())
        ]

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

        # Video upload for metadata extraction (component from Init tab)
        if "video_upload" in components:
            video_upload = components["video_upload"]
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
                handle_generate_custom,
            )
            from deforum.utils.schedule_visualizer import visualize_schedules

            btn_generate_preset = components.get("btn_generate_preset")
            btn_randomize_preset = components.get("btn_randomize_preset")
            btn_generate_custom = components.get("btn_generate_custom")

            # Get schedule textboxes
            tx = components.get("translation_x")
            ty = components.get("translation_y")
            tz = components.get("translation_z")
            rx = components.get("rotation_3d_x")
            ry = components.get("rotation_3d_y")
            rz = components.get("rotation_3d_z")
            animation_prompts = components.get("animation_prompts")

            if btn_generate_preset and camera_path_plot and tx:
                # Wire up preset generation button
                btn_generate_preset.click(
                    fn=handle_generate_preset,
                    inputs=[
                        components.get("preset_type"),
                        components.get("speed_multiplier"),
                        components.get("speed_randomization"),
                        components.get("preset_radius"),
                        components.get("preset_height"),
                        components.get("preset_num_frames"),
                        components.get("preset_closed_loop"),
                        components.get("preset_randomize"),
                        components.get("preset_random_seed"),
                        tx,
                        ty,
                        tz,
                        rx,
                        ry,
                        rz,
                    ],
                    outputs=[
                        components.get("preset_status"),
                        tx,
                        ty,
                        tz,
                        rx,
                        ry,
                        rz,  # Only update schedules
                    ],
                )

            if btn_randomize_preset and camera_path_plot and tx:
                # Wire up randomize button
                def randomize_preset_wrapper(*args):
                    """Randomize by using current params but with random seed"""
                    args_list = list(args)
                    args_list[8] = -1  # preset_random_seed index - force new randomization
                    if args_list[7] == 0:  # preset_randomize
                        args_list[7] = 0.5
                    return handle_generate_preset(*args_list)

                btn_randomize_preset.click(
                    fn=randomize_preset_wrapper,
                    inputs=[
                        components.get("preset_type"),
                        components.get("speed_multiplier"),
                        components.get("speed_randomization"),
                        components.get("preset_radius"),
                        components.get("preset_height"),
                        components.get("preset_num_frames"),
                        components.get("preset_closed_loop"),
                        components.get("preset_randomize"),
                        components.get("preset_random_seed"),
                        tx,
                        ty,
                        tz,
                        rx,
                        ry,
                        rz,
                    ],
                    outputs=[
                        components.get("preset_status"),
                        tx,
                        ty,
                        tz,
                        rx,
                        ry,
                        rz,  # Only update schedules
                    ],
                )

            if btn_generate_custom and camera_path_plot and tx:
                # Wire up custom spline generation button
                btn_generate_custom.click(
                    fn=handle_generate_custom,
                    inputs=[
                        components.get("num_control_points"),
                        components.get("spline_type"),
                        components.get("spline_smoothness"),
                        components.get("look_at_curve"),
                        components.get("custom_num_frames"),
                        components.get("custom_closed_loop"),
                        components.get("control_point_pattern"),
                        components.get("control_pattern_scale"),
                        tx,
                        ty,
                        tz,
                        rx,
                        ry,
                        rz,
                    ],
                    outputs=[
                        components.get("custom_status"),
                        tx,
                        ty,
                        tz,
                        rx,
                        ry,
                        rz,  # Only update schedules
                    ],
                )

            # Wire speed sliders to auto-regenerate preset path on change
            speed_mult = components.get("speed_multiplier")
            speed_rand = components.get("speed_randomization")
            if speed_mult and btn_generate_preset and tx:
                speed_mult.change(
                    fn=handle_generate_preset,
                    inputs=[
                        components.get("preset_type"),
                        components.get("speed_multiplier"),
                        components.get("speed_randomization"),
                        components.get("preset_radius"),
                        components.get("preset_height"),
                        components.get("preset_num_frames"),
                        components.get("preset_closed_loop"),
                        components.get("preset_randomize"),
                        components.get("preset_random_seed"),
                        tx,
                        ty,
                        tz,
                        rx,
                        ry,
                        rz,
                    ],
                    outputs=[components.get("preset_status"), tx, ty, tz, rx, ry, rz],
                )

            if speed_rand and btn_generate_preset and tx:
                speed_rand.change(
                    fn=handle_generate_preset,
                    inputs=[
                        components.get("preset_type"),
                        components.get("speed_multiplier"),
                        components.get("speed_randomization"),
                        components.get("preset_radius"),
                        components.get("preset_height"),
                        components.get("preset_num_frames"),
                        components.get("preset_closed_loop"),
                        components.get("preset_randomize"),
                        components.get("preset_random_seed"),
                        tx,
                        ty,
                        tz,
                        rx,
                        ry,
                        rz,
                    ],
                    outputs=[components.get("preset_status"), tx, ty, tz, rx, ry, rz],
                )

            # Wire schedule textboxes to update visualization whenever they change
            def update_viz_from_schedules(
                tx_val,
                ty_val,
                tz_val,
                rx_val,
                ry_val,
                rz_val,
                prompts_val,
                shake_name_val,
                shake_intensity_val,
                shake_speed_val,
                apply_shakify_toggle,
                max_frames=333,
            ):
                """Update visualization from schedule textbox values with optional shakify."""
                # Shakify params with defaults
                shake_name = shake_name_val if shake_name_val else "None"
                shake_intensity = float(shake_intensity_val) if shake_intensity_val else 1.0
                shake_speed = float(shake_speed_val) if shake_speed_val else 1.0

                fig, stats = visualize_schedules(
                    tx_val,
                    ty_val,
                    tz_val,
                    rx_val,
                    ry_val,
                    rz_val,
                    max_frames,
                    prompts_val,
                    shake_name=shake_name,
                    shake_intensity=shake_intensity,
                    shake_speed=shake_speed,
                    target_fps=60,
                    apply_shakify=apply_shakify_toggle,
                )
                return fig

            # Each schedule textbox triggers visualization update
            if tx and camera_path_plot:
                schedule_inputs = [tx, ty, tz, rx, ry, rz]
                if animation_prompts:
                    schedule_inputs.append(animation_prompts)
                # Add shakify params and toggle
                schedule_inputs.extend(
                    [
                        components.get("shake_name"),
                        components.get("shake_intensity"),
                        components.get("shake_speed"),
                        components.get("show_shakify_in_camera_path"),
                    ]
                )

                for schedule_box in [tx, ty, tz, rx, ry, rz]:
                    if schedule_box:
                        schedule_box.change(
                            fn=update_viz_from_schedules,
                            inputs=schedule_inputs,
                            outputs=[camera_path_plot],
                        )

                # Also trigger on prompt changes (for keyframe markers)
                if animation_prompts:
                    animation_prompts.change(
                        fn=update_viz_from_schedules,
                        inputs=schedule_inputs,
                        outputs=[camera_path_plot],
                    )

                # Trigger on shakify changes
                for shakify_comp in [
                    components.get("shake_name"),
                    components.get("shake_intensity"),
                    components.get("shake_speed"),
                    components.get("show_shakify_in_camera_path"),
                ]:
                    if shakify_comp:
                        shakify_comp.change(
                            fn=update_viz_from_schedules,
                            inputs=schedule_inputs,
                            outputs=[camera_path_plot],
                        )

                # Trigger visualization on settings load
                load_settings_btn.click(
                    fn=update_viz_from_schedules, inputs=schedule_inputs, outputs=[camera_path_plot]
                )

                # Also trigger on video upload (metadata extraction)
                if "video_upload" in components:
                    components["video_upload"].upload(
                        fn=update_viz_from_schedules,
                        inputs=schedule_inputs,
                        outputs=[camera_path_plot],
                    )

        except Exception as e:
            logger.error(f"Failed to wire Camera Path buttons to right panel: {e}")
            import traceback

            traceback.print_exc()

        # Depth preview visibility toggle based on animation_mode
        def update_depth_preview_visibility(save_depth, anim_mode):
            # Show depth preview in 3D mode (depth maps are always generated for warping)
            should_show = anim_mode == "3D"
            return gr.update(visible=should_show)

        # Only bind events if components exist (skip in blocker mode)
        if "save_depth_maps" in components and "animation_mode" in components:
            components["save_depth_maps"].change(
                fn=update_depth_preview_visibility,
                inputs=[components["save_depth_maps"], components["animation_mode"]],
                outputs=[depth_preview_image],
            )

            components["animation_mode"].change(
                fn=update_depth_preview_visibility,
                inputs=[components["save_depth_maps"], components["animation_mode"]],
                outputs=[depth_preview_image],
            )

            # Also update visibility when settings are loaded
            load_settings_btn.click(
                fn=update_depth_preview_visibility,
                inputs=[components["save_depth_maps"], components["animation_mode"]],
                outputs=[depth_preview_image],
            )

            # And when video is uploaded (metadata extraction)
            if "video_upload" in components:
                components["video_upload"].upload(
                    fn=update_depth_preview_visibility,
                    inputs=[components["save_depth_maps"], components["animation_mode"]],
                    outputs=[depth_preview_image],
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
            logger.info(
                f"No settings found in webui root, using default settings from: {settings_file_path}"
            )

        # Update the settings path field with the path
        settings_path.value = settings_file_path

        # Now call load_all_settings with ui_launch=True to update all components
        wrapped_fn = wrap_gradio_call(
            lambda *args, **kwargs: load_all_settings(*args, ui_launch=True, **kwargs)
        )
        inputs = [settings_file_path] + [component.value for component in settings_component_list]
        outputs = settings_component_list
        updated_values = wrapped_fn(*inputs, *outputs)[0]

        # Update all the component values
        settings_component_name_to_obj = {
            name: component
            for name, component in zip(get_settings_component_names(), settings_component_list)
        }
        for key, value in updated_values.items():
            if key in settings_component_name_to_obj:
                settings_component_name_to_obj[key].value = value["value"]

        # Update depth preview visibility based on loaded settings (skip in blocker mode)
        if "animation_mode" in components:
            anim_mode = components["animation_mode"].value
            should_show = anim_mode == "3D"
            depth_preview_image.visible = should_show
            logger.info(f"Depth preview gallery: visible={should_show} (anim_mode={anim_mode})")

    # Always load settings on startup - either from persistent settings path (if enabled),
    # from webui root, or from the extension's default settings
    trigger_load_general_settings()

    return [(deforum_interface, "Deforum", "deforum_interface")]
