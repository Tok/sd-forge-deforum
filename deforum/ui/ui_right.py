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
from deforum.ui.handlers.model_preset_handler import (
    handle_apply_model_defaults,
    get_preset_for_current_model,
    create_model_info_html
)

# Initialize logger
logger = get_logger()


# State tracking to prevent UI flashing when no previews available
_last_preview_state = {"frame": None, "depth": None, "timer": None}


def get_latest_frames():
    """Poll for latest frame and depth map preview files during generation.

    Uses state tracking to prevent UI updates when previews haven't changed.
    This eliminates blinking/flashing when backend disconnects or no generation active.

    Returns gr.skip() for all three outputs if state unchanged to avoid UI updates.
    """
    import glob
    from pathlib import Path
    import time
    import gradio as gr

    try:
        from modules import shared
        # Respect Forge's video output directory setting, fallback to output/deforum
        deforum_outdir = shared.opts.outdir_videos or os.path.join(os.getcwd(), "output", "deforum")

        # Find most recent directory (Deforum_TIMESTAMP pattern)
        subdirs = [
            d for d in glob.glob(os.path.join(deforum_outdir, "Deforum_*")) if os.path.isdir(d)
        ]
        if not subdirs:
            # No output directories - check if state unchanged
            if (_last_preview_state["frame"] is None and
                _last_preview_state["depth"] is None and
                _last_preview_state["timer"] is None):
                return gr.skip(), gr.skip(), gr.skip()  # Skip update if already showing nothing
            _last_preview_state["frame"] = None
            _last_preview_state["depth"] = None
            _last_preview_state["timer"] = None
            return None, None, None

        latest_dir = max(subdirs, key=os.path.getmtime)

        # Look for fixed preview filenames
        frame_preview = os.path.join(latest_dir, "frame-preview.png")
        depth_preview = os.path.join(latest_dir, "depth-preview.png")
        timing_file = os.path.join(latest_dir, ".generation-timing")

        # Check if preview files are fresh (modified within last 5 seconds)
        # This prevents showing stale previews and stops polling when generation ends
        current_time = time.time()
        frame_is_fresh = (
            os.path.exists(frame_preview) and (current_time - os.path.getmtime(frame_preview)) < 5.0
        )
        depth_is_fresh = (
            os.path.exists(depth_preview) and (current_time - os.path.getmtime(depth_preview)) < 5.0
        )

        # Read timing info if available
        timer_text = None
        if os.path.exists(timing_file):
            try:
                with open(timing_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 1:
                        start_time = float(lines[0].strip())
                        is_resume = len(lines) >= 2 and lines[1].strip() == "RESUME"

                        if not is_resume:
                            elapsed = current_time - start_time
                            hours = int(elapsed // 3600)
                            minutes = int((elapsed % 3600) // 60)
                            seconds = int(elapsed % 60)
                            if hours > 0:
                                timer_text = f"⏱️ {hours}h {minutes}m {seconds}s"
                            elif minutes > 0:
                                timer_text = f"⏱️ {minutes}m {seconds}s"
                            else:
                                timer_text = f"⏱️ {seconds}s"
                        else:
                            # On resume, don't show misleading timing stats
                            timer_text = None
            except (ValueError, IndexError, IOError):
                timer_text = None

        # Determine new preview paths
        latest_frame = frame_preview if frame_is_fresh else None
        latest_depth = depth_preview if depth_is_fresh else None

        # Check if state changed - skip update if unchanged to prevent flashing
        if (
            latest_frame == _last_preview_state["frame"]
            and latest_depth == _last_preview_state["depth"]
            and timer_text == _last_preview_state["timer"]
        ):
            return gr.skip(), gr.skip(), gr.skip()

        # Update state
        _last_preview_state["frame"] = latest_frame
        _last_preview_state["depth"] = latest_depth
        _last_preview_state["timer"] = timer_text

        return latest_frame, latest_depth, timer_text

    except Exception:
        # Silently handle errors (e.g., backend disconnected, filesystem issues)
        # Only update UI if state actually changes
        if (_last_preview_state["frame"] is None and
            _last_preview_state["depth"] is None and
            _last_preview_state["timer"] is None):
            return gr.skip(), gr.skip(), gr.skip()  # Skip update if already showing nothing
        _last_preview_state["frame"] = None
        _last_preview_state["depth"] = None
        _last_preview_state["timer"] = None
        return None, None, None


def on_ui_tabs():
    # extend paths using sys.path.extend so we can access all of our files and folders
    deforum_sys_extend()

    # DEBUG: Verify we're loading the correct version (Git commit fc2071bb - CSS f-string fix)
    print("[DEFORUM DEBUG] ui_right.py loaded - CSS f-string fix applied (commit fc2071bb)")

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
    # Add timestamp to force CSS cache bust on every restart
    import time
    cache_bust = int(time.time())

    # Use % formatting instead of f-string to avoid Python 3.12 parsing issues
    css_header = """
    /* ========================================================================== */
    /* DEFORUM SLOPCORE CSS v2.0 - Loaded at: %s */
    /* Gradient Direction Fix: BB0=Vertical(180deg) DA3=Horizontal(135deg) */
    /* If buttons still wrong: Clear ALL browser cache + hard refresh (Ctrl+Shift+F5) */
    /* ========================================================================== */

    /* Visual confirmation that CSS loaded - adds subtle indicator */
    #deforum_interface {
        --css-version: "%s";
    }
    """ % (cache_bust, cache_bust)

    css_body = """
    /* BB0 Slopcore gradient for OTHER buttons (audio sync, etc.) - TOP TO BOTTOM (VERTICAL) */
    /* Electric purple at top (#5606ff) → Azure cyan at bottom (#17a7fe) - matching BB0 album */
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
        background: linear-gradient(180deg, #5606ff 0%, #17a7fe 100%) !important;
        background-image: linear-gradient(180deg, #5606ff 0%, #17a7fe 100%) !important;
        background-color: #5606ff !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
        box-shadow: 0 4px 6px rgba(86, 6, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    /* BB0 Hover states for OTHER buttons - REVERSED TOP TO BOTTOM */
    /* Azure cyan at top → Electric purple at bottom (reversed from normal) */
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
        background: linear-gradient(180deg, #17a7fe 0%, #5606ff 100%) !important;
        background-image: linear-gradient(180deg, #17a7fe 0%, #5606ff 100%) !important;
        background-color: #17a7fe !important;
        box-shadow: 0 6px 12px rgba(86, 6, 255, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* DA3 slopcore gradient for DA3/3DGS-related buttons (cyan → red/pink) */
    .da3-button,
    .da3-button *,
    .da3-button button,
    div.da3-button button,
    button.da3-button,
    *[class*="da3-button"],
    *[class*="da3-button"] button {
        background: linear-gradient(135deg, #1cc4e6 0%, #f64a5e 100%) !important;
        background-image: linear-gradient(135deg, #1cc4e6 0%, #f64a5e 100%) !important;
        background-color: #1cc4e6 !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
        box-shadow: 0 4px 6px rgba(28, 196, 230, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    /* DA3 button hover - reversed gradient */
    .da3-button:hover,
    .da3-button *:hover,
    .da3-button button:hover,
    div.da3-button button:hover,
    button.da3-button:hover,
    *[class*="da3-button"]:hover,
    *[class*="da3-button"] button:hover {
        background: linear-gradient(135deg, #f64a5e 0%, #1cc4e6 100%) !important;
        background-image: linear-gradient(135deg, #f64a5e 0%, #1cc4e6 100%) !important;
        background-color: #f64a5e !important;
        box-shadow: 0 6px 12px rgba(246, 74, 94, 0.4) !important;
        transform: translateY(-1px) !important;
    }

    /* Visualization refresh buttons - smaller and rounded */
    #refresh_camera_path_btn,
    #refresh_camera_path_btn *,
    #refresh_camera_path_btn button,
    button#refresh_camera_path_btn,
    #refresh_wormtrail_btn,
    #refresh_wormtrail_btn *,
    #refresh_wormtrail_btn button,
    button#refresh_wormtrail_btn {
        min-height: 32px !important;
        max-height: 32px !important;
        height: 32px !important;
        padding: 4px 12px !important;
        font-size: 13px !important;
        border-radius: 16px !important;
        background: #4a5568 !important;
        border: 1px solid #6b7280 !important;
        color: #e5e7eb !important;
        font-weight: 500 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.2) !important;
        transition: all 0.2s ease !important;
    }

    #refresh_camera_path_btn:hover,
    #refresh_camera_path_btn *:hover,
    #refresh_camera_path_btn button:hover,
    button#refresh_camera_path_btn:hover,
    #refresh_wormtrail_btn:hover,
    #refresh_wormtrail_btn *:hover,
    #refresh_wormtrail_btn button:hover,
    button#refresh_wormtrail_btn:hover {
        background: #5a6678 !important;
        border-color: #7b8390 !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.3) !important;
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

    /* DA3 slopcore gradient for MAIN GENERATE BUTTON ONLY (electric cyan → watermelon) LEFT TO RIGHT */
    /* MUST COME LAST - Overrides BB0 button styles above */
    /* Target WebUI-generated Deforum generate button specifically */
    /* Button has elem_id="deforum_generate" and variant="primary" */
    #deforum_generate_box button[id$="_generate"],
    #deforum_generate_box button.primary,
    button#deforum_generate,
    button[id="deforum_generate"],
    #deforum_generate,
    #deforum_generate_box > div > button,
    #deforum_results #deforum_generate,
    div#deforum_generate_box button:not(.generate-box-interrupt):not(.generate-box-skip):not(.generate-box-interrupting) {
        background: linear-gradient(135deg, #1cc4e6 0%, #f64a5e 100%) !important;
        background-image: linear-gradient(135deg, #1cc4e6 0%, #f64a5e 100%) !important;
        background-color: #1cc4e6 !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.2) !important;
        box-shadow: 0 4px 6px rgba(28, 196, 230, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    /* Main generate button hover - reversed DA3 gradient (LEFT TO RIGHT) */
    #deforum_generate_box button[id$="_generate"]:hover,
    #deforum_generate_box button.primary:hover,
    button#deforum_generate:hover,
    button[id="deforum_generate"]:hover,
    #deforum_generate:hover,
    #deforum_generate_box > div > button:hover,
    #deforum_results #deforum_generate:hover,
    div#deforum_generate_box button:not(.generate-box-interrupt):not(.generate-box-skip):not(.generate-box-interrupting):hover {
        background: linear-gradient(135deg, #f64a5e 0%, #1cc4e6 100%) !important;
        background-image: linear-gradient(135deg, #f64a5e 0%, #1cc4e6 100%) !important;
        background-color: #f64a5e !important;
        box-shadow: 0 6px 12px rgba(246, 74, 94, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    """

    # Combine CSS header with body (header already formatted above)
    slopcore_css = css_header + css_body

    # JavaScript to force-apply gradient styles after page load
    # This ensures styles override Gradio's built-in variant="primary" styles
    js_force_gradients = """
    <script>
    (function() {
        // Apply slopcore gradients after DOM is ready
        function applySlopcoreGradients() {
            // BB0 gradient (vertical top-to-bottom: purple → cyan) for ALL buttons
            const bb0Buttons = document.querySelectorAll('.slopcore-button');
            bb0Buttons.forEach(btn => {
                btn.style.background = 'linear-gradient(180deg, #5606ff 0%, #17a7fe 100%)';
                btn.style.backgroundImage = 'linear-gradient(180deg, #5606ff 0%, #17a7fe 100%)';
            });

            // DA3 gradient (horizontal left-to-right: cyan → watermelon) for generate button ONLY
            const generateBtn = document.querySelector('#deforum_generate');
            if (generateBtn) {
                generateBtn.style.background = 'linear-gradient(135deg, #1cc4e6 0%, #f64a5e 100%)';
                generateBtn.style.backgroundImage = 'linear-gradient(135deg, #1cc4e6 0%, #f64a5e 100%)';
                generateBtn.style.backgroundColor = '#1cc4e6';
                generateBtn.style.color = 'white';
                generateBtn.style.fontWeight = '600';
            }
        }

        // Run after page load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', applySlopcoreGradients);
        } else {
            applySlopcoreGradients();
        }

        // Also run after a short delay to catch Gradio's dynamic updates
        setTimeout(applySlopcoreGradients, 500);
        setTimeout(applySlopcoreGradients, 2000);
    })();
    </script>
    """

    with gr.Blocks(analytics_enabled=False, css=slopcore_css, head=js_force_gradients) as deforum_interface:
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

                # Use Forge's video output directory setting, fallback to output/deforum
                from modules import shared
                deforum_outdir = shared.opts.outdir_videos or os.path.join(os.getcwd(), "output", "deforum")
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

                # Generation timer - shows elapsed time during generation
                generation_timer = gr.Textbox(
                    label="",
                    value="",
                    interactive=False,
                    show_label=False,
                    elem_id="deforum_generation_timer",
                    container=False,
                    visible=True,
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
                    components["settings_path"] = settings_path
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
                    refresh_emoji = emoji.refresh_icon()
                    if refresh_emoji:
                        refresh_emoji += " "
                    refresh_camera_path_btn = gr.Button(
                        value=f"{refresh_emoji}Refresh Camera Path" if refresh_emoji else "Refresh Camera Path",
                        variant="secondary",
                        size="sm",
                        scale=0,
                        elem_id="refresh_camera_path_btn",
                    )
                # Get visualization visibility default from settings (persistent)
                from modules import shared
                show_viz_default = getattr(shared.opts, 'deforum_show_visualizations', False)

                with gr.Row(variant="compact"):
                    camera_path_plot = gr.Plot(
                        label="Camera Path Visualization (Real-time)",
                        show_label=False,
                        elem_id="deforum_camera_path_viz",
                        visible=show_viz_default,  # Uses persistent setting from Settings > Deforum
                    )

                # Frame Overlap Simulator (shows preservation/novelty metrics)
                with gr.Row(variant="compact"):
                    frame_emoji = emoji.purple_square()
                    if frame_emoji:
                        frame_emoji += " "
                    gr.Markdown(f"### {frame_emoji}Frame Overlap Simulator (Worm Trail)")
                    show_visualizations = gr.Checkbox(
                        value=show_viz_default,  # Uses persistent setting from Settings > Deforum
                        label="Show Visualizations",
                        info="Toggle camera path and wormtrail preview visibility",
                        scale=0,
                    )
                    show_shakify_in_overlap = gr.Checkbox(
                        value=False,
                        label="Show Shakify Preview",
                        info="Add subtle camera shake overlay to visualization",
                        scale=0,
                    )
                with gr.Row(variant="compact"):
                    wormtrail_quality_full = gr.Checkbox(
                        value=False,
                        label="Full Quality",
                        info="Use full schedules (slow for 1000+ frames, shows complete movement)",
                        scale=0,
                    )
                    refresh_emoji = emoji.refresh_icon()
                    if refresh_emoji:
                        refresh_emoji += " "
                    refresh_wormtrail_btn = gr.Button(
                        value=f"{refresh_emoji}Refresh Wormtrail" if refresh_emoji else "Refresh Wormtrail",
                        variant="secondary",
                        size="sm",
                        scale=0,
                        elem_id="refresh_wormtrail_btn",
                    )
                with gr.Row(variant="compact"):
                    frame_overlap_simulator = gr.HTML(
                        value='<div style="padding: 20px; color: #C8C8DC;">Loading frame overlap simulator...</div>',
                        label="Frame Overlap Simulator",
                        show_label=False,
                        elem_id="deforum_frame_overlap_sim",
                        visible=show_viz_default,  # Uses persistent setting from Settings > Deforum
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
                        elem_classes=["slopcore-button"]
                    )
                with gr.Row(variant="compact"):
                    path_analysis_output = gr.Markdown(
                        value="",
                        label="Analysis Results",
                        elem_id="deforum_path_analysis_output",
                        visible=True,
                    )

                components["camera_path_plot"] = camera_path_plot
                components["refresh_camera_path_btn"] = refresh_camera_path_btn
                components["frame_overlap_simulator"] = frame_overlap_simulator
                components["show_visualizations"] = show_visualizations
                components["show_shakify_in_camera_path"] = show_shakify_in_camera_path
                components["show_shakify_in_overlap"] = show_shakify_in_overlap
                components["wormtrail_quality_full"] = wormtrail_quality_full
                components["refresh_wormtrail_btn"] = refresh_wormtrail_btn
                components["analyze_path_btn"] = analyze_path_btn
                components["optimize_path_btn"] = optimize_path_btn
                components["path_analysis_output"] = path_analysis_output

                # Wire up visualization visibility toggle
                show_visualizations.change(
                    fn=lambda visible: (gr.update(visible=visible), gr.update(visible=visible)),
                    inputs=[show_visualizations],
                    outputs=[camera_path_plot, frame_overlap_simulator],
                )

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
                max_frames_val,
            ):
                """Update camera path visualization with optional shakify overlay."""
                try:
                    # Shakify params with defaults
                    shake_name = shake_name_val if shake_name_val else "None"
                    shake_intensity = float(shake_intensity_val) if shake_intensity_val else 1.0
                    shake_speed = float(shake_speed_val) if shake_speed_val else 1.0

                    # Get max_frames from component or use sensible default (100 frames = ~1.67 sec at 60fps)
                    max_frames = int(max_frames_val) if max_frames_val else 100

                    fig, _ = visualize_schedules(
                        tx or "",
                        ty or "",
                        tz or "",
                        rx or "",
                        ry or "",
                        rz or "",
                        max_frames,
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

            # Wire up manual refresh button for camera path
            # This prevents automatic updates that could be expensive for large animations
            camera_path_inputs = [
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
                components.get("max_frames"),
            ]

            if components.get("refresh_camera_path_btn"):
                components["refresh_camera_path_btn"].click(
                    fn=update_camera_path_visualization,
                    inputs=camera_path_inputs,
                    outputs=[camera_path_plot],
                )

            # Load visualization on UI startup with default preset
            deforum_interface.load(
                fn=update_camera_path_visualization,
                inputs=camera_path_inputs,
                outputs=[camera_path_plot],
            )

            # NOTE: Schedule textbox .change() handlers removed to prevent automatic updates
            # Preset buttons NO LONGER auto-update visualizations - user must click refresh buttons

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
                zoom,
                width_val,
                height_val,
                shake_name_val,
                shake_intensity_val,
                shake_speed_val,
                apply_shakify_toggle,
                prompts,
                max_frames_val,
                use_full_quality=False,
            ):
                """Update frame overlap visualization with optional shakify overlay and zoom."""
                # Guard against empty inputs during UI initialization
                if tx is None and ty is None and tz is None:
                    return '<div style="padding: 20px; color: #C8C8DC;">Loading frame overlap simulator...</div>'

                # Handle None zoom (might be None during initialization)
                if zoom is None:
                    zoom = ""

                # Handle None prompts
                if prompts is None:
                    prompts = ""

                # Get max_frames from component or use sensible default
                max_frames = int(max_frames_val) if max_frames_val else 100
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
                    zoom=zoom or "",
                    max_frames=max_frames,
                    width=width,
                    height=height,
                    shake_name=shake_name,
                    shake_intensity=shake_intensity,
                    shake_speed=shake_speed,
                    target_fps=60,
                    animation_prompts=prompts or "",
                    use_full_schedules=use_full_quality,
                )

            def init_overlap_viz_with_preset():
                """Initialize frame overlap visualization with default 'rotate-around' preset."""
                try:
                    # Sensible default: 100 frames (~1.67 sec at 60fps)
                    default_frames = 100

                    # Generate default rotate-around path with smaller radius to reduce rotation
                    _, schedules, _ = generate_preset_path(
                        preset_type="rotate-around",
                        radius=30.0,  # Reduced from 100 to minimize rotation
                        height=0.0,
                        num_frames=default_frames,
                        closed_loop=True,
                        speed_multiplier=0.5,  # Slower movement
                        speed_randomization=0.0,
                    )

                    # Generate visualization with preset schedules (NO shakify on init)
                    result = update_frame_overlap_visualization(
                        translation_x=schedules.get("translation_x", "0:(0)"),
                        translation_y=schedules.get("translation_y", "0:(0)"),
                        translation_z=schedules.get("translation_z", "0:(0)"),
                        rotation_3d_x=schedules.get("rotation_3d_x", "0:(0)"),
                        rotation_3d_y=schedules.get("rotation_3d_y", "0:(0)"),
                        rotation_3d_z=schedules.get("rotation_3d_z", "0:(0)"),
                        zoom="",  # No zoom on init
                        max_frames=default_frames,
                        width=1920,
                        height=1080,
                        shake_name="None",  # Explicitly disable shakify on init
                        shake_intensity=1.0,
                        shake_speed=1.0,
                        target_fps=60,
                    )
                    return result

                except Exception as e:
                    import traceback
                    error_msg = f"ERROR in init_overlap_viz_with_preset: {e}\n{traceback.format_exc()}"
                    print(error_msg)
                    return f'<div style="color: #FF5050; padding: 20px;">Init Error: {e}</div>'

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
                components.get("zoom"),  # Zoom schedule for wormtrail
                components.get("W"),
                components.get("H"),
                components.get("shake_name"),
                components.get("shake_intensity"),
                components.get("shake_speed"),
                components.get("show_shakify_in_overlap"),  # Toggle control
                components.get("animation_prompts"),  # For keyframe detection
                components.get("max_frames"),  # Actual frame count
                components.get("wormtrail_quality_full"),  # Full quality toggle
            ]

            # Wire up manual refresh button for wormtrail
            # This prevents automatic updates that freeze UI for large animations
            if components.get("refresh_wormtrail_btn"):
                components["refresh_wormtrail_btn"].click(
                    fn=update_overlap_viz,
                    inputs=viz_inputs,
                    outputs=[frame_overlap_simulator],
                )

            # Note: .change() handlers removed to prevent duplicate updates
            # The preset button already returns frame_overlap_simulator as output
            # Manual changes to schedules don't need auto-update (user can click refresh button)

        # Path Analysis & Optimization handlers
        if components.get("analyze_path_btn") and components.get("optimize_path_btn"):
            from deforum.utils.camera_path_optimizer import (
                analyze_camera_path,
                generate_optimization_report,
                auto_optimize_for_depth_warping,
            )

            def handle_analyze_path(tx, ty, tz, rx, ry, rz, width_val, height_val, max_frames_val):
                """Analyze camera path and show preservation metrics."""
                try:
                    max_frames = int(max_frames_val) if max_frames_val else 100
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

            def handle_optimize_path(tx, ty, tz, rx, ry, rz, width_val, height_val, max_frames_val):
                """Auto-optimize translation schedules for depth warping."""
                try:
                    max_frames = int(max_frames_val) if max_frames_val else 100
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
                    components.get("max_frames"),
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
                    components.get("max_frames"),
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
            fn=get_latest_frames, inputs=[], outputs=[live_preview_image, depth_preview_image, generation_timer]
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

        # Model Preset Button Handler
        if 'apply_model_defaults_btn' in components:
            def apply_preset_wrapper(render_mode):
                """Wrapper to apply model defaults and return component updates."""
                from deforum.config.model_presets import get_preset_for_model, detect_loaded_model, detect_model_type
                from deforum.utils.system.logging import get_logger

                logger = get_logger()
                logger.info("Apply Model Defaults clicked", emoji='target')

                # Get current model
                model_name = detect_loaded_model()
                logger.info(f"Detected model file: {model_name}")

                if not model_name:
                    cross = emoji_if_enabled("❌")
                    logger.error("No model loaded - cannot apply presets")
                    return [f"{cross} No model loaded"] + [gr.skip()] * 6

                # Detect model type
                model_type = detect_model_type(model_name)
                logger.info(f"Detected model type: {model_type.value}")

                preset = get_preset_for_model(model_name)
                if not preset:
                    cross = emoji_if_enabled("❌")
                    logger.error(f"No preset found for model type: {model_type.value}")
                    return [f"{cross} No preset for: {model_type.value}"] + [gr.skip()] * 6

                logger.info(f"Found preset for {preset.model_type.value}: steps={preset.steps}, scheduler={preset.scheduler}, cfg={preset.cfg_scale}")

                # Get settings
                status_msg, settings = handle_apply_model_defaults(render_mode)
                logger.info(f"Settings dict to apply: {settings}")

                # Log each setting that will be applied
                for key, value in settings.items():
                    logger.info(f"  Setting {key} = {value}")

                # Build detailed status
                changed = []
                if 'steps' in settings:
                    changed.append(f"Steps={settings['steps']}")
                if 'scheduler' in settings:
                    changed.append(f"Scheduler={settings['scheduler']}")
                if 'scale' in settings:
                    changed.append(f"CFG={settings['scale']}")
                if 'W' in settings and 'H' in settings:
                    changed.append(f"Resolution={settings['W']}x{settings['H']}")

                check = emoji_if_enabled("✅")
                status_msg = f"{check} {preset.model_type.value.upper()}: {', '.join(changed)}"
                logger.info(f"Status message: {status_msg}")

                # Create update dict for all affected components
                updates = []
                for comp_name in ['steps', 'sampler', 'scheduler', 'W', 'H', 'scale']:
                    if comp_name in settings:
                        updates.append(settings[comp_name])
                        logger.debug(f"Updating component '{comp_name}' to: {settings[comp_name]}")
                    else:
                        updates.append(gr.skip())
                        logger.debug(f"Skipping component '{comp_name}' (not in settings)")

                logger.info("Model preset applied successfully")
                return [status_msg] + updates

            components['apply_model_defaults_btn'].click(
                fn=apply_preset_wrapper,
                inputs=[components.get('render_mode', dummy_component)],
                outputs=[
                    components.get('model_preset_status', dummy_component),
                    components.get('steps', dummy_component),
                    components.get('sampler', dummy_component),
                    components.get('scheduler', dummy_component),
                    components.get('W', dummy_component),
                    components.get('H', dummy_component),
                    components.get('scale', dummy_component),
                ]
            )

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
                def handle_preset_with_overlap(*args):
                    """Handle preset generation and update both camera path plot and wormtrail."""
                    # First 20 args are for handle_generate_preset
                    preset_args = args[:20]

                    # Generate preset schedules (FULL, not truncated) and visualization
                    result = handle_generate_preset(*preset_args)

                    # Extract schedule values from result (these are already downsampled for large animations)
                    status, tx_val, ty_val, tz_val, rx_val, ry_val, rz_val, plot = result

                    # Also update wormtrail visualization
                    wormtrail_html = None
                    if frame_overlap_simulator:
                        try:
                            wormtrail_html = update_overlap_viz(
                                tx_val, ty_val, tz_val, rx_val, ry_val, rz_val,
                                args[20],  # zoom
                                args[21],  # width
                                args[22],  # height
                                args[23],  # shake_name
                                args[24],  # shake_intensity
                                args[25],  # shake_speed
                                args[26],  # apply_shakify_toggle
                                args[27],  # prompts
                                args[28],  # max_frames
                                args[29],  # use_full_quality
                            )
                        except Exception as e:
                            import traceback
                            print(f"Warning: Wormtrail update failed: {e}\n{traceback.format_exc()}")
                            wormtrail_html = f'<div style="color: #FF5050; padding: 20px;">Wormtrail update failed: {e}</div>'

                    # Return schedules, camera path plot, and wormtrail
                    return (status, tx_val, ty_val, tz_val, rx_val, ry_val, rz_val, plot, wormtrail_html)

                btn_generate_preset.click(
                    fn=handle_preset_with_overlap,
                    inputs=[
                        # Preset generation inputs (20 args)
                        components.get("preset_type"),
                        components.get("speed_multiplier"),
                        components.get("speed_randomization"),
                        components.get("preset_radius"),
                        components.get("preset_height"),
                        components.get("max_frames"),  # Use main max_frames from Run tab, not preset_num_frames
                        components.get("preset_closed_loop"),
                        components.get("preset_randomize"),
                        components.get("preset_random_seed"),
                        components.get("preset_rotation_mode"),
                        components.get("preset_rotation_factor"),
                        components.get("preset_look_at_mode"),
                        components.get("preset_look_at_blend"),
                        tx,
                        ty,
                        tz,
                        rx,
                        ry,
                        rz,
                        components.get("animation_prompts"),
                        # Additional inputs for wormtrail update (10 args)
                        components.get("zoom"),
                        components.get("W"),
                        components.get("H"),
                        components.get("shake_name"),
                        components.get("shake_intensity"),
                        components.get("shake_speed"),
                        components.get("show_shakify_in_overlap"),
                        components.get("animation_prompts"),
                        components.get("max_frames"),
                        components.get("wormtrail_quality_full"),
                    ],
                    outputs=[
                        components.get("preset_status"),
                        tx,
                        ty,
                        tz,
                        rx,
                        ry,
                        rz,
                        camera_path_plot,  # Camera path plot
                        frame_overlap_simulator,  # Wormtrail auto-update
                    ],
                )

            if btn_randomize_preset and camera_path_plot and tx:
                # Wire up randomize button
                def randomize_preset_wrapper(*args):
                    """Randomize by using current params but with random seed and update visualizations."""
                    # First 20 args are preset params (like handle_preset_with_overlap)
                    args_list = list(args[:20])
                    args_list[8] = -1  # preset_random_seed index - force new randomization
                    if args_list[7] == 0:  # preset_randomize
                        args_list[7] = 0.5

                    # Generate preset schedules and visualization
                    result = handle_generate_preset(*args_list)
                    status, tx_val, ty_val, tz_val, rx_val, ry_val, rz_val, plot = result

                    # Also update wormtrail visualization
                    wormtrail_html = None
                    if frame_overlap_simulator:
                        try:
                            wormtrail_html = update_overlap_viz(
                                tx_val, ty_val, tz_val, rx_val, ry_val, rz_val,
                                args[20],  # zoom
                                args[21],  # width
                                args[22],  # height
                                args[23],  # shake_name
                                args[24],  # shake_intensity
                                args[25],  # shake_speed
                                args[26],  # apply_shakify_toggle
                                args[27],  # prompts
                                args[28],  # max_frames
                                args[29],  # use_full_quality
                            )
                        except Exception as e:
                            import traceback
                            print(f"Warning: Wormtrail update failed: {e}\n{traceback.format_exc()}")
                            wormtrail_html = f'<div style="color: #FF5050; padding: 20px;">Wormtrail update failed: {e}</div>'

                    return (status, tx_val, ty_val, tz_val, rx_val, ry_val, rz_val, plot, wormtrail_html)

                btn_randomize_preset.click(
                    fn=randomize_preset_wrapper,
                    inputs=[
                        # Preset generation inputs (20 args)
                        components.get("preset_type"),
                        components.get("speed_multiplier"),
                        components.get("speed_randomization"),
                        components.get("preset_radius"),
                        components.get("preset_height"),
                        components.get("max_frames"),  # Use main max_frames from Run tab
                        components.get("preset_closed_loop"),
                        components.get("preset_randomize"),
                        components.get("preset_random_seed"),
                        components.get("preset_rotation_mode"),
                        components.get("preset_rotation_factor"),
                        components.get("preset_look_at_mode"),
                        components.get("preset_look_at_blend"),
                        tx,
                        ty,
                        tz,
                        rx,
                        ry,
                        rz,
                        components.get("animation_prompts"),
                        # Additional inputs for wormtrail update (10 args)
                        components.get("zoom"),
                        components.get("W"),
                        components.get("H"),
                        components.get("shake_name"),
                        components.get("shake_intensity"),
                        components.get("shake_speed"),
                        components.get("show_shakify_in_overlap"),
                        components.get("animation_prompts"),
                        components.get("max_frames"),
                        components.get("wormtrail_quality_full"),
                    ],
                    outputs=[
                        components.get("preset_status"),
                        tx,
                        ty,
                        tz,
                        rx,
                        ry,
                        rz,
                        camera_path_plot,  # Camera path plot
                        frame_overlap_simulator,  # Wormtrail auto-update
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
                        components.get("max_frames"),  # Use main max_frames from Run tab
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

            # Wire ALL preset sliders to update both camera path and wormtrail
            # This prevents "No camera movement detected" and avoids multiple reloads
            preset_slider_components = [
                components.get("preset_type"),
                components.get("speed_multiplier"),
                components.get("speed_randomization"),
                components.get("preset_radius"),
                components.get("preset_height"),
                components.get("max_frames"),  # Use main max_frames from Run tab
                components.get("preset_closed_loop"),
                components.get("preset_rotation_mode"),
                components.get("preset_rotation_factor"),
                components.get("preset_look_at_mode"),
                components.get("preset_look_at_blend"),
            ]

            preset_inputs = [
                # Preset generation inputs (20 args)
                components.get("preset_type"),
                components.get("speed_multiplier"),
                components.get("speed_randomization"),
                components.get("preset_radius"),
                components.get("preset_height"),
                components.get("max_frames"),  # Use main max_frames from Run tab
                components.get("preset_closed_loop"),
                components.get("preset_randomize"),
                components.get("preset_random_seed"),
                components.get("preset_rotation_mode"),
                components.get("preset_rotation_factor"),
                components.get("preset_look_at_mode"),
                components.get("preset_look_at_blend"),
                tx,
                ty,
                tz,
                rx,
                ry,
                rz,
                components.get("animation_prompts"),
                # Overlap viz inputs (7 args)
                components.get("zoom"),
                components.get("W"),
                components.get("H"),
                components.get("shake_name"),
                components.get("shake_intensity"),
                components.get("shake_speed"),
                components.get("show_shakify_in_overlap"),
            ]

            preset_outputs = [
                components.get("preset_status"),
                tx,
                ty,
                tz,
                rx,
                ry,
                rz,
                camera_path_plot,
                frame_overlap_simulator,
            ]

            # Wire each preset slider to auto-regenerate on change
            if btn_generate_preset and tx and frame_overlap_simulator:
                for slider in preset_slider_components:
                    if slider:
                        slider.change(
                            fn=handle_preset_with_overlap,
                            inputs=preset_inputs,
                            outputs=preset_outputs,
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
                max_frames=100,
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

            # NOTE: Schedule textbox .change() handlers removed to prevent infinite loops
            # Visualization is generated directly by preset buttons, so real-time updates
            # on textbox changes are redundant and cause 6+ simultaneous visualization calls
            # when a preset updates all 6 textboxes at once.
            # Users can regenerate visualization by clicking preset buttons again.
            #
            # Shakify .change() handlers also removed for same reason - they trigger
            # visualization updates on every parameter change, but preset buttons already
            # handle this. Users can click preset buttons to regenerate with new settings.

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
