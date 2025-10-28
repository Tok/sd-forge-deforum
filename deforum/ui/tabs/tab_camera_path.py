"""3D Camera Path Tab

Generates smooth camera paths using splines with visualization.
Populates translation/rotation schedules for Deforum animation.
"""

import gradio as gr
from types import SimpleNamespace
from deforum.utils.system.logging import emoji as emoji_utils
from modules.ui_components import FormRow, FormColumn


def get_tab_camera_path(da: SimpleNamespace, skip_tabitem=False):
    """Create the 3D Camera Path tab.

    Args:
        da: DeforumAnimArgs namespace
        skip_tabitem: If True, don't wrap in TabItem (for embedding)

    Returns:
        Dict of Gradio components
    """
    components = {}

    tab_content = _build_camera_path_ui(da, components)

    if skip_tabitem:
        return components
    else:
        # Main tab title respects emoji setting
        tab_emoji = emoji_utils.wan_video() + " " if emoji_utils.wan_video() else ""
        with gr.TabItem(f"{tab_emoji}Camera Path", elem_id='camera_path_tab'):
            return components


def _build_camera_path_ui(da: SimpleNamespace, components: dict):
    """Build the camera path UI components."""

    gr.HTML(value="""
        <p style='margin-bottom: 1em;'>
        Generate smooth 3D camera paths using splines. Choose presets (Rotate-Around, Circle, Figure-8)
        or create custom paths. Visualize in 3D and populate translation/rotation schedules automatically.
        </p>
    """)

    with gr.Tabs():
        # ===== PRESETS TAB =====
        preset_emoji = emoji_utils.bulb() + " " if emoji_utils.bulb() else ""
        with gr.Tab(f"{preset_emoji}Presets"):
            gr.Markdown("### Quick Camera Movements")

            with FormRow(variant="compact"):
                preset_type = gr.Dropdown(
                    choices=[
                        "rotate-around",
                        "figure-eight",
                        "forward-zoom",
                        "orbit-up",
                        "spiral",
                        "street",
                        "dashcam",
                        "bodycam"
                    ],
                    value="rotate-around",
                    label="Preset Type",
                    info="Choose a camera movement pattern"
                )

            gr.Markdown("### Parameters")

            with FormRow(variant="compact"):
                preset_radius = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10,
                    label="Radius / Scale",
                    info="Size of the camera movement"
                )

            with FormRow(variant="compact"):
                preset_height = gr.Slider(
                    minimum=-200,
                    maximum=200,
                    value=0,
                    step=10,
                    label="Height (Y)",
                    info="Vertical position offset"
                )

            with FormRow(variant="compact"):
                preset_rotation_factor = gr.Slider(
                    minimum=-10,
                    maximum=10,
                    value=-5,
                    step=0.1,
                    label="Rotation Factor",
                    info="For rotate-around: rotation_y = translation_x * factor (typically -5)"
                )

            with FormRow(variant="compact"):
                preset_num_frames = gr.Number(
                    value=333,
                    label="Number of Frames",
                    info="Total frames for animation"
                )

            with FormRow(variant="compact"):
                preset_closed_loop = gr.Checkbox(
                    value=True,
                    label="Closed Loop",
                    info="Loop back to start smoothly"
                )

            gr.Markdown("### Randomization")

            with FormRow(variant="compact"):
                preset_randomize = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    step=0.1,
                    label="Randomize Amount",
                    info="0 = pure preset, 1 = maximum variation (adds random perturbations)"
                )

            with FormRow(variant="compact"):
                preset_random_seed = gr.Number(
                    value=-1,
                    label="Random Seed",
                    info="-1 = random seed each time, fixed value = reproducible randomization"
                )

            # Purple gradient slopcore buttons
            with FormRow(variant="compact"):
                with gr.Column(scale=2):
                    btn_generate_preset = gr.Button(
                        f"{emoji_utils.wan_video()} Generate Preset Path",
                        variant="primary",
                        elem_id="btn_generate_preset",
                        elem_classes=["slopcore-button"]
                    )
                with gr.Column(scale=1):
                    btn_randomize_preset = gr.Button(
                        f"{emoji_utils.leaf()} Randomize",
                        variant="secondary",
                        elem_id="btn_randomize_preset",
                        elem_classes=["slopcore-button"]
                    )

            with FormRow(variant="compact"):
                preset_status = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False,
                    lines=2
                )

        # ===== CUSTOM SPLINE TAB =====
        custom_emoji = emoji_utils.palette() + " " if emoji_utils.palette() else ""
        with gr.Tab(f"{custom_emoji}Custom Spline"):
            gr.Markdown("### Control Points")

            gr.Markdown("Define waypoints for camera path. Camera will smoothly interpolate between these points.")

            with FormRow(variant="compact"):
                num_control_points = gr.Slider(
                    minimum=3,
                    maximum=20,
                    value=6,
                    step=1,
                    label="Number of Control Points",
                    info="Waypoints along the path"
                )

            with FormRow(variant="compact"):
                spline_type = gr.Dropdown(
                    choices=["catmull_rom", "linear"],
                    value="catmull_rom",
                    label="Spline Type",
                    info="Catmull-Rom = smooth curves, Linear = straight segments"
                )

            with FormRow(variant="compact"):
                spline_smoothness = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="Smoothness",
                    info="0 = sharp turns, 1 = very smooth"
                )

            with FormRow(variant="compact"):
                look_at_curve = gr.Checkbox(
                    value=True,
                    label="Look Along Path",
                    info="Camera looks tangent to curve (forward direction)"
                )

            with FormRow(variant="compact"):
                custom_num_frames = gr.Number(
                    value=333,
                    label="Number of Frames"
                )

            with FormRow(variant="compact"):
                custom_closed_loop = gr.Checkbox(
                    value=False,
                    label="Closed Loop"
                )

            gr.Markdown("### Control Point Positions (X, Y, Z)")

            # We'll generate control points procedurally for now
            # In future could add manual entry
            with FormRow(variant="compact"):
                control_point_pattern = gr.Dropdown(
                    choices=["circle", "figure-eight", "random", "line"],
                    value="circle",
                    label="Control Point Pattern",
                    info="How to distribute control points"
                )

            with FormRow(variant="compact"):
                control_pattern_scale = gr.Slider(
                    minimum=10,
                    maximum=500,
                    value=100,
                    step=10,
                    label="Pattern Scale"
                )

            # Purple gradient slopcore button
            with FormRow(variant="compact"):
                btn_generate_custom = gr.Button(
                    f"{emoji_utils.palette()} Generate Custom Path",
                    variant="primary",
                    elem_id="btn_generate_custom",
                    elem_classes=["slopcore-button"]
                )

            with FormRow(variant="compact"):
                custom_status = gr.Textbox(
                    label="Status",
                    value="",
                    interactive=False,
                    lines=2
                )

        # Visualization tab removed - visualization now in right panel for real-time feedback

    # Store components for event handlers
    components.update({
        'preset_type': preset_type,
        'preset_radius': preset_radius,
        'preset_height': preset_height,
        'preset_rotation_factor': preset_rotation_factor,
        'preset_num_frames': preset_num_frames,
        'preset_closed_loop': preset_closed_loop,
        'preset_randomize': preset_randomize,
        'preset_random_seed': preset_random_seed,
        'btn_generate_preset': btn_generate_preset,
        'btn_randomize_preset': btn_randomize_preset,
        'preset_status': preset_status,
        'num_control_points': num_control_points,
        'spline_type': spline_type,
        'spline_smoothness': spline_smoothness,
        'look_at_curve': look_at_curve,
        'custom_num_frames': custom_num_frames,
        'custom_closed_loop': custom_closed_loop,
        'control_point_pattern': control_point_pattern,
        'control_pattern_scale': control_pattern_scale,
        'btn_generate_custom': btn_generate_custom,
        'custom_status': custom_status,
        # Visualization removed from tab - now in right panel
        # Include references to schedule textboxes (will be passed from parent)
        'translation_x': None,  # Will be set by parent
        'translation_y': None,
        'translation_z': None,
        'rotation_3d_x': None,
        'rotation_3d_y': None,
        'rotation_3d_z': None,
    })

    return components
