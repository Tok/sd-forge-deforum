"""3D Camera Path Tab

Generates smooth camera paths using splines with visualization.
Populates translation/rotation schedules for Deforum animation.
"""

import gradio as gr
from types import SimpleNamespace
from deforum.utils.system.logging import emoji as emoji_utils
from modules.ui_components import FormRow, FormColumn, ToolButton
from deforum.utils.ui.builders import create_gr_elem, create_row


def get_tab_camera_path(da: SimpleNamespace, skip_tabitem=False):
    """Create the 3D Camera Path tab.

    Args:
        da: DeforumAnimArgs namespace
        skip_tabitem: If True, don't wrap in TabItem (for embedding)

    Returns:
        Dict of Gradio components
    """
    components = {}

    if skip_tabitem:
        # When embedding as subtab, just build UI in current context
        _build_camera_path_ui(da, components)
        return components
    else:
        # Main tab title respects emoji setting
        tab_emoji = emoji_utils.wan_video() + " " if emoji_utils.wan_video() else ""
        with gr.TabItem(f"{tab_emoji}Camera Path", elem_id='camera_path_tab'):
            _build_camera_path_ui(da, components)
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
                btn_save_camera_path_as_default = gr.Button(
                    f"{emoji_utils.floppy_disk()} Save Camera Path as Default",
                    variant="primary",
                    elem_id="btn_save_camera_path_as_default",
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

        # ===== MOTION TAB ===== (moved from Keyframes tab)
        motion_emoji = emoji_utils.bicycle() + " " if emoji_utils.bicycle() else ""
        with gr.Tab(f"{motion_emoji}Motion"):
            gr.Markdown("### Manual Motion Schedules")
            gr.Markdown("Define precise camera movement schedules by frame number. Use this for fine-tuning or when presets don't fit your needs.")

            with FormColumn() as only_2d_motion_column:
                with FormRow(variant="compact"):
                    zoom = create_gr_elem(da.zoom)
                    reset_zoom_button = ToolButton(
                        elem_id='reset_zoom_btn',
                        value=emoji_utils.refresh,
                        tooltip="Reset zoom to static."
                    )
                    components['zoom'] = zoom

                    def reset_zoom_field():
                        return {zoom: gr.update(value='0:(1)', visible=True)}

                    reset_zoom_button.click(fn=reset_zoom_field, inputs=[], outputs=[zoom])

                angle = create_row(da.angle)
                transform_center_x = create_row(da.transform_center_x)
                transform_center_y = create_row(da.transform_center_y)

            with FormColumn() as both_anim_mode_motion_params_column:
                translation_x = create_row(da.translation_x)
                translation_y = create_row(da.translation_y)

            is_3d_motion_column_visible = True  # FIXME init, overridden because default is 3D
            with FormColumn(visible=is_3d_motion_column_visible) as only_3d_motion_column:
                with FormRow():
                    translation_z = create_gr_elem(da.translation_z)
                    reset_tr_z_button = ToolButton(
                        elem_id='reset_tr_z_btn',
                        value=emoji_utils.refresh,
                        tooltip="Reset translation Z to static."
                    )
                    components['tr_z'] = translation_z

                    def reset_tr_z_field():
                        return {translation_z: gr.update(value='0:(0)', visible=True)}

                    reset_tr_z_button.click(fn=reset_tr_z_field, inputs=[], outputs=[translation_z])

                rotation_3d_x = create_row(da.rotation_3d_x)
                rotation_3d_y = create_row(da.rotation_3d_y)
                rotation_3d_z = create_row(da.rotation_3d_z)

            # PERSPECTIVE FLIP - inner params are hidden if not enabled
            with FormRow() as enable_per_f_row:
                enable_perspective_flip = create_gr_elem(da.enable_perspective_flip)
            with FormRow(visible=False) as per_f_th_row:
                perspective_flip_theta = create_gr_elem(da.perspective_flip_theta)
            with FormRow(visible=False) as per_f_ph_row:
                perspective_flip_phi = create_gr_elem(da.perspective_flip_phi)
            with FormRow(visible=False) as per_f_ga_row:
                perspective_flip_gamma = create_gr_elem(da.perspective_flip_gamma)
            with FormRow(visible=False) as per_f_f_row:
                perspective_flip_fv = create_gr_elem(da.perspective_flip_fv)

        # Visualization tab removed - visualization now in right panel for real-time feedback

    # Store components for event handlers
    # Collect all local variables (motion components are already in locals)
    all_components = {k: v for k, v in {**locals(), **vars()}.items()}

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
        'btn_save_camera_path_as_default': btn_save_camera_path_as_default,
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
        # Motion components (moved from Keyframes tab)
        'zoom': all_components.get('zoom'),
        'angle': all_components.get('angle'),
        'transform_center_x': all_components.get('transform_center_x'),
        'transform_center_y': all_components.get('transform_center_y'),
        'translation_x': all_components.get('translation_x'),
        'translation_y': all_components.get('translation_y'),
        'translation_z': all_components.get('translation_z'),
        'rotation_3d_x': all_components.get('rotation_3d_x'),
        'rotation_3d_y': all_components.get('rotation_3d_y'),
        'rotation_3d_z': all_components.get('rotation_3d_z'),
        'enable_perspective_flip': all_components.get('enable_perspective_flip'),
        'perspective_flip_theta': all_components.get('perspective_flip_theta'),
        'perspective_flip_phi': all_components.get('perspective_flip_phi'),
        'perspective_flip_gamma': all_components.get('perspective_flip_gamma'),
        'perspective_flip_fv': all_components.get('perspective_flip_fv'),
        'only_2d_motion_column': all_components.get('only_2d_motion_column'),
        'both_anim_mode_motion_params_column': all_components.get('both_anim_mode_motion_params_column'),
        'only_3d_motion_column': all_components.get('only_3d_motion_column'),
        'enable_per_f_row': all_components.get('enable_per_f_row'),
        'per_f_th_row': all_components.get('per_f_th_row'),
        'per_f_ph_row': all_components.get('per_f_ph_row'),
        'per_f_ga_row': all_components.get('per_f_ga_row'),
        'per_f_f_row': all_components.get('per_f_f_row'),
    })

    return components
