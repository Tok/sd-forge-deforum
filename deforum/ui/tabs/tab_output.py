"""Output tab for Deforum UI.

Contains video output settings, frame interpolation (RIFE/FILM), video upscaling,
and frames-to-video stitching utilities.
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn
from deforum.utils.system.logging import emoji as emoji_utils
from deforum.utils.ui.builders import create_gr_elem, create_row
from deforum.config.defaults import get_gradio_html
from deforum.ui.gradio_funcs import (
    upload_vid_to_interpolate,
    upload_pics_to_interpolate,
    ncnn_upload_vid_to_upscale
)
from deforum.media.video_audio_utilities import direct_stitch_vid_from_frames


def get_tab_output(da, dv):
    """Create the Output tab with video output and post-processing settings.

    Args:
        da: DeforumAnimArgs namespace
        dv: DeforumOutputArgs namespace

    Returns:
        dict: Component dictionary for event binding
    """
    with gr.TabItem(f"{emoji_utils.document()} Output", elem_id='output_tab'):
        # VID OUTPUT ACCORD
        with gr.Accordion('Video Output Settings', open=True):
            # fps moved to top-level setting in ui_left.py - create hidden copy for button handlers
            with gr.Row(visible=False):
                fps = create_gr_elem(dv.fps)
                add_soundtrack = create_gr_elem(dv.add_soundtrack)
                soundtrack_path = create_gr_elem(dv.soundtrack_path)

            with FormColumn():
                with FormRow():
                    skip_video_creation = create_gr_elem(dv.skip_video_creation)
                    delete_imgs = create_gr_elem(dv.delete_imgs)
                    delete_input_frames = create_gr_elem(dv.delete_input_frames)
                    store_frames_in_ram = create_gr_elem(dv.store_frames_in_ram)
                    save_depth_maps = create_gr_elem(da.save_depth_maps)
                    make_gif = create_gr_elem(dv.make_gif)
            with FormRow(equal_height=True) as r_upscale_row:
                r_upscale_video = create_gr_elem(dv.r_upscale_video)
                r_upscale_model = create_gr_elem(dv.r_upscale_model)
                r_upscale_factor = create_gr_elem(dv.r_upscale_factor)
                r_upscale_keep_imgs = create_gr_elem(dv.r_upscale_keep_imgs)

        # FRAME INTERPOLATION TAB
        with gr.Tab('Frame Interpolation') as frame_interp_tab:
            with gr.Accordion('Important notes and Help', open=False, elem_id="f_interp_accord"):
                gr.HTML(value=get_gradio_html('frame_interpolation'))
            with gr.Column():
                with gr.Row():
                    # Interpolation Engine
                    with gr.Column(min_width=110, scale=3):
                        frame_interpolation_engine = create_gr_elem(dv.frame_interpolation_engine)
                    with gr.Column(min_width=30, scale=1):
                        frame_interpolation_slow_mo_enabled = create_gr_elem(dv.frame_interpolation_slow_mo_enabled)
                    with gr.Column(min_width=30, scale=1):
                        # If this is set to True, we keep all the interpolated frames in a folder. Default is False - means we delete them at the end of the run
                        frame_interpolation_keep_imgs = create_gr_elem(dv.frame_interpolation_keep_imgs)
                    with gr.Column(min_width=30, scale=1):
                        frame_interpolation_use_upscaled = create_gr_elem(dv.frame_interpolation_use_upscaled)
                with FormRow(visible=False) as frame_interp_amounts_row:
                    with gr.Column(min_width=180) as frame_interp_x_amount_column:
                        # How many times to interpolate (interp X)
                        frame_interpolation_x_amount = create_gr_elem(dv.frame_interpolation_x_amount)
                    with gr.Column(min_width=180, visible=False) as frame_interp_slow_mo_amount_column:
                        # Interp Slow-Mo (setting final output fps, not really doing anything directly with RIFE/FILM)
                        frame_interpolation_slow_mo_amount = create_gr_elem(dv.frame_interpolation_slow_mo_amount)
                with gr.Row(visible=False) as interp_existing_video_row:
                    # Interpolate any existing video from the connected PC
                    with gr.Accordion('Interpolate existing Video/ Images', open=False) as interp_existing_video_accord:
                        with gr.Row(variant='compact') as interpolate_upload_files_row:
                            # A drag-n-drop UI box to which the user uploads a *single* (at this stage) video
                            vid_to_interpolate_chosen_file = gr.File(
                                label="Video to Interpolate",
                                interactive=True,
                                file_count="single",
                                file_types=["video"],
                                elem_id="vid_to_interpolate_chosen_file"
                            )
                            # A drag-n-drop UI box to which the user uploads a pictures to interpolate
                            pics_to_interpolate_chosen_file = gr.File(
                                label="Pics to Interpolate",
                                interactive=True,
                                file_count="multiple",
                                file_types=["image"],
                                elem_id="pics_to_interpolate_chosen_file"
                            )
                        with FormRow(visible=False) as interp_live_stats_row:
                            # Non-interactive textbox showing uploaded input vid total Frame Count
                            in_vid_frame_count_window = gr.Textbox(
                                label="In Frame Count",
                                lines=1,
                                interactive=False,
                                value='---'
                            )
                            # Non-interactive textbox showing uploaded input vid FPS
                            in_vid_fps_ui_window = gr.Textbox(
                                label="In FPS",
                                lines=1,
                                interactive=False,
                                value='---'
                            )
                            # Non-interactive textbox showing expected output interpolated video FPS
                            out_interp_vid_estimated_fps = gr.Textbox(
                                label="Interpolated Vid FPS",
                                value='---'
                            )
                        with FormRow() as interp_buttons_row:
                            # This is the actual button that's pressed to initiate the interpolation:
                            interpolate_button = gr.Button(value="*Interpolate Video*")
                            interpolate_pics_button = gr.Button(value="*Interpolate Pics*")
                        # Show a text about CLI outputs:
                        gr.HTML("* check your CLI for outputs *", elem_id="below_interpolate_butts_msg")
                        # make the function call when the interpolation button is clicked
                        interpolate_button.click(
                            fn=upload_vid_to_interpolate,
                            inputs=[
                                vid_to_interpolate_chosen_file,
                                frame_interpolation_engine,
                                frame_interpolation_x_amount,
                                frame_interpolation_slow_mo_enabled,
                                frame_interpolation_slow_mo_amount,
                                frame_interpolation_keep_imgs,
                                in_vid_fps_ui_window
                            ]
                        )
                        interpolate_pics_button.click(
                            fn=upload_pics_to_interpolate,
                            inputs=[
                                pics_to_interpolate_chosen_file,
                                frame_interpolation_engine,
                                frame_interpolation_x_amount,
                                frame_interpolation_slow_mo_enabled,
                                frame_interpolation_slow_mo_amount,
                                frame_interpolation_keep_imgs,
                                fps,
                                add_soundtrack,
                                soundtrack_path
                            ]
                        )

        # VIDEO UPSCALE TAB - not built using our args.py at all - all data and params are here and in .upscaling file
        with gr.TabItem(f"{emoji_utils.up()} Video Upscaling"):
            vid_to_upscale_chosen_file = gr.File(
                label="Video to Upscale",
                interactive=True,
                file_count="single",
                file_types=["video"],
                elem_id="vid_to_upscale_chosen_file"
            )
            with gr.Column():
                # NCNN UPSCALE TAB
                with FormRow() as ncnn_upload_vid_stats_row:
                    # Non-interactive textbox showing uploaded input vid Frame Count
                    ncnn_upscale_in_vid_frame_count_window = gr.Textbox(
                        label="In Frame Count",
                        lines=1,
                        interactive=False,
                        value='---'
                    )
                    # Non-interactive textbox showing uploaded input vid FPS
                    ncnn_upscale_in_vid_fps_ui_window = gr.Textbox(
                        label="In FPS",
                        lines=1,
                        interactive=False,
                        value='---'
                    )
                    # Non-interactive textbox showing uploaded input resolution
                    ncnn_upscale_in_vid_res = gr.Textbox(
                        label="In Res",
                        lines=1,
                        interactive=False,
                        value='---'
                    )
                    # Non-interactive textbox showing expected output resolution
                    ncnn_upscale_out_vid_res = gr.Textbox(
                        label="Out Res",
                        value='---'
                    )
                with gr.Column():
                    with FormRow() as ncnn_actual_upscale_row:
                        # note that we re-use *r_upscale_model* in here to create the gradio element as they are the same
                        ncnn_upscale_model = create_gr_elem(dv.r_upscale_model)
                        # note that we re-use *r_upscale_factor* in here to create the gradio element as they are the same
                        ncnn_upscale_factor = create_gr_elem(dv.r_upscale_factor)
                        # note that we re-use *r_upscale_keep_imgs* in here to create the gradio element as they are the same
                        ncnn_upscale_keep_imgs = create_gr_elem(dv.r_upscale_keep_imgs)
                ncnn_upscale_btn = gr.Button(value="*Upscale uploaded video*")
                ncnn_upscale_btn.click(
                    fn=ncnn_upload_vid_to_upscale,
                    inputs=[
                        vid_to_upscale_chosen_file,
                        ncnn_upscale_in_vid_fps_ui_window,
                        ncnn_upscale_in_vid_res,
                        ncnn_upscale_out_vid_res,
                        ncnn_upscale_model,
                        ncnn_upscale_factor,
                        ncnn_upscale_keep_imgs
                    ]
                )

        # STITCH FRAMES TO VID TAB
        with gr.TabItem(f"{emoji_utils.frames()} Frames to Video") as stitch_imgs_to_vid_row:
            gr.HTML(value=get_gradio_html('frames_to_video'))
            image_path = create_row(dv.image_path)
            ffmpeg_stitch_imgs_but = gr.Button(value="*Stitch frames to video*")
            ffmpeg_stitch_imgs_but.click(
                fn=direct_stitch_vid_from_frames,
                inputs=[image_path, fps, add_soundtrack, soundtrack_path]
            )

    return {k: v for k, v in {**locals(), **vars()}.items()}
