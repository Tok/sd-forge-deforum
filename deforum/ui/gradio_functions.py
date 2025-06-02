import gradio as gr
import modules.paths as ph
from .general_utils import get_os
from ..media.image_upscaling import process_ncnn_upscale_vid_upload_logic
from ..media.video_audio_pipeline import extract_number, get_quick_vid_info, get_ffmpeg_params
from ..media.frame_interpolation_pipeline import process_interp_vid_upload_logic, process_interp_pics_upload_logic, gradio_f_interp_get_fps_and_fcount
from ..depth.video_depth_extraction import process_depth_vid_upload_logic
import os
import cv2
import subprocess

f_models_path = ph.models_path + '/Deforum'

def handle_change_functions(l_vars):
    """
    Set up Gradio component change handlers.
    
    Args:
        l_vars: Dictionary of local variables containing UI components
    """
    try:
        # Helper function to safely get components
        def safe_get_component(component_name):
            if component_name in l_vars:
                return l_vars[component_name]
            else:
                print(f"⚠️ Warning: UI component '{component_name}' not found, skipping related event handlers")
                return None
        
        # Helper function to safely set up change handlers
        def safe_change_handler(component, handler_func, inputs, outputs):
            # Validate the main component
            if component is None or not hasattr(component, '_id'):
                return
            
            # Validate inputs - can be single component or list
            if isinstance(inputs, list):
                valid_inputs = [inp for inp in inputs if inp is not None and hasattr(inp, '_id')]
                if len(valid_inputs) != len(inputs):
                    return  # Skip if any input is invalid
                inputs = valid_inputs
            else:
                if inputs is None or not hasattr(inputs, '_id'):
                    return
            
            # Validate outputs - can be single component or list  
            if isinstance(outputs, list):
                valid_outputs = [out for out in outputs if out is not None and hasattr(out, '_id')]
                if not valid_outputs:
                    return  # Skip if no valid outputs
                outputs = valid_outputs
            else:
                if outputs is None or not hasattr(outputs, '_id'):
                    return
            
            # Set up the change handler with validated components
            try:
                component.change(fn=handler_func, inputs=inputs, outputs=outputs)
            except Exception as e:
                print(f"⚠️ Error setting up change handler for {getattr(component, 'elem_id', 'unknown')}: {e}")
        
        # Original change handlers with error handling
        override_settings_with_file = safe_get_component('override_settings_with_file')
        custom_settings_file = safe_get_component('custom_settings_file')
        if override_settings_with_file and custom_settings_file:
            safe_change_handler(override_settings_with_file, hide_if_false, override_settings_with_file, custom_settings_file)
        
        sampler = safe_get_component('sampler')
        enable_ddim_eta_scheduling = safe_get_component('enable_ddim_eta_scheduling')
        enable_ancestral_eta_scheduling = safe_get_component('enable_ancestral_eta_scheduling')
        ancestral_eta_schedule = safe_get_component('ancestral_eta_schedule')
        ddim_eta_schedule = safe_get_component('ddim_eta_schedule')
        
        if sampler and enable_ddim_eta_scheduling:
            safe_change_handler(sampler, show_when_ddim, sampler, enable_ddim_eta_scheduling)
        if sampler and enable_ancestral_eta_scheduling:
            safe_change_handler(sampler, show_when_ancestral_samplers, sampler, enable_ancestral_eta_scheduling)
        if enable_ancestral_eta_scheduling and ancestral_eta_schedule:
            safe_change_handler(enable_ancestral_eta_scheduling, hide_if_false, enable_ancestral_eta_scheduling, ancestral_eta_schedule)
        if enable_ddim_eta_scheduling and ddim_eta_schedule:
            safe_change_handler(enable_ddim_eta_scheduling, hide_if_false, enable_ddim_eta_scheduling, ddim_eta_schedule)
        
        animation_mode = safe_get_component('animation_mode')
        max_frames = safe_get_component('max_frames')
        if animation_mode and max_frames:
            safe_change_handler(animation_mode, change_max_frames_visibility, animation_mode, max_frames)
        
        # Diffusion cadence outputs - handle missing components gracefully
        diffusion_cadence = safe_get_component('diffusion_cadence')
        optical_flow_cadence_row = safe_get_component('optical_flow_cadence_row')
        cadence_flow_factor_schedule = safe_get_component('cadence_flow_factor_schedule')
        optical_flow_redo_generation = safe_get_component('optical_flow_redo_generation')
        redo_flow_factor_schedule = safe_get_component('redo_flow_factor_schedule')
        diffusion_redo = safe_get_component('diffusion_redo')
        
        diffusion_cadence_outputs = [comp for comp in [diffusion_cadence, optical_flow_cadence_row, cadence_flow_factor_schedule,
                                     optical_flow_redo_generation, redo_flow_factor_schedule, diffusion_redo] if comp is not None]
        
        if animation_mode:
            for output in diffusion_cadence_outputs:
                safe_change_handler(animation_mode, change_diffusion_cadence_visibility, animation_mode, output)
        
        # Continue with remaining handlers, using safe access
        if animation_mode:
            for comp_name in ['only_3d_motion_column', 'depth_warp_row_1', 'depth_warp_row_2', 'depth_warp_row_3', 'depth_warp_row_4',
                             'depth_warp_row_5', 'depth_warp_row_6', 'depth_warp_row_7']:
                comp = safe_get_component(comp_name)
                if comp:
                    safe_change_handler(animation_mode, disble_3d_related_stuff, animation_mode, comp)
        
        enable_perspective_flip = safe_get_component('enable_perspective_flip')
        if animation_mode and enable_perspective_flip:
            for comp_name in ['per_f_th_row', 'per_f_ph_row', 'per_f_ga_row', 'per_f_f_row']:
                comp = safe_get_component(comp_name)
                if comp:
                    safe_change_handler(enable_perspective_flip, hide_if_false, enable_perspective_flip, comp)
                    safe_change_handler(animation_mode, per_flip_handle, [animation_mode, enable_perspective_flip], comp)
        
        # More safe handlers
        if animation_mode:
            for comp_name, handler in [
                ('depth_warp_msg_html', only_show_in_non_3d_mode),
                ('only_2d_motion_column', enable_2d_related_stuff),
                ('color_force_grayscale', disable_by_interpolation),
                ('noise_tab_column', disable_by_interpolation),
                ('enable_per_f_row', disable_pers_flip_accord),
                ('both_anim_mode_motion_params_column', disable_pers_flip_accord)
            ]:
                comp = safe_get_component(comp_name)
                if comp:
                    safe_change_handler(animation_mode, handler, animation_mode, comp)
        
        # Continue with rest of handlers using safe access
        for comp_name, input_comp, output_comp, handler in [
            ('aspect_ratio_use_old_formula', 'aspect_ratio_use_old_formula', 'aspect_ratio_schedule', hide_if_true),
            ('optical_flow_redo_generation', 'optical_flow_redo_generation', 'redo_flow_factor_schedule_column', hide_if_none),
            ('optical_flow_cadence', 'optical_flow_cadence', 'cadence_flow_factor_schedule_column', hide_if_none),
            ('seed_behavior', 'seed_behavior', 'seed_iter_N_row', change_seed_iter_visibility),
            ('seed_behavior', 'seed_behavior', 'seed_schedule_row', change_seed_schedule_visibility),
            ('color_coherence', 'color_coherence', 'color_coherence_video_every_N_frames_row', change_color_coherence_video_every_N_frames_visibility),
            ('color_coherence', 'color_coherence', 'color_coherence_image_path_row', change_color_coherence_image_path_visibility),
            ('noise_type', 'noise_type', 'perlin_row', change_perlin_visibility),
            ('depth_algorithm', 'depth_algorithm', 'midas_weight', legacy_3d_mode),
            ('depth_algorithm', 'depth_algorithm', 'leres_license_msg', show_leres_html_msg),
            ('fps', 'fps', 'make_gif', change_gif_button_visibility),
            ('r_upscale_model', 'r_upscale_model', 'r_upscale_factor', update_r_upscale_factor),
            ('ncnn_upscale_model', 'ncnn_upscale_model', 'ncnn_upscale_factor', update_r_upscale_factor)
        ]:
            input_component = safe_get_component(input_comp)
            output_component = safe_get_component(output_comp)
            if input_component and output_component:
                safe_change_handler(input_component, handler, input_component, output_component)
        
        # Special handlers with multiple inputs/outputs
        if diffusion_cadence and optical_flow_cadence_row:
            safe_change_handler(diffusion_cadence, hide_optical_flow_cadence, diffusion_cadence, optical_flow_cadence_row)
        
        # Skip video creation outputs
        skip_video_creation = safe_get_component('skip_video_creation')
        if skip_video_creation:
            for comp_name in ['fps_out_format_row', 'soundtrack_row', 'store_frames_in_ram', 'make_gif', 'r_upscale_row',
                             'delete_imgs', 'delete_input_frames']:
                comp = safe_get_component(comp_name)
                if comp:
                    safe_change_handler(skip_video_creation, change_visibility_from_skip_video, skip_video_creation, comp)
        
        # Frame interpolation handlers
        frame_interpolation_slow_mo_enabled = safe_get_component('frame_interpolation_slow_mo_enabled')
        frame_interp_slow_mo_amount_column = safe_get_component('frame_interp_slow_mo_amount_column')
        if frame_interpolation_slow_mo_enabled and frame_interp_slow_mo_amount_column:
            safe_change_handler(frame_interpolation_slow_mo_enabled, hide_if_false, frame_interpolation_slow_mo_enabled, frame_interp_slow_mo_amount_column)
        
        frame_interpolation_engine = safe_get_component('frame_interpolation_engine')
        frame_interpolation_x_amount = safe_get_component('frame_interpolation_x_amount')
        if frame_interpolation_engine and frame_interpolation_x_amount:
            safe_change_handler(frame_interpolation_engine, change_interp_x_max_limit, [frame_interpolation_engine, frame_interpolation_x_amount], frame_interpolation_x_amount)
        
        # Interpolation hide list
        if frame_interpolation_engine:
            for comp_name in ['frame_interpolation_slow_mo_enabled', 'frame_interpolation_keep_imgs', 'frame_interpolation_use_upscaled', 'frame_interp_amounts_row']:
                comp = safe_get_component(comp_name)
                if comp:
                    safe_change_handler(frame_interpolation_engine, hide_interp_by_interp_status, frame_interpolation_engine, comp)
    except Exception as e:
        print(f"⚠️ Error in handle_change_functions: {e}")
        print("Extension will continue loading with limited UI functionality")

# START gradio-to-frame-interoplation/ upscaling functions
def upload_vid_to_interpolate(file, engine, x_am, sl_enabled, sl_am, keep_imgs, in_vid_fps):
    # print msg and do nothing if vid not uploaded or interp_x not provided
    if not file or engine == 'None':
        return print("Please upload a video and set a proper value for 'Interp X'. Can't interpolate x0 times :)")
    f_location, f_crf, f_preset = get_ffmpeg_params()

    process_interp_vid_upload_logic(file, engine, x_am, sl_enabled, sl_am, keep_imgs, f_location, f_crf, f_preset, in_vid_fps, f_models_path, file.name)

def upload_pics_to_interpolate(pic_list, engine, x_am, sl_enabled, sl_am, keep_imgs, fps, add_audio, audio_track):
    from PIL import Image

    if pic_list is None or len(pic_list) < 2:
        return print("Please upload at least 2 pics for interpolation.")
    f_location, f_crf, f_preset = get_ffmpeg_params()
    # make sure all uploaded pics have the same resolution
    pic_sizes = [Image.open(picture_path.name).size for picture_path in pic_list]
    if len(set(pic_sizes)) != 1:
        return print("All uploaded pics need to be of the same Width and Height / resolution.")

    resolution = pic_sizes[0]

    process_interp_pics_upload_logic(pic_list, engine, x_am, sl_enabled, sl_am, keep_imgs, f_location, f_crf, f_preset, fps, f_models_path, resolution, add_audio, audio_track)

def ncnn_upload_vid_to_upscale(vid_path, in_vid_fps, in_vid_res, out_vid_res, upscale_model, upscale_factor, keep_imgs):
    if vid_path is None:
        print("Please upload a video :)")
        return
    f_location, f_crf, f_preset = get_ffmpeg_params()
    current_user = get_os()
    process_ncnn_upscale_vid_upload_logic(vid_path, in_vid_fps, in_vid_res, out_vid_res, f_models_path, upscale_model, upscale_factor, keep_imgs, f_location, f_crf, f_preset, current_user)

def upload_vid_to_depth(vid_to_depth_chosen_file, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth, depth_keep_imgs):
    # print msg and do nothing if vid not uploaded
    if not vid_to_depth_chosen_file:
        return print("Please upload a video :()")
    f_location, f_crf, f_preset = get_ffmpeg_params()

    process_depth_vid_upload_logic(vid_to_depth_chosen_file, mode, thresholding, threshold_value, threshold_value_max, adapt_block_size, adapt_c, invert, end_blur, midas_weight_vid2depth,
                                   vid_to_depth_chosen_file.name, depth_keep_imgs, f_location, f_crf, f_preset, f_models_path)

# END gradio-to-frame-interoplation/ upscaling functions

def change_visibility_from_skip_video(choice):
    return gr.update(visible=False) if choice else gr.update(visible=True)

def update_r_upscale_factor(choice):
    return gr.update(value='x4', choices=['x4']) if choice != 'realesr-animevideov3' else gr.update(value='x2', choices=['x2', 'x3', 'x4'])

def change_perlin_visibility(choice):
    return gr.update(visible=choice == "perlin")

def legacy_3d_mode(choice):
    return gr.update(visible=choice.lower() in ["midas+adabins (old)", 'zoe+adabins (old)'])

def change_color_coherence_image_path_visibility(choice):
    return gr.update(visible=choice == "Image")

def change_color_coherence_video_every_N_frames_visibility(choice):
    return gr.update(visible=choice == "Video Input")

def change_seed_iter_visibility(choice):
    return gr.update(visible=choice == "iter")

def change_seed_schedule_visibility(choice):
    return gr.update(visible=choice == "schedule")

def disable_pers_flip_accord(choice):
    return gr.update(visible=True) if choice in ['2D', '3D'] else gr.update(visible=False)

def per_flip_handle(anim_mode, per_f_enabled):
    if anim_mode in ['2D', '3D'] and per_f_enabled:
        return gr.update(visible=True)
    return gr.update(visible=False)

def change_max_frames_visibility(choice):
    return gr.update(visible=choice != "Video Input")

def change_diffusion_cadence_visibility(choice):
    return gr.update(visible=choice not in ['Video Input', 'Interpolation'])

def disble_3d_related_stuff(choice):
    return gr.update(visible=False) if choice != '3D' else gr.update(visible=True)

def only_show_in_non_3d_mode(choice):
    return gr.update(visible=False) if choice == '3D' else gr.update(visible=True)

def enable_2d_related_stuff(choice):
    return gr.update(visible=True) if choice == '2D' else gr.update(visible=False)

def disable_by_interpolation(choice):
    return gr.update(visible=False) if choice in ['Interpolation'] else gr.update(visible=True)

def disable_by_video_input(choice):
    return gr.update(visible=False) if choice in ['Video Input'] else gr.update(visible=True)

def hide_if_none(choice):
    return gr.update(visible=choice != "None")

def change_gif_button_visibility(choice):
    if choice is None or choice == "":
        return gr.update(visible=True)
    return gr.update(visible=False, value=False) if int(choice) > 30 else gr.update(visible=True)

def hide_if_false(choice):
    return gr.update(visible=True) if choice else gr.update(visible=False)

def hide_if_true(choice):
    return gr.update(visible=False) if choice else gr.update(visible=True)

def hide_optical_flow_cadence(cadence_value):
    return gr.update(visible=True) if cadence_value > 1 else gr.update(visible=False)

def hide_interp_by_interp_status(choice):
    return gr.update(visible=False) if choice == 'None' else gr.update(visible=True)

def change_interp_x_max_limit(engine_name, current_value):
    if engine_name == 'FILM':
        return gr.update(maximum=300)
    elif current_value > 10:
        return gr.update(maximum=10, value=2)
    return gr.update(maximum=10)

def hide_interp_stats(choice):
    return gr.update(visible=True) if choice is not None else gr.update(visible=False)

def show_leres_html_msg(choice):
    return gr.update(visible=True) if choice.lower() == 'leres' else gr.update(visible=False)

def show_when_ddim(sampler_name):
    return gr.update(visible=True) if sampler_name.lower() == 'ddim' else gr.update(visible=False)

def show_when_ancestral_samplers(sampler_name):
    return gr.update(visible=True) if sampler_name.lower() in ['euler a', 'dpm++ 2s a', 'dpm2 a', 'dpm2 a karras', 'dpm++ 2s a karras'] else gr.update(visible=False)

def change_css(checkbox_status):
    if checkbox_status:
        display = "block"
    else:
        display = "none"

    html_template = f'''
        <style>
            #tab_deforum_interface .svelte-e8n7p6, #f_interp_accord {{
                display: {display} !important;
            }}
        </style>
        '''
    return html_template

# Upscaling Gradio UI related funcs
def vid_upscale_gradio_update_stats(vid_path, upscale_factor):
    if not vid_path:
        return '---', '---', '---', '---'
    factor = extract_number(upscale_factor)
    fps, fcount, resolution = get_quick_vid_info(vid_path.name)
    in_res_str = f"{resolution[0]}*{resolution[1]}"
    out_res_str = f"{resolution[0] * factor}*{resolution[1] * factor}"
    return fps, fcount, in_res_str, out_res_str

def update_upscale_out_res(in_res, upscale_factor):
    if not in_res:
        return '---'
    factor = extract_number(upscale_factor)
    w, h = [int(x) * factor for x in in_res.split('*')]
    return f"{w}*{h}"

def update_upscale_out_res_by_model_name(in_res, upscale_model_name):
    if not upscale_model_name or in_res == '---':
        return '---'
    factor = 2 if upscale_model_name == 'realesr-animevideov3' else 4
    return f"{int(in_res.split('*')[0]) * factor}*{int(in_res.split('*')[1]) * factor}"
