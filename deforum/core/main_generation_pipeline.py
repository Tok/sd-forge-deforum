from PIL import Image
import os
import math
import json
import itertools
import requests
import numexpr
from modules import processing, sd_models
from modules.shared import sd_model, state, cmd_opts
from ..integrations.controlnet.core_integration import is_controlnet_enabled, get_controlnet_script_args
from ..prompt import split_weighted_subprompts
from ..media.image_loading import load_img, prepare_mask, check_mask_for_errors
from .webui_sd_pipeline import get_webui_sd_pipeline
from ..utils.rich import console
from .defaults import get_samplers_list, get_schedulers_list
from ..prompt import check_is_number
from .opts_overrider import A1111OptionsOverrider
import cv2
import numpy as np
from .data_models import ProcessingResult

from .deforum_scripts_overrides import add_forge_script_to_deforum_run, initialise_forge_scripts

from .general_utils import debug_print

def load_mask_latent(mask_input, shape):
    # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape

    if isinstance(mask_input, str):  # mask input is probably a file name
        if mask_input.startswith('http://') or mask_input.startswith('https://'):
            mask_image = Image.open(requests.get(mask_input, stream=True).raw).convert('RGBA')
        else:
            mask_image = Image.open(mask_input).convert('RGBA')
    elif isinstance(mask_input, Image.Image):
        mask_image = mask_input
    else:
        raise Exception("mask_input must be a PIL image or a file name")

    mask_w_h = (shape[-1], shape[-2])
    mask = mask_image.resize(mask_w_h, resample=Image.LANCZOS)
    mask = mask.convert("L")
    return mask

def isJson(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True

# Add pairwise implementation here not to upgrade
# the whole python to 3.10 just for one function
def pairwise_repl(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def generate(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter,  frame=0, sampler_name=None, scheduler_name=None):
    # Fallback if no image was produced by any means
    debug_print(f'frame: {frame}')
    try:
        return generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)
    except Exception as e:
        image, caught_vae_exception = generate_with_nans_check(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)
        if caught_vae_exception:
            debug_print(f'VAE decode exception: {e}, making empty frame!')
            image = Image.new('RGB', (args.W, args.H), color='black')
        return image


def auto_fix_and_regenerate(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame=0, sampler_name=None, scheduler_name=None):
    image, caught_vae_exception = generate_with_nans_check(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)
    if caught_vae_exception:
        debug_print(f'VAE decode exception encountered, falling back to auto-fix mode')
        args.enable_vae_auto_fix = False # A simple toggle to prevent any kind of loops.
        # TODO? Auto-detect values that will fix VAE encode/decode BSOD on a per model basis

        # Set VAE to auto to enable automatic switching to fp32
        if args.enable_vae_auto_fix:
            args.vae = 'auto'
            image = generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)
        else:
            image = Image.new('RGB', (args.W, args.H), color='black')
        return image


def generate_with_nans_check(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame=0, sampler_name=None, scheduler_name=None):
    try:
        image = generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)
        caught_vae_exception = False
    except RuntimeError as e:
        image = generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)
        debug_print(f'Runtime exception: {e}, continuing generation')
        caught_vae_exception = False
    except Exception as e:
        debug_print(f'Exception: {e}')
        caught_vae_exception = True
        image = Image.new('RGB', (args.W, args.H), color='black')
    return image, caught_vae_exception


def generate_inner(args, keys, anim_args, loop_args, controlnet_args,
                   root, parseq_adapter, frame=0, sampler_name=None, scheduler_name=None):
    # Get generation args
    current_t = frame
    
    # ControlNet
    cn_script = get_controlnet_script_if_available()
    
    # Looper
    looper_script = get_looper_script_if_available()
    
    # Assign model and other script objects to get data from
    model_wrap = get_webui_sd_pipeline()

    # Setup auto detect for Depth Map gen by ControlNet if it's a depth controlnet
    if is_controlnet_enabled(controlnet_args):
        controlnet_args = get_inpaint_mode_cn_mask(args, anim_args, controlnet_args, root, frame)

    # Setup for adetailer extension support
    adetailer_script_args = get_adetailer_script_args()

    # Run the webui generation for both formats (Automatic1111, ComfyUI)
    if root.current_user_os == 'ComfyUI':
        # Initialize the script args list for ComfyUI mode
        script_args = []
        
        # ControlNet
        if cn_script is not None and is_controlnet_enabled(controlnet_args):
            controlnet_script_args = get_controlnet_script_args(controlnet_args, root, anim_args, current_t)
            script_args.append(controlnet_script_args)
        
        # Looper
        if looper_script is not None and loop_args.use_looper:
            looper_script_args = get_looper_script_args(loop_args, anim_args, frame, keys, current_t)
            script_args.append(looper_script_args)

        # Note: FreeU and Kohya HR Fix functionality removed
        
        # ADetailer
        if adetailer_script_args:
            script_args.append(adetailer_script_args)

        return do_generate_comfyui(args, anim_args, frame, script_args, root, model_wrap, sampler_name, scheduler_name)
    else:
        # Initialize the script args list for A1111 mode  
        script_args = [None] * max_script_num_from_shared()

        # ControlNet
        if cn_script is not None and is_controlnet_enabled(controlnet_args):
            controlnet_script_args = get_controlnet_script_args(controlnet_args, root, anim_args, current_t)
            script_args[cn_script.args_from] = controlnet_script_args
            script_args[cn_script.args_to] = None

        # Looper
        if looper_script is not None and loop_args.use_looper:
            looper_script_args = get_looper_script_args(loop_args, anim_args, frame, keys, current_t)
            script_args[looper_script.args_from] = looper_script_args
            script_args[looper_script.args_to] = None

        # Note: FreeU and Kohya HR Fix functionality removed
        
        # ADetailer
        if adetailer_script_args:
            script_args[get_adetailer_script().args_from] = adetailer_script_args
            script_args[get_adetailer_script().args_to] = None

        return do_generate_a1111(args, anim_args, frame, script_args, root, model_wrap, sampler_name, scheduler_name)

# Run this instead of actual diffusion when doing motion preview.
def mock_process_images(args, p, init_image):
  
    input_image = cv2.cvtColor(np.array(init_image), cv2.COLOR_RGB2BGR)

    start_point = (int(args.H/3), int(args.W/3))
    end_point = (int(args.H-args.H/3), int(args.W-args.W/3))
    color = (255, 255, 255, float(p.denoising_strength))
    thickness = 2
    mock_generated_image = np.zeros_like(input_image, np.uint8)
    cv2.rectangle(mock_generated_image, start_point, end_point, color, thickness)


    blend = cv2.addWeighted(input_image, float(1.0-p.denoising_strength), mock_generated_image, float(p.denoising_strength), 0)

    image = Image.fromarray(cv2.cvtColor(blend, cv2.COLOR_BGR2RGB))
    state.assign_current_image(image)
    return ProcessingResult.create_motion_preview(image)

def print_combined_table(args, anim_args, p, keys, frame_idx):
    from rich.table import Table
    from rich import box

    table = Table(padding=0, box=box.ROUNDED)

    field_names1 = ["Steps", "CFG", "Dist. CFG"]
    if anim_args.animation_mode != 'Interpolation':
        field_names1.append("Denoise")
    field_names1 += ["Subseed", "Subs. str"] * (anim_args.enable_subseed_scheduling)
    field_names1 += ["Sampler"] * anim_args.enable_sampler_scheduling
    field_names1 += ["Scheduler"] * anim_args.enable_scheduler_scheduling
    field_names1 += ["Checkpoint"] * anim_args.enable_checkpoint_scheduling

    for field_name in field_names1:
        table.add_column(field_name, justify="center")

    rows1 = [str(p.steps), str(p.cfg_scale), str(p.distilled_cfg_scale)]
    if anim_args.animation_mode != 'Interpolation':
        rows1.append(f"{p.denoising_strength:.5g}" if p.denoising_strength is not None else "None")

    rows1 += [str(p.subseed), f"{p.subseed_strength:.5g}"] * anim_args.enable_subseed_scheduling
    rows1 += [p.sampler_name] * anim_args.enable_sampler_scheduling
    rows1 += [p.scheduler] * anim_args.enable_scheduler_scheduling
    rows1 += [str(args.checkpoint)] * anim_args.enable_checkpoint_scheduling

    rows2 = []
    if anim_args.animation_mode not in ['Video Input', 'Interpolation']:
        if anim_args.animation_mode == '2D':
            field_names2 = ["Angle", "Zoom", "Tr C X", "Tr C Y"]
        else:
            field_names2 = []
        field_names2 += ["Tr X", "Tr Y"]
        if anim_args.animation_mode == '3D':
            field_names2 += ["Tr Z", "Ro X", "Ro Y", "Ro Z"]
            if anim_args.aspect_ratio_schedule.replace(" ", "") != '0:(1)':
                field_names2 += ["Asp. Ratio"]
        if anim_args.enable_perspective_flip:
            field_names2 += ["Pf T", "Pf P", "Pf G", "Pf F"]

        for field_name in field_names2:
            table.add_column(field_name, justify="center")

        if anim_args.animation_mode == '2D':
            rows2 += [f"{keys.angle_series[frame_idx]:.5g}", f"{keys.zoom_series[frame_idx]:.5g}",
                      f"{keys.transform_center_x_series[frame_idx]:.5g}", f"{keys.transform_center_y_series[frame_idx]:.5g}"]
            
        rows2 += [f"{keys.translation_x_series[frame_idx]:.5g}", f"{keys.translation_y_series[frame_idx]:.5g}"]

        if anim_args.animation_mode == '3D':
            rows2 += [f"{keys.translation_z_series[frame_idx]:.5g}", f"{keys.rotation_3d_x_series[frame_idx]:.5g}",
                      f"{keys.rotation_3d_y_series[frame_idx]:.5g}", f"{keys.rotation_3d_z_series[frame_idx]:.5g}"]
            if anim_args.aspect_ratio_schedule.replace(" ", "") != '0:(1)':
                rows2 += [f"{keys.aspect_ratio_series[frame_idx]:.5g}"]
        if anim_args.enable_perspective_flip:
            rows2 += [f"{keys.perspective_flip_theta_series[frame_idx]:.5g}", f"{keys.perspective_flip_phi_series[frame_idx]:.5g}",
                      f"{keys.perspective_flip_gamma_series[frame_idx]:.5g}", f"{keys.perspective_flip_fv_series[frame_idx]:.5g}"]

    table.add_row(*rows1, *rows2)
    console.print(table)
