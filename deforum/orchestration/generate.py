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

from PIL import Image
import os
import math
import json
import itertools
import requests
import numexpr
import numpy as np
from modules import processing, sd_models
from modules.shared import sd_model, state, cmd_opts
from deforum.integrations.controlnet.legacy_controlnet_stubs import is_controlnet_enabled, get_controlnet_script_args
from deforum.utils.generation.prompts import split_weighted_subprompts
from deforum.media.load_images import load_img, prepare_mask, check_mask_for_errors
from deforum.pipeline.webui_sd_pipeline import get_webui_sd_pipeline
from deforum.utils.ui.console import console
from deforum.config.defaults import get_samplers_list, get_schedulers_list
from deforum.utils.generation.prompts import check_is_number
from deforum.utils.system.opts_overrider import A1111OptionsOverrider
from deforum.utils.system.output_filter import suppress_forge_output
import cv2
import numpy as np
from types import SimpleNamespace

from deforum.core.deforum_scripts_overrides import add_forge_script_to_deforum_run, initialise_forge_scripts

from deforum.utils.general import debug_print

# Import pure functions from refactored utils module
from deforum.utils.validation.validators import is_valid_json as isJson
from deforum.utils.functional import pairwise as pairwise_repl
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


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

# ANSI escape codes for background color
_RESET_BG = "\033[0m"

# Removed _handle_forge_message - no longer needed with simplified dashboard
# (was causing RecursionError due to logger.info() being intercepted)

def _get_mean_color(image):
    """Calculate mean RGB color of an image.

    Args:
        image: PIL Image

    Returns:
        Tuple of (r, g, b) with values 0-255
    """
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Handle different image formats
    if len(img_array.shape) == 2:  # Grayscale
        mean_val = int(np.mean(img_array))
        return (mean_val, mean_val, mean_val)
    elif len(img_array.shape) == 3:  # RGB or RGBA
        # Only use RGB channels
        rgb_array = img_array[:, :, :3]
        mean_rgb = np.mean(rgb_array, axis=(0, 1))
        return tuple(int(v) for v in mean_rgb)
    else:
        # Fallback for unexpected format
        return (128, 128, 128)

def _rgb_to_ansi_color_block(rgb):
    """Convert RGB tuple to ANSI color block (foreground + background same color).

    Args:
        rgb: Tuple of (r, g, b) with values 0-255

    Returns:
        ANSI escape sequence for solid color block (both fg and bg set to same color)
    """
    r, g, b = rgb
    # Set both foreground (38;2) and background (48;2) to same color for solid block
    return f"\033[38;2;{r};{g};{b}m\033[48;2;{r};{g};{b}m"

def _get_movement_indicators(anim_args, keys, frame_idx):
    """Generate Unicode movement indicators for current frame with dominant movement names.

    Translation indicators:
        ← = left, → = right
        ↑ = up, ↓ = down
        + = forward/zoom (toward camera), - = backward/zoom (away from camera)
        Diagonals: ↖ ↗ ↘ ↙ (when X+Y movement combined)

    Rotation indicators (pitch/yaw/roll):
        ⤴ = pitch up, ⤵ = pitch down
        ⤺ = yaw left, ⤻ = yaw right
        ↺ = roll counter-clockwise, ↻ = roll clockwise

    Args:
        anim_args: Animation arguments
        keys: Animation keys with scheduled values
        frame_idx: Current frame index

    Returns:
        String with Unicode movement indicators and dominant movement name, or empty string if no movement
    """
    if anim_args.animation_mode in ['Video Input', 'Interpolation']:
        return ""  # No transform data available

    movements = {}  # Track movements with their magnitudes
    threshold = 0.01  # Minimum value to show indicator

    # Translation indicators
    tr_x = keys.translation_x_series[frame_idx]
    tr_y = keys.translation_y_series[frame_idx]

    # Check for diagonal movement (X+Y combined)
    has_x = abs(tr_x) > threshold
    has_y = abs(tr_y) > threshold

    if has_x and has_y:
        # Diagonal movement - combine into single indicator
        magnitude = (tr_x**2 + tr_y**2)**0.5  # Euclidean distance
        if tr_x > 0 and tr_y < 0:
            movements['tr_xy'] = (magnitude, "↗ (up-right)")
        elif tr_x < 0 and tr_y < 0:
            movements['tr_xy'] = (magnitude, "↖ (up-left)")
        elif tr_x > 0 and tr_y > 0:
            movements['tr_xy'] = (magnitude, "↘ (down-right)")
        else:  # tr_x < 0 and tr_y > 0
            movements['tr_xy'] = (magnitude, "↙ (down-left)")
    else:
        # Single-axis movement
        if has_x:
            movements['tr_x'] = (abs(tr_x), "→ (right)" if tr_x > 0 else "← (left)")
        if has_y:
            movements['tr_y'] = (abs(tr_y), "↓ (down)" if tr_y > 0 else "↑ (up)")

    # 3D specific: Z translation and rotations
    if anim_args.animation_mode == '3D':
        tr_z = keys.translation_z_series[frame_idx]
        if abs(tr_z) > threshold:
            movements['tr_z'] = (abs(tr_z), "+ (zoom)" if tr_z > 0 else "- (zoom)")

        # Rotations (using better Unicode symbols)
        rot_x = keys.rotation_3d_x_series[frame_idx]  # Pitch
        rot_y = keys.rotation_3d_y_series[frame_idx]  # Yaw
        rot_z = keys.rotation_3d_z_series[frame_idx]  # Roll

        if abs(rot_x) > threshold:
            movements['rot_x'] = (abs(rot_x), "⤵ (pitch down)" if rot_x > 0 else "⤴ (pitch up)")
        if abs(rot_y) > threshold:
            movements['rot_y'] = (abs(rot_y), "⤻ (yaw right)" if rot_y > 0 else "⤺ (yaw left)")
        if abs(rot_z) > threshold:
            movements['rot_z'] = (abs(rot_z), "↻ (roll cw)" if rot_z > 0 else "↺ (roll ccw)")

    if movements:
        # Sort by magnitude to get dominant movement first
        sorted_movements = sorted(movements.items(), key=lambda x: x[1][0], reverse=True)
        indicators = [indicator for _, (_, indicator) in sorted_movements]
        return "Move: " + " ".join(indicators)
    else:
        return ""

def _update_dashboard(args, anim_args, p, keys, frame_idx, previous_image, dashboard):
    """Update dashboard with current frame information.

    Args:
        args: Generation arguments
        anim_args: Animation arguments
        p: Processing pipeline object
        keys: Animation keys with scheduled values
        frame_idx: Current frame index
        previous_image: PIL Image of previous frame (optional)
        dashboard: RenderDashboard instance
    """
    # Update frame info
    dashboard.frame_info['current'] = frame_idx
    dashboard.frame_info['total'] = anim_args.max_frames
    dashboard.frame_info['type'] = 'KEYFRAME'  # Will be updated by caller if cadence
    dashboard.frame_info['seed'] = p.seed
    dashboard.frame_info['movement'] = _get_movement_indicators(anim_args, keys, frame_idx)
    dashboard.frame_info['prompt'] = p.prompt if isinstance(p.prompt, str) else p.prompt[0] if p.prompt else ""

    # Update color and image if previous image available
    if previous_image is not None:
        dashboard.frame_info['color_rgb'] = _get_mean_color(previous_image)
        dashboard.last_frame_image = previous_image  # Store for ASCII preview (printed before next frame)

    # Update table data
    total_steps = p.steps
    if p.denoising_strength is not None and anim_args.animation_mode != 'Interpolation':
        actual_steps = int(total_steps * (1.0 - p.denoising_strength))
        dashboard.table_data['steps'] = f"{actual_steps}/{total_steps}"
    else:
        dashboard.table_data['steps'] = str(total_steps)

    dashboard.table_data['cfg'] = str(p.cfg_scale)
    dashboard.table_data['dist_cfg'] = str(p.distilled_cfg_scale)
    dashboard.table_data['denoise'] = f"{p.denoising_strength:.5g}" if p.denoising_strength is not None else "None"

    # Transform parameters (mode-specific)
    if anim_args.animation_mode not in ['Video Input', 'Interpolation']:
        dashboard.table_data['tr_x'] = f"{keys.translation_x_series[frame_idx]:.5g}"
        dashboard.table_data['tr_y'] = f"{keys.translation_y_series[frame_idx]:.5g}"

        if anim_args.animation_mode == '3D':
            dashboard.table_data['tr_z'] = f"{keys.translation_z_series[frame_idx]:.5g}"
            dashboard.table_data['ro_x'] = f"{keys.rotation_3d_x_series[frame_idx]:.5g}"
            dashboard.table_data['ro_y'] = f"{keys.rotation_3d_y_series[frame_idx]:.5g}"
            dashboard.table_data['ro_z'] = f"{keys.rotation_3d_z_series[frame_idx]:.5g}"

    # Trigger dashboard update
    dashboard.update()

def print_combined_table(args, anim_args, p, keys, frame_idx, previous_image=None, dashboard=None):
    """Print comprehensive frame parameters table OR update dashboard.

    Routes to dashboard if enabled and dashboard instance provided,
    otherwise prints to console in classic format.

    Displays all relevant parameters for the current frame including:
    - Seed with color indicator and movement arrows
    - Mean color of previous frame (if available) shown as colored block ██
    - Movement indicators: < > (left/right), ^ v (up/down), + - (forward/back)
    - Rotation indicators: ↑ ↓ (pitch), ← → (yaw), ↶ ↷ (roll)
    - Prompts (printed separately above table)
    - Sampling parameters (steps, CFG, denoise)
    - Optional schedules (subseed, sampler, scheduler, checkpoint)
    - Transform parameters (translation, rotation for 3D)

    Args:
        args: Generation arguments
        anim_args: Animation arguments
        p: Processing pipeline object
        keys: Animation keys with scheduled values
        frame_idx: Current frame index
        previous_image: PIL Image of previous frame (optional, for color display)
        dashboard: Optional RenderDashboard instance (routes to dashboard if provided)
    """
    from rich.table import Table
    from rich import box
    from deforum.utils.model_detection import is_flux_model, is_lumina_model
    from deforum.rendering import options as opt_utils

    # If dashboard provided, update it (but still print table to log)
    if dashboard is not None and opt_utils.is_dashboard_enabled():
        _update_dashboard(args, anim_args, p, keys, frame_idx, previous_image, dashboard)
        # Continue to print table to scrolling log below

    # Detect if model ignores negative prompts
    model_ignores_negative = is_flux_model() or is_lumina_model()

    # ========================================================================
    # Print seed, color, and movement info BEFORE table
    # ========================================================================
    seed_info = f"Seed: {p.seed}"

    # Add color indicator if we have a previous image
    if previous_image is not None:
        mean_color = _get_mean_color(previous_image)
        color_block = _rgb_to_ansi_color_block(mean_color)
        # Use █ with both fg and bg set to same color for solid block
        seed_info += f", Color: {color_block}██{_RESET_BG}"

    # Add movement indicators
    movement = _get_movement_indicators(anim_args, keys, frame_idx)
    if movement:
        seed_info += ", " + movement

    logger.info(seed_info)

    # ========================================================================
    # Print prompts
    # ========================================================================
    prompt_to_print = p.prompt if isinstance(p.prompt, str) else p.prompt[0] if p.prompt else ""
    logger.info(f"Prompt: {prompt_to_print}")

    # Only print negative prompt if model uses it and it's not empty
    if not model_ignores_negative:
        neg_prompt = p.negative_prompt if isinstance(p.negative_prompt, str) else p.negative_prompt[0] if p.negative_prompt else ""
        if neg_prompt and neg_prompt.strip():
            logger.info(f"Neg Prompt: {neg_prompt}")

    # ========================================================================
    # Create table with sampling and transform parameters
    # ========================================================================
    table = Table(padding=0, box=box.ROUNDED, show_header=True)

    # ========================================================================
    # Sampling Parameters
    # ========================================================================
    columns = []
    values = []

    # Core parameters
    columns.extend(["Steps", "CFG", "Dist. CFG"])

    # Calculate actual steps used based on denoise strength
    total_steps = p.steps
    if p.denoising_strength is not None and anim_args.animation_mode != 'Interpolation':
        actual_steps = int(total_steps * (1.0 - p.denoising_strength))
        steps_display = f"{actual_steps}/{total_steps}"
    else:
        steps_display = str(total_steps)

    values.extend([steps_display, str(p.cfg_scale), str(p.distilled_cfg_scale)])

    # Denoise (skip for Interpolation mode)
    if anim_args.animation_mode != 'Interpolation':
        columns.append("Denoise")
        values.append(f"{p.denoising_strength:.5g}" if p.denoising_strength is not None else "None")

    # Optional schedules
    if anim_args.enable_subseed_scheduling:
        columns.extend(["Subseed", "Subs. str"])
        values.extend([str(p.subseed), f"{p.subseed_strength:.5g}"])

    if anim_args.enable_sampler_scheduling:
        columns.append("Sampler")
        values.append(p.sampler_name)

    if anim_args.enable_scheduler_scheduling:
        columns.append("Scheduler")
        values.append(p.scheduler_name)

    if anim_args.enable_checkpoint_scheduling:
        columns.append("Checkpoint")
        values.append(str(args.checkpoint))

    # ========================================================================
    # Transform Parameters (mode-specific)
    # ========================================================================
    if anim_args.animation_mode not in ['Video Input', 'Interpolation']:
        # 2D specific transforms
        if anim_args.animation_mode == '2D':
            columns.extend(["Angle", "Zoom", "Tr C X", "Tr C Y"])
            values.extend([
                f"{keys.angle_series[frame_idx]:.5g}",
                f"{keys.zoom_series[frame_idx]:.5g}",
                f"{keys.transform_center_x_series[frame_idx]:.5g}",
                f"{keys.transform_center_y_series[frame_idx]:.5g}"
            ])

        # Common X/Y translation
        columns.extend(["Tr X", "Tr Y"])
        values.extend([
            f"{keys.translation_x_series[frame_idx]:.5g}",
            f"{keys.translation_y_series[frame_idx]:.5g}"
        ])

        # 3D specific transforms (removed Aspect Ratio as it's rarely scheduled)
        if anim_args.animation_mode == '3D':
            columns.extend(["Tr Z", "Ro X", "Ro Y", "Ro Z"])
            values.extend([
                f"{keys.translation_z_series[frame_idx]:.5g}",
                f"{keys.rotation_3d_x_series[frame_idx]:.5g}",
                f"{keys.rotation_3d_y_series[frame_idx]:.5g}",
                f"{keys.rotation_3d_z_series[frame_idx]:.5g}"
            ])

        # Perspective flip (if enabled)
        if anim_args.enable_perspective_flip:
            columns.extend(["Pf T", "Pf P", "Pf G", "Pf F"])
            values.extend([
                f"{keys.perspective_flip_theta_series[frame_idx]:.5g}",
                f"{keys.perspective_flip_phi_series[frame_idx]:.5g}",
                f"{keys.perspective_flip_gamma_series[frame_idx]:.5g}",
                f"{keys.perspective_flip_fv_series[frame_idx]:.5g}"
            ])

    # ========================================================================
    # Add columns and single data row
    # ========================================================================
    for col in columns:
        table.add_column(col, justify="center", no_wrap=True)

    table.add_row(*values)
    console.print(table)

def generate(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter,  frame=0, sampler_name=None, scheduler_name=None):
    if state.interrupted:
        return None

    if args.reroll_blank_frames == 'ignore':
        return generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)

    image, caught_vae_exception = generate_with_nans_check(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)

    if caught_vae_exception or not image.getbbox():
        patience = args.reroll_patience
        logger.info("Blank frame detected! If you don't have the NSFW filter enabled, this may be due to a glitch!")
        if args.reroll_blank_frames == 'reroll':
            while caught_vae_exception or not image.getbbox():
                logger.info("Rerolling with +1 seed...")
                args.seed += 1
                image, caught_vae_exception = generate_with_nans_check(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)
                patience -= 1
                if patience == 0:
                    logger.error("Rerolling with +1 seed failed for 10 iterations! Try setting webui's precision to 'full' and if it fails, please report this to the devs! Interrupting...")
                    state.interrupted = True
                    state.assign_current_image(image)
                    return None
        elif args.reroll_blank_frames == 'interrupt':
            logger.info("Interrupting to save your eyes...")
            state.interrupted = True
            state.assign_current_image(image)
            return None
    return image

def generate_with_nans_check(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame=0, sampler_name=None, scheduler_name=None):
    if cmd_opts.disable_nan_check:
        image = generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)
    else:
        try:
            image = generate_inner(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame, sampler_name, scheduler_name)
        except Exception as e:
            if "A tensor with all NaNs was produced in VAE." in repr(e):
                logger.info(e)
                return None, True
            else:
                raise e
    return image, False

def generate_inner(args, keys, anim_args, loop_args, controlnet_args,
                   root, parseq_adapter, frame=0, sampler_name=None, scheduler_name=None):
    # Setup the pipeline
    p = get_webui_sd_pipeline(args, root)
    p.prompt, p.negative_prompt = split_weighted_subprompts(args.prompt, frame, anim_args.max_frames)

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        args.strength = 0
    processed = None
    mask_image = None
    init_image = None
    image_init0 = None
    image_init0_box = None

    if loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']:

        debug_print(f"Looper: use_looper={loop_args.use_looper}, imageStrength={loop_args.imageStrength}, blendFactorMax={loop_args.blendFactorMax}, blendFactorSlope={loop_args.blendFactorSlope}, tweeningFrames={loop_args.tweeningFrameSchedule}, colorCorrectionFactor={loop_args.colorCorrectionFactor}")
        args.strength = loop_args.imageStrength
        tweeningFrames = loop_args.tweeningFrameSchedule
        blendFactor = .07
        colorCorrectionFactor = loop_args.colorCorrectionFactor
        jsonImages = json.loads(loop_args.imagesToKeyframe)
        # find which image to show
        parsedImages = {}
        frameToChoose = 0
        max_f = anim_args.max_frames - 1

        for key, value in jsonImages.items():
            if check_is_number(key):  # default case 0:(1 + t %5), 30:(5-t%2)
                parsedImages[key] = value
            else:  # math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
                parsedImages[int(numexpr.evaluate(key))] = value

        framesToImageSwapOn = list(map(int, list(parsedImages.keys())))

        for swappingFrame in framesToImageSwapOn[1:]:
            frameToChoose += (frame >= int(swappingFrame))

        # find which frame to do our swapping on for tweening
        skipFrame = 25
        for fs, fe in pairwise_repl(framesToImageSwapOn):
            if fs <= frame <= fe:
                skipFrame = fe - fs

        if frame % skipFrame <= tweeningFrames:  # number of tweening frames
            blendFactor = loop_args.blendFactorMax - loop_args.blendFactorSlope * math.cos((frame % tweeningFrames) / (tweeningFrames / 2))
        init_image2, _ = load_img(list(jsonImages.values())[frameToChoose],
                                  None, # init_image_box not used in this case
                                  shape=(args.W, args.H),
                                  use_alpha_as_mask=args.use_alpha_as_mask)
        image_init0 = list(jsonImages.values())[0]

    else:  # they passed in a single init image
        image_init0 = args.init_image
        image_init0_box = args.init_image_box

    available_samplers = get_samplers_list()
    if sampler_name is not None:
        if sampler_name in available_samplers.keys():
            p.sampler_name = available_samplers[sampler_name]
        else:
            raise RuntimeError(f"Sampler name '{sampler_name}' is invalid. Please check the available sampler list in the 'Run' tab")

    available_schedulers = get_schedulers_list()
    if scheduler_name is not None:
        if scheduler_name in available_schedulers.keys():
            p.scheduler_name = available_schedulers[scheduler_name]
        else:
            raise RuntimeError(f"Scheduler name '{scheduler_name}' is invalid. Please check the available scheduler list in the 'Run' tab")

    if args.checkpoint is not None:
        info = sd_models.get_closet_checkpoint_match(args.checkpoint)
        if info is None:
            raise RuntimeError(f"Unknown checkpoint: {args.checkpoint}")
        sd_models.reload_model_weights(info=info)

    if root.init_sample is not None:
        # TODO: cleanup init_sample remains later
        img = root.init_sample
        init_image = img
        if loop_args.use_looper and isJson(loop_args.imagesToKeyframe) and anim_args.animation_mode in ['2D', '3D']:
            init_image = Image.blend(init_image, init_image2, blendFactor)
            correction_colors = Image.blend(init_image, init_image2, colorCorrectionFactor)
            p.color_corrections = [processing.setup_color_correction(correction_colors)]

    # this is the first pass
    elif (loop_args.use_looper and anim_args.animation_mode in ['2D', '3D']) or (args.use_init and ((args.init_image != None and args.init_image != '') or args.init_image_box != None)):
        init_image, mask_image = load_img(image_init0,  # initial init image
                                          image_init0_box,  # initial init image from box (if single init image is used, not json list)
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)

    else:

        if anim_args.animation_mode != 'Interpolation':
            logger.info(f"Not using an init image (doing pure txt2img)")
        
        if args.motion_preview_mode:
            state.assign_current_image(root.default_img)
            processed = SimpleNamespace(images = [root.default_img], info = "Generating motion preview...")
        else:
            p_txt = processing.StableDiffusionProcessingTxt2Img(
                sd_model=sd_model,
                outpath_samples=root.tmp_deforum_run_duplicated_folder,
                outpath_grids=root.tmp_deforum_run_duplicated_folder,
                prompt=p.prompt,
                styles=p.styles,
                negative_prompt=p.negative_prompt,
                seed=p.seed,
                subseed=p.subseed,
                subseed_strength=p.subseed_strength,
                seed_resize_from_h=p.seed_resize_from_h,
                seed_resize_from_w=p.seed_resize_from_w,
                sampler_name=p.sampler_name,
                scheduler=p.scheduler_name,
                batch_size=p.batch_size,
                n_iter=p.n_iter,
                steps=p.steps,
                cfg_scale=p.cfg_scale,
                distilled_cfg_scale=p.distilled_cfg_scale,
                width=p.width,
                height=p.height,
                restore_faces=p.restore_faces,
                tiling=p.tiling,
                enable_hr=False,
                denoising_strength=0,
            )

            # Get dashboard from root if available
            dashboard = getattr(root, 'dashboard', None)
            print_combined_table(args, anim_args, p_txt, keys, frame, root.init_sample, dashboard)  # print dynamic table to cli

            initialise_forge_scripts(p_txt)

            if is_controlnet_enabled(controlnet_args):
                cnet_args = get_controlnet_script_args(args, anim_args, controlnet_args, root, parseq_adapter, frame_idx=frame)
                add_forge_script_to_deforum_run(p_txt, "ControlNet", cnet_args)

            # Lumina compatibility: Ensure num_tokens is set before sampling
            try:
                from deforum.integrations.lumina import apply_lumina_patch_if_needed
                apply_lumina_patch_if_needed(p_txt)
            except Exception as e:
                logger.debug(f"Lumina patch not applied: {e}")

            # Note: Flux ControlNet V2 control samples are retrieved from global storage
            # by the patched KModel.apply_model during sampling (no injection needed here)

            with A1111OptionsOverrider({"control_net_detectedmap_dir" : os.path.join(args.outdir, "controlnet_detected_map")}):
                p_txt.scheduler = "Simple"  # FIXME provide
                # Suppress redundant Forge output (info already shown in Deforum's table)
                # No callback needed for simplified dashboard (would cause recursion)
                with suppress_forge_output():
                    processed = processing.process_images(p_txt)

            try:
                p_txt.close()
            except Exception as e:
                ...

    if processed is None:
        # Mask functions
        if args.use_mask:
            mask_image = args.mask_image
            mask = prepare_mask(args.mask_file if mask_image is None else mask_image,
                                (args.W, args.H),
                                args.mask_contrast_adjust,
                                args.mask_brightness_adjust)
            p.inpainting_mask_invert = args.invert_mask
            p.inpainting_fill = args.fill
            p.inpaint_full_res = args.full_res_mask
            p.inpaint_full_res_padding = args.full_res_mask_padding
            # prevent loaded mask from throwing errors in Image operations if completely black and crop and resize in webui pipeline
            # doing this after contrast and brightness adjustments to ensure that mask is not passed as black or blank
            mask = check_mask_for_errors(mask, args.invert_mask)
            root.noise_mask = mask
        else:
            mask = None

        assert not ((mask is not None and args.use_mask and args.overlay_mask) and (
                root.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

        p.init_images = [init_image]
        p.image_mask = mask
        p.image_cfg_scale = args.cfg_scale
        p.image_distilled_cfg_scale = args.distilled_cfg_scale

        # Get dashboard from root if available
        dashboard = getattr(root, 'dashboard', None)
        print_combined_table(args, anim_args, p, keys, frame, root.init_sample, dashboard)  # print dynamic table to cli

        if args.motion_preview_mode:
            processed = mock_process_images(args, p, init_image)
        else:
            initialise_forge_scripts(p)

            if is_controlnet_enabled(controlnet_args):
                cnet_args = get_controlnet_script_args(args, anim_args, controlnet_args, root, parseq_adapter, frame_idx=frame)
                add_forge_script_to_deforum_run(p, "ControlNet", cnet_args)

            # Lumina compatibility: Ensure num_tokens is set before sampling
            try:
                from deforum.integrations.lumina import apply_lumina_patch_if_needed
                apply_lumina_patch_if_needed(p)
            except Exception as e:
                logger.debug(f"Lumina patch not applied: {e}")

            with A1111OptionsOverrider({"control_net_detectedmap_dir" : os.path.join(args.outdir, "controlnet_detected_map")}):
                # Suppress redundant Forge output (info already shown in Deforum's table)
                # No callback needed for simplified dashboard (would cause recursion)
                with suppress_forge_output():
                    processed = processing.process_images(p)


    if root.initial_info is None:
        root.initial_info = processed.info

    if root.first_frame is None:
        root.first_frame = processed.images[0]

    results = processed.images[0]

    return results

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
    return SimpleNamespace(images = [image], info = "Generating motion preview...")
