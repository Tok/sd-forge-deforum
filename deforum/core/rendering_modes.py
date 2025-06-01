"""
Rendering modes for different types of Deforum animations.
Note: FreeU and Kohya HR Fix functionality has been removed.
"""

import os
import time
import pathlib
import re
import numexpr
from modules.shared import opts, state
from .render import render_animation
from ..utils.color_constants import BOLD, BLUE, GREEN, PURPLE, RESET_COLOR
from .seed import next_seed
from ..media.video_audio_pipeline import vid2frames, render_preview
from ..prompt import interpolate_prompts
from .generate import generate
from .keyframe_animation import DeformAnimKeys
from .parseq_adapter import ParseqAdapter
from .save_images import save_image
from .settings import save_settings_from_animation_run

def render_input_video(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    """
    Render animation using video input frames.
    Note: FreeU and Kohya HR Fix functionality has been removed.
    """
    # create a folder for the video input frames
    video_in_frame_path = os.path.join(args.outdir, 'inputframes') 
    os.makedirs(video_in_frame_path, exist_ok=True)
    
    # save the video frames from input video
    print(f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {video_in_frame_path}...")
    try:
        for f in os.listdir(video_in_frame_path):
            os.remove(os.path.join(video_in_frame_path, f))
    except:
        pass
    
    vf = f'select=not(mod(n\\,{anim_args.extract_nth_frame}))'
    if os.path.exists(args.video_init_path):
        try:
            subprocess.run(['ffmpeg', '-i', f'{args.video_init_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '1', '-pix_fmt', 'rgb24', f'{video_in_frame_path}/%04d.jpg', '-y'], cwd=video_in_frame_path, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error exporting frames: {e}")
            return
    else:
        print(f"Video not found: {args.video_init_path}")
        return
    
    # determine max frames from length of input frames
    anim_args.max_frames = len([f for f in os.listdir(video_in_frame_path) if f.endswith('.jpg')])
    args.use_init = True
    print(f"Loading {anim_args.max_frames} input frames from {video_in_frame_path} and saving video frames to {args.outdir}")
    
    render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root)


def render_animation_with_video_mask(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    """
    Render animation with video mask support.
    Note: FreeU and Kohya HR Fix functionality has been removed.
    """
    # create a folder for the video mask frames
    mask_in_frame_path = os.path.join(args.outdir, 'maskframes') 
    os.makedirs(mask_in_frame_path, exist_ok=True)
    
    # save the video frames from mask video
    print(f"Exporting Video Mask Frames (1 every {anim_args.extract_nth_frame}) frames to {mask_in_frame_path}...")
    try:
        for f in os.listdir(mask_in_frame_path):
            os.remove(os.path.join(mask_in_frame_path, f))
    except:
        pass
    
    vf = f'select=not(mod(n\\,{anim_args.extract_nth_frame}))'
    if os.path.exists(anim_args.video_mask_path):
        try:
            subprocess.run(['ffmpeg', '-i', f'{anim_args.video_mask_path}', '-vf', f'{vf}', '-vsync', 'vfr', '-q:v', '1', '-pix_fmt', 'rgb24', f'{mask_in_frame_path}/%04d.jpg', '-y'], cwd=mask_in_frame_path, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error exporting mask frames: {e}")
            return
    else:
        print(f"Video mask not found: {anim_args.video_mask_path}")
        return
    
    render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root)


def render_interpolation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):
    """
    Render interpolation between keyframes.
    Note: FreeU and Kohya HR Fix functionality has been removed.
    """
    # initialise Parseq adapter
    parseq_adapter = ParseqAdapter(parseq_args, anim_args, video_args, controlnet_args, loop_args)

    # expand key frame strings to values
    keys = DeformAnimKeys(anim_args, args.seed) if not parseq_adapter.use_parseq else parseq_adapter.anim_keys

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving interpolation frames to {args.outdir}")

    # save settings.txt file for the current run
    save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root)

    # Interpolation mode - render images for each keyframe
    frame_idx = 0
    
    # Get keyframes from prompts
    keyframes = []
    for i, prompt in root.animation_prompts.items():
        if str(i).isdigit():
            keyframes.append(int(i))
        else:
            keyframes.append(int(numexpr.evaluate(i)))
    
    keyframes.sort()
    
    # Generate images for each keyframe
    for keyframe in keyframes:
        if keyframe >= anim_args.max_frames:
            break
            
        print(f"Generating keyframe {keyframe}")
        
        # Set frame-specific parameters
        args.prompt = root.animation_prompts.get(str(keyframe), args.prompt)
        args.seed = int(keys.seed_schedule_series[keyframe]) if args.seed_behavior == 'schedule' else args.seed
        
        # Generate image
        image = generate(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, keyframe)
        
        if image is None:
            break
            
        # Save image
        filename = f"{root.timestring}_{keyframe:09}.png"
        image.save(os.path.join(args.outdir, filename))
        
        frame_idx += 1

def get_parsed_value(value, frame_idx, max_f):
    pattern = r'`.*?`'
    regex = re.compile(pattern)
    parsed_value = value
    for match in regex.finditer(parsed_value):
        matched_string = match.group(0)
        parsed_string = matched_string.replace('t', f'{frame_idx}').replace("max_f" , f"{max_f}").replace('`','')
        value = numexpr.evaluate(parsed_string)
        parsed_value = parsed_value.replace(matched_string, str(value))
    return parsed_value

def render_interpolation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, root):

    # use parseq if manifest is provided
    parseq_adapter = ParseqAdapter(parseq_args, anim_args, video_args, controlnet_args, loop_args)

    # expand key frame strings to values
    keys = DeformAnimKeys(anim_args, args.seed) if not parseq_adapter.use_parseq else parseq_adapter.anim_keys

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving interpolation animation frames to {args.outdir}")

    # save settings.txt file for the current run
    save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, video_args, root)
        
    # Compute interpolated prompts
    if parseq_adapter.manages_prompts():
        print("Parseq prompts are assumed to already be interpolated - not doing any additional prompt interpolation")
        prompt_series = keys.prompts
    else: 
        print("Generating interpolated prompts for all frames")
        prompt_series = interpolate_prompts(root.animation_prompts, anim_args.max_frames)
    
    state.job_count = anim_args.max_frames
    frame_idx = 0
    last_preview_frame = 0
    # INTERPOLATION MODE
    while frame_idx < anim_args.max_frames:
        # print data to cli
        prompt_to_print = get_parsed_value(prompt_series[frame_idx].strip(), frame_idx, anim_args.max_frames)
        
        if prompt_to_print.endswith("--neg"):
            prompt_to_print = prompt_to_print[:-5]

        print(f"{BLUE}Interpolation frame: {RESET_COLOR}"
              f"{BOLD}{frame_idx}{RESET_COLOR}/{anim_args.max_frames}  ")
        print(f"{GREEN}Seed: {RESET_COLOR}{args.seed}")
        print(f"{PURPLE}Prompt: {RESET_COLOR}{prompt_to_print}")

        state.job = f"frame {frame_idx + 1}/{anim_args.max_frames}"
        state.job_no = frame_idx + 1
        
        if state.interrupted:
            break
        if state.skipped:
            print("\n** PAUSED **")
            state.skipped = False
            while not state.skipped:
                time.sleep(0.1)
            print("** RESUMING **")
        
        # grab inputs for current frame generation
        args.prompt = prompt_to_print
        args.cfg_scale = keys.cfg_scale_schedule_series[frame_idx]
        args.distilled_cfg_scale = keys.distilled_cfg_scale_schedule_series[frame_idx]

        scheduled_sampler_name = keys.sampler_schedule_series[frame_idx].casefold() if anim_args.enable_sampler_scheduling and keys.sampler_schedule_series[frame_idx] is not None else None
        scheduled_scheduler_name = keys.scheduler_schedule_series[frame_idx].casefold() if anim_args.enable_scheduler_scheduling and keys.scheduler_schedule_series[frame_idx] is not None else None
        args.steps = int(keys.steps_schedule_series[frame_idx]) if anim_args.enable_steps_scheduling and keys.steps_schedule_series[frame_idx] is not None else args.steps
        scheduled_clipskip = int(keys.clipskip_schedule_series[frame_idx]) if anim_args.enable_clipskip_scheduling and keys.clipskip_schedule_series[frame_idx] is not None else None
        args.checkpoint = keys.checkpoint_schedule_series[frame_idx] if anim_args.enable_checkpoint_scheduling else None
        if anim_args.enable_subseed_scheduling:
            root.subseed = int(keys.subseed_schedule_series[frame_idx])
            root.subseed_strength = keys.subseed_strength_schedule_series[frame_idx]
        else:
            root.subseed, root.subseed_strength = keys.subseed_schedule_series[frame_idx], keys.subseed_strength_schedule_series[frame_idx]
        if parseq_adapter.manages_seed():
            anim_args.enable_subseed_scheduling = True
            root.subseed, root.subseed_strength = int(keys.subseed_schedule_series[frame_idx]), keys.subseed_strength_schedule_series[frame_idx]
        args.seed = int(keys.seed_schedule_series[frame_idx]) if (args.seed_behavior == 'schedule' or parseq_adapter.manages_seed()) else args.seed
        opts.data["CLIP_stop_at_last_layers"] = scheduled_clipskip if scheduled_clipskip is not None else opts.data["CLIP_stop_at_last_layers"]

        image = generate(args, keys, anim_args, loop_args, controlnet_args, root, parseq_adapter, frame_idx, scheduled_sampler_name, scheduled_scheduler_name)
        filename = f"{root.timestring}_{frame_idx:09}.png"

        save_image(image, 'PIL', filename, args, video_args, root)

        state.current_image = image
        
        if args.seed_behavior != 'schedule':
            args.seed = next_seed(args, root)

        last_preview_frame = render_preview(args, anim_args, video_args, root, frame_idx, last_preview_frame)

        frame_idx += 1

        
