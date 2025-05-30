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
import time
import pathlib
import re
import numexpr
from modules.shared import opts, state
from .render import render_animation
from .rendering.util.log_utils import BOLD, BLUE, GREEN, PURPLE, RESET_COLOR
from .seed import next_seed
from .video_audio_utilities import vid2frames, render_preview
from .prompt import interpolate_prompts
from .generate import generate
from .animation_key_frames import DeformAnimKeys
from .parseq_adapter import ParseqAdapter
from .save_images import save_image
from .settings import save_settings_from_animation_run

def render_input_video(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, root):
    # create a folder for the video input frames to live in
    video_in_frame_path = os.path.join(args.outdir, 'inputframes') 
    os.makedirs(video_in_frame_path, exist_ok=True)

    # save the video frames from input video
    print(f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {video_in_frame_path}...")
    vid2frames(video_path = anim_args.video_init_path, video_in_frame_path=video_in_frame_path, n=anim_args.extract_nth_frame, overwrite=anim_args.overwrite_extracted_frames, extract_from_frame=anim_args.extract_from_frame, extract_to_frame=anim_args.extract_to_frame)

    # determine max frames from length of input frames
    anim_args.max_frames = len([f for f in pathlib.Path(video_in_frame_path).glob('*.jpg')])
    args.use_init = True
    print(f"Loading {anim_args.max_frames} input frames from {video_in_frame_path} and saving video frames to {args.outdir}")

    if anim_args.use_mask_video:
        # create a folder for the mask video input frames to live in
        mask_in_frame_path = os.path.join(args.outdir, 'maskframes') 
        os.makedirs(mask_in_frame_path, exist_ok=True)

        # save the video frames from mask video
        print(f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {mask_in_frame_path}...")
        vid2frames(video_path=anim_args.video_mask_path,video_in_frame_path=mask_in_frame_path, n=anim_args.extract_nth_frame, overwrite=anim_args.overwrite_extracted_frames, extract_from_frame=anim_args.extract_from_frame, extract_to_frame=anim_args.extract_to_frame)
        max_mask_frames = len([f for f in pathlib.Path(mask_in_frame_path).glob('*.jpg')])

        # limit max frames if there are less frames in the video mask compared to input video
        if max_mask_frames < anim_args.max_frames :
            anim_args.max_mask_frames
            print ("Video mask contains less frames than init video, max frames limited to number of mask frames.")
        args.use_mask = True
        args.overlay_mask = True

    render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, root)

# Modified a copy of the above to allow using masking video with out a init video.
def render_animation_with_video_mask(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, root):
    # create a folder for the video input frames to live in
    mask_in_frame_path = os.path.join(args.outdir, 'maskframes') 
    os.makedirs(mask_in_frame_path, exist_ok=True)

    # save the video frames from mask video
    print(f"Exporting Video Frames (1 every {anim_args.extract_nth_frame}) frames to {mask_in_frame_path}...")
    vid2frames(video_path=anim_args.video_mask_path, video_in_frame_path=mask_in_frame_path, n=anim_args.extract_nth_frame, overwrite=anim_args.overwrite_extracted_frames, extract_from_frame=anim_args.extract_from_frame, extract_to_frame=anim_args.extract_to_frame)
    args.use_mask = True
    #args.overlay_mask = True

    # determine max frames from length of input frames
    anim_args.max_frames = len([f for f in pathlib.Path(mask_in_frame_path).glob('*.jpg')])
    #args.use_init = True
    print(f"Loading {anim_args.max_frames} input frames from {mask_in_frame_path} and saving video frames to {args.outdir}")

    render_animation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, root)

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

def render_interpolation(args, anim_args, video_args, parseq_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, root):

    # use parseq if manifest is provided
    parseq_adapter = ParseqAdapter(parseq_args, anim_args, video_args, controlnet_args, loop_args, freeu_args, kohya_hrfix_args)

    # expand key frame strings to values
    keys = DeformAnimKeys(anim_args) if not parseq_adapter.use_parseq else parseq_adapter.anim_keys

    # create output folder for the batch
    os.makedirs(args.outdir, exist_ok=True)
    print(f"Saving interpolation animation frames to {args.outdir}")

    # save settings.txt file for the current run
    save_settings_from_animation_run(args, anim_args, parseq_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, video_args, root)
        
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

        image = generate(args, keys, anim_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args,
                         root, parseq_adapter, frame_idx, scheduled_sampler_name, scheduled_scheduler_name)
        filename = f"{root.timestring}_{frame_idx:09}.png"

        save_image(image, 'PIL', filename, args, video_args, root)

        state.current_image = image
        
        if args.seed_behavior != 'schedule':
            args.seed = next_seed(args, root)

        last_preview_frame = render_preview(args, anim_args, video_args, root, frame_idx, last_preview_frame)

        frame_idx += 1

        
