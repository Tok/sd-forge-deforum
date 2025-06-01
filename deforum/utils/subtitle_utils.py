from typing import List

from . import log_utils, opt_utils
# Move these imports inside functions to avoid circular dependency
# from ..core.data import RenderData
# from ..core.data.frame import DiffusionFrame
from .subtitle_handler import calculate_frame_duration, frame_time, time_to_srt_format


def create_all_subtitles_if_active(data, diffusion_frames: List):
    """
    Create subtitles if enabled. Uses local imports to avoid circular dependency.
    Args:
        data: RenderData object
        diffusion_frames: List[DiffusionFrame] objects
    """
    # Doesn't check if the .srt file already exists, because all frames are recalculated again when resuming a run.
    # Since subtitle generation is not relevant for overall performance, we can just overwrite it and have it reflect
    # any changes that may have been made on the subtitle config parameters before the restart.
    if not opt_utils.is_generate_subtitles():
        log_utils.debug("Skipping subtitle creation (disabled in settings).")
        return

    subtitle_count = _write_subtitle_lines(data, diffusion_frames)
    _check_and_log_subtitle_count(data, diffusion_frames, subtitle_count)


def _write_subtitle_lines(data, diffusion_frames):
    write_interval = _interval(data, opt_utils.desired_subtitles_per_second())
    sub_info = opt_utils.generation_info_for_subtitles()
    frame_duration = calculate_frame_duration(data.fps())
    log_utils.info(f"Creating subtitles aiming for {opt_utils.desired_subtitles_per_second()} per second with "
                   f"{'a fuzzy' if opt_utils.is_always_write_keyframe_subs() else 'an'} interval of {write_interval} "
                   f"using frame duration {frame_duration:.3f} with info: {sub_info}")
    return _write_subtitle_lines_internal(data, diffusion_frames, frame_duration, write_interval,
                                          opt_utils.is_always_write_keyframe_subs())


def _write_subtitle_lines_internal(data, diffusion_frames, frame_duration, write_interval,
                                   is_always_write_keyframe_subs):
    subtitle_count = 0
    previous_diffusion_frame = None
    from_time = 0
    to_time = 0
    for diffusion_frame in diffusion_frames:
        if not diffusion_frame.has_tween_frames():  # 1st frame has no matching tween.
            to_time = frame_time(diffusion_frame.i, frame_duration)
            diffusion_frame.write_subtitle_from_to(data, subtitle_count, diffusion_frame.i, from_time, to_time)
            from_time = to_time
            subtitle_count += 1
        for tween_frame in diffusion_frame.tweens:
            to_time = frame_time(tween_frame.i, frame_duration)
            if is_always_write_keyframe_subs and tween_frame.is_last(diffusion_frame):
                diffusion_frame.write_subtitle_from_to(data, subtitle_count, tween_frame.i, from_time, to_time)
                from_time = to_time
                subtitle_count += 1
            elif _is_do_not_skip(subtitle_count, write_interval):
                _write_tween_subtitle(data, subtitle_count, tween_frame.i, diffusion_frame, tween_frame,
                                      previous_diffusion_frame, from_time, to_time)
                from_time = to_time
                subtitle_count += 1
        previous_diffusion_frame = diffusion_frame

    log_utils.info(f"Created {subtitle_count} subtitles. Last timestamp: {time_to_srt_format(to_time)}.")
    return subtitle_count


def _write_tween_subtitle(data, sub_i, frame_i, diffusion_frame, tween_frame, previous_diffusion_frame,
                          from_time, to_time):
    # Each diffusion frame has 0 to many tweens. If there are any, the last tween in the collection
    # has the same index as the diffusion frame it belongs to (asserted on creation).
    # With both options available, subtitles are written using the method provided by the diffusion frame,
    # making it easy to hardcode and pass the correct value for 'is_cadence'
    # without doing any additional calculations or checks.
    if tween_frame.is_last(diffusion_frame):
        diffusion_frame.write_subtitle_from_to(data, sub_i, frame_i, from_time, to_time)
    else:
        tween_frame.write_tween_subtitle_from_to(data, sub_i, previous_diffusion_frame, from_time, to_time)


def _check_and_log_subtitle_count(data, diffusion_frames, subtitle_count):
    if opt_utils.always_write_keyframe_subtitle() or not opt_utils.is_subtitles_per_second_same_as_animation_fps(data):
        return  # no check because subtitle frequency is expected to be irregular.
    max_frames = data.args.anim_args.max_frames
    if subtitle_count != max_frames:
        log_utils.warn(f"Subtitle count {subtitle_count} is different from max. frames {max_frames}.")
    calculated_count = 1 + sum(len(ks.tweens) for ks in diffusion_frames)
    if subtitle_count != calculated_count:
        log_utils.warn(f"Subtitle count {subtitle_count} is different from calculated count {calculated_count}.")


def _is_do_not_skip(subtitle_index, write_interval):
    return subtitle_index % write_interval == 0


def _interval(data, desired_subtitles_per_second):
    # How many tween frames to skip between writing tween subtitles
    # To keep subtitles in sync with actual keyframes, subtitle FPS is only applied to tween frames.
    # It's therefore not meant to be 100% accurate.
    animation_fps = data.fps()
    return int(animation_fps / desired_subtitles_per_second)
