from typing import List

from . import log_utils, opt_utils
from ..data import RenderData
from ..data.frame import DiffusionFrame


def create_all_subtitles_if_active(data: RenderData, diffusion_frames: List[DiffusionFrame]):
    # Doesn't check if the .srt file already exists, because all frames are recalculated again when resuming a run.
    # Since subtitle generation is not relevant for overall performance, we can just overwrite it and have it reflect
    # any changes that may have been made on the subtitle config parameters before the restart.
    if not opt_utils.is_generate_subtitles():
        log_utils.debug("Skipping subtitle creation (disabled in settings).")
        return

    subtitle_count = _write_subtitle_lines(data, diffusion_frames)
    _check_and_log_subtitle_count(data, diffusion_frames, subtitle_count)


def _write_subtitle_lines(data, diffusion_frames):
    log_utils.debug(f"Subtitle generation info: {opt_utils.generation_info_for_subtitles()}")
    subtitle_index = 0
    previous_diffusion_frame = None
    for diffusion_frame in diffusion_frames:
        if not diffusion_frame.has_tween_frames():  # 1st frame has no matching tween.
            diffusion_frame.write_frame_subtitle(data, subtitle_index)
            subtitle_index += 1
        for tween_frame in diffusion_frame.tweens:
            _write_tween_subtitle(data, subtitle_index, diffusion_frame, tween_frame, previous_diffusion_frame)
            subtitle_index += 1
        previous_diffusion_frame = diffusion_frame

    log_utils.info(f"Created {subtitle_index} subtitles.")
    return subtitle_index


def _write_tween_subtitle(data, subtitle_index, diffusion_frame, tween_frame, previous_diffusion_frame):
    # Each diffusion frame has 0 to many tweens. If there are any, the last tween in the collection
    # has the same index as the diffusion frame it belongs to (asserted on creation).
    # With both options available, subtitles are written using the method provided by the diffusion frame,
    # making it easy to hardcode and pass the correct value for 'is_cadence'
    # without doing any additional calculations or checks.
    is_last_tween = tween_frame.i == diffusion_frame.i
    if is_last_tween:
        diffusion_frame.write_frame_subtitle(data, subtitle_index)
    else:
        tween_frame.write_tween_frame_subtitle(data, previous_diffusion_frame)


def _check_and_log_subtitle_count(data, diffusion_frames, subtitle_count):
    max_frames = data.args.anim_args.max_frames
    if subtitle_count != max_frames:
        log_utils.warn(f"Subtitle count {subtitle_count} is different from max. frames {max_frames}.")
    calculated_count = 1 + sum(len(ks.tweens) for ks in diffusion_frames)
    if subtitle_count != calculated_count:
        log_utils.warn(f"Subtitle count {subtitle_count} is different from calculated count {calculated_count}.")
