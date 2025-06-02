"""
Frame Interpolation Utilities

Pure utility functions for frame interpolation processing.
All functions are side-effect free and follow functional programming principles.
"""

from pathlib import Path


def extract_rife_name(string: str) -> str:
    """
    Pure function: RIFE version string -> model folder name
    Gets 'RIFE v4.3', returns: 'RIFE43'
    """
    parts = string.split()
    if len(parts) != 2 or parts[0] != "RIFE" or (parts[1][0] != "v" or not parts[1][1:].replace('.','').isdigit()):
        raise ValueError("Input string should contain exactly 2 words, first word should be 'RIFE' and second word should start with 'v' followed by 2 numbers")
    return "RIFE" + parts[1][1:].replace('.', '')


def clean_folder_name(string: str) -> str:
    """
    Pure function: filename -> legal folder name
    Converts a filename to a legal linux/windows folder name
    """
    illegal_chars = "/\\<>:\"|?*.,\" "
    translation_table = str.maketrans(illegal_chars, "_" * len(illegal_chars))
    return string.translate(translation_table)


def set_interp_out_fps(interp_x, slow_x_enabled: bool, slom_x, in_vid_fps) -> str:
    """
    Pure function: interpolation settings -> output FPS
    Calculate expected output FPS based on interpolation and slow motion settings
    """
    if interp_x == 'Disabled' or in_vid_fps in ('---', None, '', 'None'):
        return '---'

    fps = float(in_vid_fps) * int(interp_x)
    if slow_x_enabled:
        fps /= int(slom_x)
    return int(fps) if fps.is_integer() else fps


def calculate_frames_to_add(total_frames: int, interp_x: int) -> int:
    """
    Pure function: total frames + interpolation factor -> frames to add
    Get FILM number of frames to add after each pic from total frames in interp_x values
    """
    frames_to_add = (total_frames * interp_x - total_frames) / (total_frames - 1)
    return int(round(frames_to_add))


def generate_unique_output_dir(base_dir: str, folder_name: str) -> str:
    """
    Pure function: base directory + folder name -> unique output directory
    Generate a unique output directory path by appending numbers if needed
    """
    outdir_no_tmp = f"{base_dir}/frame-interpolation/{folder_name}"
    i = 1
    while Path(outdir_no_tmp).exists():
        outdir_no_tmp = f"{base_dir}/frame-interpolation/{folder_name}_{i}"
        i += 1
    return outdir_no_tmp


def build_film_paths(raw_output_imgs_path: str, img_batch_id: str, orig_vid_name: str, x_am: int) -> dict:
    """
    Pure function: path inputs -> FILM processing paths
    Build all necessary paths for FILM interpolation processing
    """
    parent_folder = Path(raw_output_imgs_path).parent
    
    if orig_vid_name is not None:
        interp_vid_path = parent_folder / f"{orig_vid_name}_FILM_x{x_am}"
    else:
        interp_vid_path = Path(raw_output_imgs_path) / f"{img_batch_id}_FILM_x{x_am}"
    
    output_interp_imgs_folder = Path(raw_output_imgs_path) / 'interpolated_frames_film'
    
    if orig_vid_name is not None:
        custom_interp_path = f"{output_interp_imgs_folder}_{orig_vid_name}"
    else:
        custom_interp_path = f"{output_interp_imgs_folder}_{img_batch_id}"
    
    img_path_for_ffmpeg = Path(custom_interp_path) / "frame_%09d.png"
    temp_convert_raw_png_path = Path(raw_output_imgs_path) / "tmp_film_folder"
    
    return {
        'interp_vid_path': str(interp_vid_path),
        'custom_interp_path': custom_interp_path,
        'img_path_for_ffmpeg': str(img_path_for_ffmpeg),
        'temp_convert_raw_png_path': str(temp_convert_raw_png_path)
    }


def validate_interpolation_settings(frame_interpolation_x_amount: int, 
                                   frame_interpolation_engine: str) -> tuple[bool, str]:
    """
    Pure function: interpolation settings -> validation result
    Validate interpolation parameters and return success status with error message
    """
    if frame_interpolation_engine == 'None':
        return True, "No interpolation requested"
    
    if frame_interpolation_engine.startswith("RIFE"):
        if frame_interpolation_x_amount not in range(2, 11):
            return False, "frame_interpolation_x_amount must be between 2x and 10x"
    
    return True, ""


def determine_uhd_mode(resolution: tuple) -> bool:
    """
    Pure function: resolution -> UHD mode decision
    Set UHD to True if resolution is 2K or higher
    """
    if resolution:
        return resolution[0] >= 2048 and resolution[1] >= 2048
    else:
        return False


def calculate_final_fps(orig_vid_fps: float, 
                       frame_interpolation_x_amount: int,
                       frame_interpolation_slow_mo_enabled: bool,
                       frame_interpolation_slow_mo_amount: int,
                       is_random_pics_run: bool = False) -> float:
    """
    Pure function: FPS settings -> final FPS calculation
    Calculate the final FPS for interpolated video based on all settings
    """
    fps = float(orig_vid_fps) * (1 if is_random_pics_run else frame_interpolation_x_amount)
    fps /= int(frame_interpolation_slow_mo_amount) if frame_interpolation_slow_mo_enabled and not is_random_pics_run else 1
    return fps


def should_disable_audio(real_audio_track, frame_interpolation_slow_mo_enabled: bool) -> bool:
    """
    Pure function: audio settings -> audio disable decision
    Determine if audio should be disabled based on slow motion settings
    """
    return real_audio_track is not None and frame_interpolation_slow_mo_enabled


def should_disable_subtitles(srt_path, frame_interpolation_slow_mo_enabled: bool) -> bool:
    """
    Pure function: subtitle settings -> subtitle disable decision  
    Determine if subtitles should be disabled based on slow motion settings
    """
    return srt_path is not None and frame_interpolation_slow_mo_enabled


def get_film_model_info(model_name: str) -> dict:
    """
    Pure function: model name -> model information
    Get FILM model download URL and hash information
    """
    if model_name == 'film_net_fp16.pt':
        return {
            'download_url': 'https://github.com/hithereai/frame-interpolation-pytorch/releases/download/film_net_fp16.pt/film_net_fp16.pt',
            'hash': '0a823815b111488ac2b7dd7fe6acdd25d35a22b703e8253587764cf1ee3f8f93676d24154d9536d2ce5bc3b2f102fb36dfe0ca230dfbe289d5cd7bde5a34ec12'
        }
    else:
        raise ValueError(f"Unknown FILM model: {model_name}")


def extract_pic_path_list(pic_list) -> list:
    """
    Pure function: Gradio file list -> path list
    Extract file paths from Gradio file upload list
    """
    return [pic.name for pic in pic_list]


def determine_soundtrack_mode(audio_track) -> str:
    """
    Pure function: audio track -> soundtrack mode
    Determine the soundtrack mode for ffmpeg based on audio track availability
    """
    return 'File' if audio_track is not None else 'None'


def should_keep_frames(fps: float, keep_imgs: bool, exception_raised: bool) -> bool:
    """
    Pure function: FPS + settings -> frame retention decision
    Determine if interpolated frames should be kept based on FPS and settings
    """
    if keep_imgs or exception_raised:
        return True
    
    # Keep frames automatically if output video FPS is above 450
    return fps > 450
