"""
Upload Processing for Frame Interpolation

Handles video and image upload processing for frame interpolation.
All functions follow functional programming principles with side effect isolation.
"""

import os
from pathlib import Path
from typing import Optional, List
from .utilities import (
    clean_folder_name, generate_unique_output_dir, 
    extract_pic_path_list, set_interp_out_fps
)
from .main_pipeline import process_video_interpolation

# Conditional import for WebUI modules
try:
    from modules.shared import opts
    WEBUI_AVAILABLE = True
except ImportError:
    WEBUI_AVAILABLE = False
    # Fallback opts object
    class FallbackOpts:
        def __init__(self):
            self.data = {}
            self.outdir_samples = "./outputs"
    opts = FallbackOpts()


def gradio_f_interp_get_fps_and_fcount(vid_path, interp_x, slow_x_enabled: bool, slom_x) -> tuple:
    """
    Pure function: video info + settings -> FPS and frame count for Gradio UI
    Get uploaded video frame count, fps, and return 3 values for the gradio UI
    """
    try:
        from ..video_audio_pipeline import get_quick_vid_info
    except ImportError:
        return '---', '---', '---'
    
    if vid_path is None:
        return '---', '---', '---'
    
    fps, fcount, resolution = get_quick_vid_info(vid_path.name)
    expected_out_fps = set_interp_out_fps(interp_x, slow_x_enabled, slom_x, fps)
    
    return (
        str(round(fps, 2)) if fps is not None else '---',
        round(fcount, 2) if fcount is not None else '---',
        round(expected_out_fps, 2) if expected_out_fps != '---' else '---'
    )


def process_interp_vid_upload_logic(file,
                                   engine: str,
                                   x_am: int,
                                   sl_enabled: bool,
                                   sl_am: int,
                                   keep_imgs: bool,
                                   f_location: str,
                                   f_crf: int,
                                   f_preset: str,
                                   in_vid_fps: float,
                                   f_models_path: str,
                                   vid_file_name: str) -> Optional[str]:
    """
    Handle call to interpolate an uploaded video from gradio button
    Contains side effects (file operations) that are isolated
    
    Args:
        file: Uploaded file object
        engine: Interpolation engine to use
        x_am: Interpolation factor
        sl_enabled: Slow motion enabled
        sl_am: Slow motion amount
        keep_imgs: Keep interpolated images
        f_location: FFmpeg location
        f_crf: Video quality setting
        f_preset: Encoding preset
        in_vid_fps: Input video FPS
        f_models_path: Models folder path
        vid_file_name: Video file name
        
    Returns:
        Path to output video or None if failed
    """
    print("got a request to *frame interpolate* an existing video.")
    
    try:
        from ..video_audio_pipeline import get_quick_vid_info, vid2frames, media_file_has_audio
    except ImportError as e:
        print(f"❌ Required video processing modules not available: {e}")
        return None
    
    # Get video info (side effect: file reading)
    try:
        _, _, resolution = get_quick_vid_info(file.name)
    except Exception as e:
        print(f"❌ Could not get video info: {e}")
        return None
    
    # Generate clean folder name and unique output directory (pure functions)
    folder_name = clean_folder_name(Path(vid_file_name).stem)
    outdir = opts.outdir_samples or os.path.join(os.getcwd(), 'outputs')
    outdir_no_tmp = generate_unique_output_dir(outdir, folder_name)
    
    # Create output directory structure (side effect: directory creation)
    outdir = os.path.join(outdir_no_tmp, 'tmp_input_frames')
    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception as e:
        print(f"❌ Could not create output directory: {e}")
        return None
    
    # Extract frames from video (side effect: file extraction)
    try:
        vid2frames(
            video_path=file.name,
            video_in_frame_path=outdir,
            overwrite=True,
            extract_from_frame=0,
            extract_to_frame=-1,
            numeric_files_output=True,
            out_img_format='png'
        )
    except Exception as e:
        print(f"❌ Could not extract frames from video: {e}")
        return None
    
    # Check for audio stream (side effect: file analysis)
    audio_file_to_pass = None
    try:
        if media_file_has_audio(file.name, f_location):
            audio_file_to_pass = file.name
    except Exception as e:
        print(f"Warning: Could not check for audio stream: {e}")
    
    # Process interpolation (delegated to main pipeline)
    try:
        return process_video_interpolation(
            frame_interpolation_engine=engine,
            frame_interpolation_x_amount=x_am,
            frame_interpolation_slow_mo_enabled=sl_enabled,
            frame_interpolation_slow_mo_amount=sl_am,
            orig_vid_fps=in_vid_fps,
            deforum_models_path=f_models_path,
            real_audio_track=audio_file_to_pass,
            raw_output_imgs_path=outdir,
            img_batch_id=None,
            ffmpeg_location=f_location,
            ffmpeg_crf=f_crf,
            ffmpeg_preset=f_preset,
            keep_interp_imgs=keep_imgs,
            orig_vid_name=folder_name,
            resolution=resolution
        )
    except Exception as e:
        print(f"❌ Interpolation processing failed: {e}")
        return None


def process_interp_pics_upload_logic(pic_list: List,
                                    engine: str,
                                    x_am: int,
                                    sl_enabled: bool,
                                    sl_am: int,
                                    keep_imgs: bool,
                                    f_location: str,
                                    f_crf: int,
                                    f_preset: str,
                                    fps: float,
                                    f_models_path: str,
                                    resolution: tuple,
                                    add_soundtrack: str,
                                    audio_track: Optional[str]) -> Optional[str]:
    """
    Handle call to interpolate a set of uploaded images
    Contains side effects (file operations) that are isolated
    
    Args:
        pic_list: List of uploaded image files
        engine: Interpolation engine to use
        x_am: Interpolation factor
        sl_enabled: Slow motion enabled
        sl_am: Slow motion amount
        keep_imgs: Keep interpolated images
        f_location: FFmpeg location
        f_crf: Video quality setting
        f_preset: Encoding preset
        fps: Output FPS
        f_models_path: Models folder path
        resolution: Image resolution
        add_soundtrack: Soundtrack mode
        audio_track: Audio file path
        
    Returns:
        Path to output video or None if failed
    """
    # Extract paths from upload list (pure function)
    pic_path_list = extract_pic_path_list(pic_list)
    print(f"got a request to *frame interpolate* a set of {len(pic_list)} images.")
    
    # Generate clean folder name and unique output directory (pure functions)
    folder_name = clean_folder_name(Path(pic_list[0].name).stem)
    outdir_no_tmp = generate_unique_output_dir(
        os.path.join(os.getcwd(), 'outputs'), folder_name
    )
    
    # Create output directory structure (side effect: directory creation)
    outdir = os.path.join(outdir_no_tmp, 'tmp_input_frames')
    try:
        os.makedirs(outdir, exist_ok=True)
    except Exception as e:
        print(f"❌ Could not create output directory: {e}")
        return None
    
    # Convert images to PNG format (side effect: file conversion)
    try:
        from ..general_utils import convert_images_from_list
        convert_images_from_list(paths=pic_path_list, output_dir=outdir, format='png')
    except ImportError as e:
        print(f"❌ Required image processing modules not available: {e}")
        return None
    except Exception as e:
        print(f"❌ Could not convert images: {e}")
        return None
    
    # Determine audio file (pure function)
    audio_file_to_pass = audio_track if add_soundtrack == 'File' else None
    
    # Process interpolation (delegated to main pipeline)
    try:
        return process_video_interpolation(
            frame_interpolation_engine=engine,
            frame_interpolation_x_amount=x_am,
            frame_interpolation_slow_mo_enabled=sl_enabled,
            frame_interpolation_slow_mo_amount=sl_am,
            orig_vid_fps=fps,
            deforum_models_path=f_models_path,
            real_audio_track=audio_file_to_pass,
            raw_output_imgs_path=outdir,
            img_batch_id=None,
            ffmpeg_location=f_location,
            ffmpeg_crf=f_crf,
            ffmpeg_preset=f_preset,
            keep_interp_imgs=keep_imgs,
            orig_vid_name=folder_name,
            resolution=resolution,
            dont_change_fps=True  # Use for random pics run
        )
    except Exception as e:
        print(f"❌ Interpolation processing failed: {e}")
        return None


def validate_upload_settings(file_or_pics, 
                            engine: str,
                            interpolation_factor: int) -> tuple[bool, str]:
    """
    Pure function: upload settings -> validation result
    Validate upload processing settings before starting interpolation
    """
    if not file_or_pics:
        return False, "No files provided for interpolation"
    
    if engine == 'None':
        return False, "No interpolation engine selected"
    
    if interpolation_factor < 2:
        return False, "Interpolation factor must be at least 2x"
    
    return True, ""


def estimate_upload_processing_time(file_count: int,
                                  interpolation_factor: int,
                                  engine: str,
                                  is_video: bool = False) -> float:
    """
    Pure function: upload parameters -> estimated processing time
    Estimate total processing time for upload interpolation
    """
    # Import estimation from main pipeline
    from .main_pipeline import estimate_interpolation_time
    
    # Estimate frame count based on input type
    if is_video:
        # Rough estimate: 30 FPS for video, duration unknown
        estimated_frames = 30 * 10  # Assume 10-second video
    else:
        estimated_frames = file_count
    
    # Add overhead for file processing
    file_processing_overhead = file_count * 0.1  # 0.1 seconds per file
    
    interpolation_time = estimate_interpolation_time(
        estimated_frames, interpolation_factor, engine
    )
    
    return interpolation_time + file_processing_overhead


def get_upload_processing_info() -> dict:
    """
    Pure function: -> upload processing information
    Get information about upload processing capabilities and requirements
    """
    from .main_pipeline import get_available_interpolation_engines
    
    return {
        "supported_video_formats": [".mp4", ".avi", ".mov", ".mkv", ".webm"],
        "supported_image_formats": [".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
        "available_engines": get_available_interpolation_engines(),
        "max_interpolation_factor": 10,
        "min_interpolation_factor": 2,
        "supports_slow_motion": True,
        "supports_audio_passthrough": True,
        "output_format": "MP4"
    }


def format_upload_status_report() -> str:
    """
    Pure function: -> formatted upload status report
    Generate a status report for upload processing capabilities
    """
    info = get_upload_processing_info()
    
    return f"""
📤 UPLOAD PROCESSING STATUS

🎬 Supported Input Formats:
  • Video: {', '.join(info['supported_video_formats'])}
  • Images: {', '.join(info['supported_image_formats'])}

🔧 Available Engines:
  {', '.join(info['available_engines'])}

⚙️ Processing Options:
  • Interpolation: {info['min_interpolation_factor']}x to {info['max_interpolation_factor']}x
  • Slow Motion: {'Yes' if info['supports_slow_motion'] else 'No'}
  • Audio Passthrough: {'Yes' if info['supports_audio_passthrough'] else 'No'}
  • Output Format: {info['output_format']}

🚀 Upload processing ready for frame interpolation!
"""


def create_upload_config(engine: str = "RIFE v4.3",
                        interpolation_factor: int = 2,
                        slow_motion: bool = False,
                        keep_frames: bool = False) -> dict:
    """
    Pure function: parameters -> upload processing configuration
    Create a validated configuration for upload processing
    """
    config = {
        "engine": engine,
        "interpolation_factor": interpolation_factor,
        "slow_motion_enabled": slow_motion,
        "keep_interpolated_frames": keep_frames,
        "processing_info": get_upload_processing_info()
    }
    
    # Validate configuration
    is_valid, error = validate_upload_settings("dummy", engine, interpolation_factor)
    config["valid"] = is_valid
    config["error"] = error if not is_valid else None
    
    return config
