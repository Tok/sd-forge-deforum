"""
FILM Frame Interpolation Processing

FILM-specific interpolation logic with model management and conditional loading.
All functions follow functional programming principles with side effect isolation.
"""

import os
import shutil
from pathlib import Path
from typing import Optional
from .utilities import (
    build_film_paths, calculate_frames_to_add, get_film_model_info,
    determine_soundtrack_mode, should_keep_frames
)

# Conditional import for FILM with graceful fallback
try:
    from film_interpolation.film_inference import run_film_interp_infer
    FILM_AVAILABLE = True
except ImportError:
    FILM_AVAILABLE = False
    def run_film_interp_infer(*args, **kwargs):
        raise ImportError("FILM interpolation is not installed. Please install FILM for frame interpolation functionality.")


def is_film_available() -> bool:
    """Pure function: -> FILM availability status"""
    return FILM_AVAILABLE


def validate_film_settings() -> tuple[bool, str]:
    """
    Pure function: -> FILM availability validation
    Validate FILM-specific requirements
    """
    if not FILM_AVAILABLE:
        return False, "FILM interpolation is not installed. Please install FILM for frame interpolation functionality."
    
    return True, ""


def check_and_download_film_model(model_name: str, model_dest_folder: str) -> None:
    """
    Side effect function: Download and verify FILM model if needed
    This function contains side effects (file download, checksum verification)
    """
    try:
        from torch.hub import download_url_to_file
        from ..general_utils import checksum
    except ImportError as e:
        raise ImportError(f"Required dependencies not available: {e}")
    
    # Get model info (pure function)
    try:
        model_info = get_film_model_info(model_name)
    except ValueError as e:
        raise Exception(f"Got a request to download an unknown FILM model: {e}")
    
    model_dest_path = os.path.join(model_dest_folder, model_name)
    
    # Check if model already exists
    if os.path.exists(model_dest_path):
        return
    
    try:
        # Create destination folder
        os.makedirs(model_dest_folder, exist_ok=True)
        
        # Download model from URL
        download_url_to_file(model_info['download_url'], model_dest_path)
        
        # Verify checksum
        if checksum(model_dest_path) != model_info['hash']:
            raise Exception(f"Checksum verification failed for {model_name}")
            
    except Exception as e:
        download_url = model_info['download_url']
        raise Exception(f"Error while downloading {model_name}. Please download from: {download_url}, and put in: {model_dest_folder}. Error: {e}")


def prepare_film_inference(deforum_models_path: str,
                          x_am: int,
                          sl_enabled: bool,
                          sl_am: int,
                          keep_imgs: bool,
                          raw_output_imgs_path: str,
                          img_batch_id: Optional[str],
                          f_location: str,
                          f_crf: int,
                          f_preset: str,
                          fps: float,
                          audio_track: Optional[str],
                          orig_vid_name: Optional[str],
                          is_random_pics_run: bool,
                          srt_path: Optional[str] = None) -> str:
    """
    Main FILM inference preparation and execution function
    Contains side effects (file operations, model inference) that are isolated
    """
    # Validate FILM availability (pure function)
    is_valid, error_msg = validate_film_settings()
    if not is_valid:
        print(f"❌ FILM validation failed: {error_msg}")
        return None
    
    # Build all necessary paths (pure function)
    paths = build_film_paths(raw_output_imgs_path, img_batch_id or "", orig_vid_name or "", x_am)
    
    # Determine final video path with slow motion suffix
    interp_vid_path = paths['interp_vid_path']
    if sl_enabled:
        interp_vid_path = f"{interp_vid_path}_slomo_x{sl_am}"
    interp_vid_path = f"{interp_vid_path}.mp4"
    
    # Set up model paths
    film_model_name = 'film_net_fp16.pt'
    film_model_folder = os.path.join(deforum_models_path, 'film_interpolation')
    film_model_path = os.path.join(film_model_folder, film_model_name)
    
    # Side effect: Prepare temporary folder and duplicate images
    try:
        from ..general_utils import duplicate_pngs_from_folder
        
        temp_convert_raw_png_path = paths['temp_convert_raw_png_path']
        
        if is_random_pics_run:
            # Copy-paste images without re-writing them
            total_frames = duplicate_pngs_from_folder(
                raw_output_imgs_path, temp_convert_raw_png_path, img_batch_id, 'DUMMY'
            )
        else:
            # Re-write pics as PNG to avoid 24/32 bit mixed outputs
            total_frames = duplicate_pngs_from_folder(
                raw_output_imgs_path, temp_convert_raw_png_path, img_batch_id, None
            )
    except Exception as e:
        print(f"❌ Failed to prepare images for FILM processing: {e}")
        return None
    
    # Side effect: Download and verify FILM model
    try:
        check_and_download_film_model(film_model_name, film_model_folder)
    except Exception as e:
        print(f"❌ Failed to download FILM model: {e}")
        return None
    
    # Calculate interpolation parameters (pure function)
    film_in_between_frames_count = calculate_frames_to_add(total_frames, x_am)
    
    # Side effect: Run FILM inference
    try:
        run_film_interp_infer(
            model_path=film_model_path,
            input_folder=temp_convert_raw_png_path,
            save_folder=paths['custom_interp_path'],
            inter_frames=film_in_between_frames_count
        )
    except Exception as e:
        print(f"❌ FILM inference failed: {e}")
        return None
    
    # Side effect: Stitch video with ffmpeg
    exception_raised = False
    try:
        from ..video_audio_pipeline import ffmpeg_stitch_video
        
        add_soundtrack = determine_soundtrack_mode(audio_track)
        
        print("*Passing interpolated frames to ffmpeg...*")
        ffmpeg_stitch_video(
            ffmpeg_location=f_location,
            fps=fps,
            outmp4_path=interp_vid_path,
            stitch_from_frame=0,
            stitch_to_frame=999999999,
            imgs_path=str(paths['img_path_for_ffmpeg']),
            add_soundtrack=add_soundtrack,
            audio_path=audio_track,
            crf=f_crf,
            preset=f_preset,
            srt_path=srt_path
        )
    except Exception as e:
        exception_raised = True
        print(f"An error occurred while stitching the video: {e}")
    
    # Side effect: Cleanup based on settings and results
    _cleanup_film_processing(
        paths, orig_vid_name, keep_imgs, exception_raised, fps, raw_output_imgs_path
    )
    
    return interp_vid_path


def _cleanup_film_processing(paths: dict,
                           orig_vid_name: Optional[str],
                           keep_imgs: bool,
                           exception_raised: bool,
                           fps: float,
                           raw_output_imgs_path: str) -> None:
    """
    Side effect function: Clean up temporary files after FILM processing
    Contains file system operations that are isolated side effects
    """
    custom_interp_path = paths['custom_interp_path']
    temp_convert_raw_png_path = paths['temp_convert_raw_png_path']
    parent_folder = str(Path(raw_output_imgs_path).parent)
    
    # Move interpolated frames if needed
    if orig_vid_name and (keep_imgs or exception_raised):
        try:
            shutil.move(custom_interp_path, parent_folder)
        except Exception as e:
            print(f"Warning: Could not move interpolated frames: {e}")
    
    # Remove interpolated frames if not keeping them
    keep_frames = should_keep_frames(fps, keep_imgs, exception_raised)
    if not keep_frames:
        try:
            shutil.rmtree(custom_interp_path, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not remove interpolated frames: {e}")
    
    # Delete duplicated raw non-interpolated frames
    try:
        shutil.rmtree(temp_convert_raw_png_path, ignore_errors=True)
    except Exception as e:
        print(f"Warning: Could not remove temporary frames: {e}")
    
    # Remove folder with raw input frames for video input
    if orig_vid_name:
        try:
            shutil.rmtree(raw_output_imgs_path, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Could not remove input frames: {e}")


def get_film_model_requirements() -> dict:
    """
    Pure function: -> FILM model requirements information
    Return information about FILM model requirements and availability
    """
    return {
        "available": FILM_AVAILABLE,
        "model_name": "film_net_fp16.pt",
        "model_size_mb": 41.5,
        "supports_variable_interpolation": True,
        "supports_slow_motion": True,
        "requirements": ["film_interpolation package", "torch.hub for downloads"],
        "model_info": get_film_model_info("film_net_fp16.pt") if FILM_AVAILABLE else None
    }


def estimate_film_processing_time(frame_count: int, interpolation_factor: int) -> float:
    """
    Pure function: processing parameters -> estimated time
    Estimate FILM processing time based on frame count and settings
    """
    # FILM is generally slower than RIFE but produces higher quality
    base_time_per_frame = 1.0  # seconds per frame
    
    # Factor in interpolation amount
    total_output_frames = frame_count * interpolation_factor
    
    # FILM processing time increases more with interpolation factor
    interpolation_overhead = interpolation_factor * 0.2
    
    # Estimate total time with overhead
    estimated_time = total_output_frames * (base_time_per_frame + interpolation_overhead) * 1.3
    
    return estimated_time


def get_film_memory_requirements(resolution: tuple, interpolation_factor: int) -> dict:
    """
    Pure function: processing parameters -> memory requirements
    Estimate FILM memory requirements for given settings
    """
    width, height = resolution if resolution else (1024, 1024)
    pixel_count = width * height
    
    # FILM generally requires more memory than RIFE
    base_vram = 3072  # 3GB base requirement
    
    # Additional VRAM based on resolution
    resolution_vram = (pixel_count / (1024 * 1024)) * 800  # ~800MB per megapixel
    
    # Factor in interpolation factor
    interpolation_vram = interpolation_factor * 512  # More VRAM per interpolation factor
    
    total_vram = base_vram + resolution_vram + interpolation_vram
    
    return {
        "base_vram_mb": base_vram,
        "resolution_vram_mb": resolution_vram,
        "interpolation_vram_mb": interpolation_vram,
        "total_vram_mb": total_vram,
        "recommended_vram_gb": total_vram / 1024,
        "model_size_mb": 41.5
    }


def create_film_config(interpolation_factor: int = 2,
                      slow_mo_enabled: bool = False,
                      slow_mo_amount: int = 2) -> dict:
    """
    Pure function: parameters -> FILM configuration
    Create a validated FILM configuration dictionary
    """
    config = {
        "interpolation_factor": interpolation_factor,
        "slow_motion_enabled": slow_mo_enabled,
        "slow_motion_factor": slow_mo_amount,
        "model_name": "film_net_fp16.pt",
        "engine": "FILM"
    }
    
    # Validate configuration
    is_valid, error = validate_film_settings()
    config["valid"] = is_valid
    config["error"] = error if not is_valid else None
    
    return config


def format_film_status_report(available: bool = None) -> str:
    """
    Pure function: availability status -> formatted report
    Generate a status report for FILM availability and capabilities
    """
    if available is None:
        available = FILM_AVAILABLE
    
    if available:
        requirements = get_film_model_requirements()
        return f"""
✅ FILM Frame Interpolation Available

🔧 Capabilities:
- Model: {requirements['model_name']} ({requirements['model_size_mb']} MB)
- Variable Interpolation: {'Yes' if requirements['supports_variable_interpolation'] else 'No'}
- Slow Motion: {'Yes' if requirements['supports_slow_motion'] else 'No'}
- Quality: High (slower but better quality than RIFE)

🚀 FILM is ready for high-quality frame interpolation!
"""
    else:
        return """
❌ FILM Frame Interpolation Not Available

📋 Missing Requirements:
- film_interpolation package not installed
- Required dependencies not found

💡 Installation:
Please install FILM for high-quality frame interpolation functionality.
Refer to the installation documentation for setup instructions.
"""
