"""
RIFE Frame Interpolation Processing

RIFE-specific interpolation logic with conditional dependency loading.
All functions follow functional programming principles with side effect isolation.
"""

from typing import Optional
from .utilities import extract_rife_name, determine_uhd_mode, validate_interpolation_settings

# Conditional import for RIFE with graceful fallback
try:
    from rife.inference_video import run_rife_new_video_infer
    RIFE_AVAILABLE = True
except ImportError:
    RIFE_AVAILABLE = False
    def run_rife_new_video_infer(*args, **kwargs):
        raise ImportError("RIFE is not installed. Please install RIFE for frame interpolation functionality.")


def is_rife_available() -> bool:
    """Pure function: -> RIFE availability status"""
    return RIFE_AVAILABLE


def validate_rife_settings(frame_interpolation_x_amount: int) -> tuple[bool, str]:
    """
    Pure function: RIFE settings -> validation result
    Validate RIFE-specific interpolation parameters
    """
    if not RIFE_AVAILABLE:
        return False, "RIFE is not installed. Please install RIFE for frame interpolation functionality."
    
    if frame_interpolation_x_amount not in range(2, 11):
        return False, "frame_interpolation_x_amount must be between 2x and 10x"
    
    return True, ""


def process_rife_interpolation(frame_interpolation_x_amount: int,
                              frame_interpolation_slow_mo_enabled: bool,
                              frame_interpolation_slow_mo_amount: int,
                              frame_interpolation_engine: str,
                              fps: float,
                              deforum_models_path: str,
                              real_audio_track: Optional[str],
                              raw_output_imgs_path: str,
                              img_batch_id: Optional[str],
                              ffmpeg_location: str,
                              ffmpeg_crf: int,
                              ffmpeg_preset: str,
                              keep_interp_imgs: bool,
                              orig_vid_name: Optional[str],
                              resolution: Optional[tuple],
                              srt_path: Optional[str] = None) -> Optional[str]:
    """
    Process RIFE frame interpolation with functional composition and side effect isolation
    
    Returns: Path to output video or None if failed
    """
    # Validate RIFE settings first (pure function)
    is_valid, error_msg = validate_rife_settings(frame_interpolation_x_amount)
    if not is_valid:
        print(f"❌ RIFE validation failed: {error_msg}")
        return None
    
    # Extract model folder name (pure transformation)
    try:
        actual_model_folder_name = extract_rife_name(frame_interpolation_engine)
    except ValueError as e:
        print(f"❌ Invalid RIFE engine name: {e}")
        return None
    
    # Determine UHD mode (pure function)
    uhd_mode = determine_uhd_mode(resolution)
    
    # Side effect isolation: Run RIFE inference
    try:
        result_path = run_rife_new_video_infer(
            interp_x_amount=frame_interpolation_x_amount,
            slow_mo_enabled=frame_interpolation_slow_mo_enabled,
            slow_mo_x_amount=frame_interpolation_slow_mo_amount,
            model=actual_model_folder_name,
            fps=fps,
            deforum_models_path=deforum_models_path,
            audio_track=real_audio_track,
            raw_output_imgs_path=raw_output_imgs_path,
            img_batch_id=img_batch_id,
            ffmpeg_location=ffmpeg_location,
            ffmpeg_crf=ffmpeg_crf,
            ffmpeg_preset=ffmpeg_preset,
            keep_imgs=keep_interp_imgs,
            orig_vid_name=orig_vid_name,
            UHD=uhd_mode,
            srt_path=srt_path
        )
        return result_path
    except Exception as e:
        print(f"❌ RIFE interpolation failed: {e}")
        return None


def get_rife_model_requirements() -> dict:
    """
    Pure function: -> RIFE model requirements information
    Return information about RIFE model requirements and availability
    """
    return {
        "available": RIFE_AVAILABLE,
        "min_interpolation_factor": 2,
        "max_interpolation_factor": 10,
        "supports_uhd": True,
        "supports_slow_motion": True,
        "model_formats": ["RIFE v4.3", "RIFE v4.0", "RIFE v3.8"],
        "requirements": ["rife package", "inference_video module"]
    }


def estimate_rife_processing_time(frame_count: int, 
                                 interpolation_factor: int,
                                 resolution: tuple,
                                 uhd_mode: bool = None) -> float:
    """
    Pure function: processing parameters -> estimated time
    Estimate RIFE processing time based on frame count and settings
    """
    if uhd_mode is None:
        uhd_mode = determine_uhd_mode(resolution)
    
    # Base processing time per frame (in seconds)
    base_time_per_frame = 0.5 if not uhd_mode else 1.5
    
    # Factor in interpolation amount (more frames = longer processing)
    total_output_frames = frame_count * interpolation_factor
    
    # Estimate total time with some overhead
    estimated_time = total_output_frames * base_time_per_frame * 1.2  # 20% overhead
    
    return estimated_time


def get_rife_memory_requirements(resolution: tuple, interpolation_factor: int) -> dict:
    """
    Pure function: processing parameters -> memory requirements
    Estimate RIFE memory requirements for given settings
    """
    width, height = resolution if resolution else (1024, 1024)
    pixel_count = width * height
    
    # Base VRAM requirement (in MB)
    base_vram = 2048  # 2GB base requirement
    
    # Additional VRAM based on resolution
    resolution_vram = (pixel_count / (1024 * 1024)) * 500  # ~500MB per megapixel
    
    # Factor in interpolation factor
    interpolation_vram = interpolation_factor * 256  # Additional VRAM per interpolation factor
    
    total_vram = base_vram + resolution_vram + interpolation_vram
    
    return {
        "base_vram_mb": base_vram,
        "resolution_vram_mb": resolution_vram,
        "interpolation_vram_mb": interpolation_vram,
        "total_vram_mb": total_vram,
        "recommended_vram_gb": total_vram / 1024,
        "uhd_mode": determine_uhd_mode(resolution)
    }


def create_rife_config(frame_interpolation_x_amount: int = 2,
                      slow_mo_enabled: bool = False,
                      slow_mo_amount: int = 2,
                      model_version: str = "RIFE v4.3") -> dict:
    """
    Pure function: parameters -> RIFE configuration
    Create a validated RIFE configuration dictionary
    """
    config = {
        "interpolation_factor": frame_interpolation_x_amount,
        "slow_motion_enabled": slow_mo_enabled,
        "slow_motion_factor": slow_mo_amount,
        "model_version": model_version,
        "engine": "RIFE"
    }
    
    # Validate configuration
    is_valid, error = validate_rife_settings(frame_interpolation_x_amount)
    config["valid"] = is_valid
    config["error"] = error if not is_valid else None
    
    return config


def format_rife_status_report(available: bool = None) -> str:
    """
    Pure function: availability status -> formatted report
    Generate a status report for RIFE availability and capabilities
    """
    if available is None:
        available = RIFE_AVAILABLE
    
    if available:
        requirements = get_rife_model_requirements()
        return f"""
✅ RIFE Frame Interpolation Available

🔧 Capabilities:
- Interpolation Factors: {requirements['min_interpolation_factor']}x to {requirements['max_interpolation_factor']}x
- UHD Support: {'Yes' if requirements['supports_uhd'] else 'No'}
- Slow Motion: {'Yes' if requirements['supports_slow_motion'] else 'No'}
- Supported Models: {', '.join(requirements['model_formats'])}

🚀 RIFE is ready for frame interpolation!
"""
    else:
        return """
❌ RIFE Frame Interpolation Not Available

📋 Missing Requirements:
- RIFE package not installed
- inference_video module not found

💡 Installation:
Please install RIFE for frame interpolation functionality.
Refer to the installation documentation for setup instructions.
"""
