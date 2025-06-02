"""
Main Frame Interpolation Pipeline

Central coordination of frame interpolation processing using functional composition.
Integrates RIFE and FILM processors with side effect isolation.
"""

from typing import Optional
from .utilities import (
    calculate_final_fps, should_disable_audio, should_disable_subtitles,
    validate_interpolation_settings
)
from .rife_processor import process_rife_interpolation, is_rife_available
from .film_processor import prepare_film_inference, is_film_available


def process_video_interpolation(frame_interpolation_engine: str,
                               frame_interpolation_x_amount: int,
                               frame_interpolation_slow_mo_enabled: bool,
                               frame_interpolation_slow_mo_amount: int,
                               orig_vid_fps: float,
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
                               dont_change_fps: bool = False,
                               srt_path: Optional[str] = None) -> Optional[str]:
    """
    Main frame interpolation pipeline using functional composition
    
    Args:
        frame_interpolation_engine: Engine to use ("RIFE", "FILM", or "None")
        frame_interpolation_x_amount: Interpolation factor (2x, 3x, etc.)
        frame_interpolation_slow_mo_enabled: Enable slow motion mode
        frame_interpolation_slow_mo_amount: Slow motion factor
        orig_vid_fps: Original video FPS
        deforum_models_path: Path to model directory
        real_audio_track: Path to audio file or None
        raw_output_imgs_path: Path to input images
        img_batch_id: Batch identifier
        ffmpeg_location: Path to ffmpeg executable
        ffmpeg_crf: Video quality setting
        ffmpeg_preset: Encoding preset
        keep_interp_imgs: Whether to keep interpolated frames
        orig_vid_name: Original video name
        resolution: Video resolution tuple
        dont_change_fps: Use for random pics (maintains original FPS)
        srt_path: Subtitle file path
        
    Returns:
        Path to output video or None if processing failed
    """
    # Pure function: Calculate final FPS
    is_random_pics_run = dont_change_fps
    fps = calculate_final_fps(
        orig_vid_fps, frame_interpolation_x_amount,
        frame_interpolation_slow_mo_enabled, frame_interpolation_slow_mo_amount,
        is_random_pics_run
    )
    
    # Pure function: Apply audio/subtitle disable logic for slow motion
    audio_track = None if should_disable_audio(real_audio_track, frame_interpolation_slow_mo_enabled) else real_audio_track
    subtitle_path = None if should_disable_subtitles(srt_path, frame_interpolation_slow_mo_enabled) else srt_path
    
    # Early exit for no interpolation
    if frame_interpolation_engine == 'None':
        return None
    
    # Route to appropriate processor based on engine
    if frame_interpolation_engine.startswith("RIFE"):
        return process_rife_interpolation(
            frame_interpolation_x_amount=frame_interpolation_x_amount,
            frame_interpolation_slow_mo_enabled=frame_interpolation_slow_mo_enabled,
            frame_interpolation_slow_mo_amount=frame_interpolation_slow_mo_amount,
            frame_interpolation_engine=frame_interpolation_engine,
            fps=fps,
            deforum_models_path=deforum_models_path,
            real_audio_track=audio_track,
            raw_output_imgs_path=raw_output_imgs_path,
            img_batch_id=img_batch_id,
            ffmpeg_location=ffmpeg_location,
            ffmpeg_crf=ffmpeg_crf,
            ffmpeg_preset=ffmpeg_preset,
            keep_interp_imgs=keep_interp_imgs,
            orig_vid_name=orig_vid_name,
            resolution=resolution,
            srt_path=subtitle_path
        )
    
    elif frame_interpolation_engine == 'FILM':
        return prepare_film_inference(
            deforum_models_path=deforum_models_path,
            x_am=frame_interpolation_x_amount,
            sl_enabled=frame_interpolation_slow_mo_enabled,
            sl_am=frame_interpolation_slow_mo_amount,
            keep_imgs=keep_interp_imgs,
            raw_output_imgs_path=raw_output_imgs_path,
            img_batch_id=img_batch_id,
            f_location=ffmpeg_location,
            f_crf=ffmpeg_crf,
            f_preset=ffmpeg_preset,
            fps=fps,
            audio_track=audio_track,
            orig_vid_name=orig_vid_name,
            is_random_pics_run=is_random_pics_run,
            srt_path=subtitle_path
        )
    
    else:
        print(f"Unknown Frame Interpolation engine chosen: {frame_interpolation_engine}. Doing nothing.")
        return None


def validate_interpolation_pipeline(frame_interpolation_engine: str,
                                  frame_interpolation_x_amount: int) -> tuple[bool, str]:
    """
    Pure function: pipeline settings -> validation result
    Validate complete interpolation pipeline settings
    """
    if frame_interpolation_engine == 'None':
        return True, "No interpolation requested"
    
    # Check engine availability
    if frame_interpolation_engine.startswith("RIFE"):
        if not is_rife_available():
            return False, "RIFE is not available. Please install RIFE for frame interpolation."
    elif frame_interpolation_engine == 'FILM':
        if not is_film_available():
            return False, "FILM is not available. Please install FILM for frame interpolation."
    else:
        return False, f"Unknown interpolation engine: {frame_interpolation_engine}"
    
    # Validate interpolation settings
    return validate_interpolation_settings(frame_interpolation_x_amount, frame_interpolation_engine)


def get_available_interpolation_engines() -> list[str]:
    """
    Pure function: -> list of available interpolation engines
    Check which interpolation engines are available
    """
    engines = ["None"]  # Always available
    
    if is_rife_available():
        engines.extend(["RIFE v4.3", "RIFE v4.0", "RIFE v3.8"])
    
    if is_film_available():
        engines.append("FILM")
    
    return engines


def estimate_interpolation_time(frame_count: int,
                              interpolation_factor: int,
                              engine: str,
                              resolution: tuple = None) -> float:
    """
    Pure function: processing parameters -> estimated time
    Estimate processing time for interpolation based on engine and settings
    """
    if engine == 'None':
        return 0.0
    
    if engine.startswith("RIFE"):
        from .rife_processor import estimate_rife_processing_time
        return estimate_rife_processing_time(frame_count, interpolation_factor, resolution or (1024, 1024))
    
    elif engine == 'FILM':
        from .film_processor import estimate_film_processing_time
        return estimate_film_processing_time(frame_count, interpolation_factor)
    
    else:
        return 0.0  # Unknown engine


def get_interpolation_memory_requirements(resolution: tuple,
                                        interpolation_factor: int,
                                        engine: str) -> dict:
    """
    Pure function: processing parameters -> memory requirements
    Get memory requirements for interpolation based on engine and settings
    """
    if engine == 'None':
        return {"total_vram_mb": 0, "recommended_vram_gb": 0}
    
    if engine.startswith("RIFE"):
        from .rife_processor import get_rife_memory_requirements
        return get_rife_memory_requirements(resolution, interpolation_factor)
    
    elif engine == 'FILM':
        from .film_processor import get_film_memory_requirements
        return get_film_memory_requirements(resolution, interpolation_factor)
    
    else:
        return {"total_vram_mb": 0, "recommended_vram_gb": 0, "error": f"Unknown engine: {engine}"}


def create_interpolation_config(engine: str = "RIFE v4.3",
                              interpolation_factor: int = 2,
                              slow_mo_enabled: bool = False,
                              slow_mo_factor: int = 2,
                              keep_frames: bool = False) -> dict:
    """
    Pure function: parameters -> interpolation configuration
    Create a complete interpolation configuration with validation
    """
    config = {
        "engine": engine,
        "interpolation_factor": interpolation_factor,
        "slow_motion_enabled": slow_mo_enabled,
        "slow_motion_factor": slow_mo_factor,
        "keep_interpolated_frames": keep_frames,
        "available_engines": get_available_interpolation_engines()
    }
    
    # Validate configuration
    is_valid, error = validate_interpolation_pipeline(engine, interpolation_factor)
    config["valid"] = is_valid
    config["error"] = error if not is_valid else None
    
    return config


def format_interpolation_status_report() -> str:
    """
    Pure function: -> formatted status report
    Generate a comprehensive status report for frame interpolation capabilities
    """
    available_engines = get_available_interpolation_engines()
    rife_available = is_rife_available()
    film_available = is_film_available()
    
    report = """
🎬 FRAME INTERPOLATION STATUS REPORT

🔧 Available Engines:
"""
    
    for engine in available_engines:
        if engine == "None":
            report += "  ✅ None (No Interpolation)\n"
        elif engine.startswith("RIFE"):
            report += f"  {'✅' if rife_available else '❌'} {engine}\n"
        elif engine == "FILM":
            report += f"  {'✅' if film_available else '❌'} {engine}\n"
    
    report += f"""
📊 Engine Status:
  • RIFE: {'Available' if rife_available else 'Not Available'}
  • FILM: {'Available' if film_available else 'Not Available'}

💡 Recommendations:
  • RIFE: Fast processing, good for most use cases
  • FILM: Higher quality, slower processing time
  • Use slow motion with caution - disables audio/subtitles

🚀 Frame interpolation system ready!
"""
    
    return report


def get_engine_comparison() -> dict:
    """
    Pure function: -> engine comparison data
    Compare available interpolation engines with their characteristics
    """
    engines = {}
    
    if is_rife_available():
        from .rife_processor import get_rife_model_requirements
        rife_req = get_rife_model_requirements()
        engines["RIFE"] = {
            "available": True,
            "speed": "Fast",
            "quality": "Good",
            "vram_requirement": "Medium",
            "supports_uhd": rife_req["supports_uhd"],
            "interpolation_range": f"{rife_req['min_interpolation_factor']}x-{rife_req['max_interpolation_factor']}x"
        }
    else:
        engines["RIFE"] = {"available": False}
    
    if is_film_available():
        from .film_processor import get_film_model_requirements
        film_req = get_film_model_requirements()
        engines["FILM"] = {
            "available": True,
            "speed": "Slow",
            "quality": "Excellent",
            "vram_requirement": "High",
            "model_size": f"{film_req['model_size_mb']} MB",
            "interpolation_range": "2x and higher"
        }
    else:
        engines["FILM"] = {"available": False}
    
    return engines
