"""
Frame Interpolation Module

Modular frame interpolation system following functional programming principles.
This module provides a clean interface to the underlying interpolation components.

The original 500+ line frame_interpolation_pipeline.py has been split into focused modules:
- utilities.py: Pure utility functions and calculations
- rife_processor.py: RIFE-specific interpolation logic
- film_processor.py: FILM-specific interpolation logic
- main_pipeline.py: Central coordination and pipeline management
- upload_processing.py: Video and image upload handling
"""

# Import main pipeline functions
from .main_pipeline import (
    process_video_interpolation,
    validate_interpolation_pipeline,
    get_available_interpolation_engines,
    estimate_interpolation_time,
    get_interpolation_memory_requirements,
    create_interpolation_config,
    format_interpolation_status_report,
    get_engine_comparison
)

# Import upload processing functions
from .upload_processing import (
    gradio_f_interp_get_fps_and_fcount,
    process_interp_vid_upload_logic,
    process_interp_pics_upload_logic,
    validate_upload_settings,
    estimate_upload_processing_time,
    get_upload_processing_info,
    format_upload_status_report,
    create_upload_config
)

# Import utilities
from .utilities import (
    extract_rife_name,
    clean_folder_name,
    set_interp_out_fps,
    calculate_frames_to_add,
    generate_unique_output_dir,
    build_film_paths,
    validate_interpolation_settings,
    determine_uhd_mode,
    calculate_final_fps,
    should_disable_audio,
    should_disable_subtitles,
    get_film_model_info,
    extract_pic_path_list,
    determine_soundtrack_mode,
    should_keep_frames
)

# Import RIFE processor
from .rife_processor import (
    is_rife_available,
    validate_rife_settings,
    process_rife_interpolation,
    get_rife_model_requirements,
    estimate_rife_processing_time,
    get_rife_memory_requirements,
    create_rife_config,
    format_rife_status_report
)

# Import FILM processor
from .film_processor import (
    is_film_available,
    validate_film_settings,
    prepare_film_inference,
    check_and_download_film_model,
    get_film_model_requirements,
    estimate_film_processing_time,
    get_film_memory_requirements,
    create_film_config,
    format_film_status_report
)

# Define public API
__all__ = [
    # Main pipeline functions
    "process_video_interpolation",
    "validate_interpolation_pipeline", 
    "get_available_interpolation_engines",
    "estimate_interpolation_time",
    "get_interpolation_memory_requirements",
    "create_interpolation_config",
    "format_interpolation_status_report",
    "get_engine_comparison",
    
    # Upload processing functions
    "gradio_f_interp_get_fps_and_fcount",
    "process_interp_vid_upload_logic",
    "process_interp_pics_upload_logic",
    "validate_upload_settings",
    "estimate_upload_processing_time",
    "get_upload_processing_info",
    "format_upload_status_report",
    "create_upload_config",
    
    # Utility functions
    "extract_rife_name",
    "clean_folder_name",
    "set_interp_out_fps",
    "calculate_frames_to_add",
    "generate_unique_output_dir",
    "build_film_paths",
    "validate_interpolation_settings",
    "determine_uhd_mode",
    "calculate_final_fps",
    "should_disable_audio",
    "should_disable_subtitles",
    "get_film_model_info",
    "extract_pic_path_list",
    "determine_soundtrack_mode",
    "should_keep_frames",
    
    # RIFE processor functions
    "is_rife_available",
    "validate_rife_settings",
    "process_rife_interpolation",
    "get_rife_model_requirements",
    "estimate_rife_processing_time",
    "get_rife_memory_requirements",
    "create_rife_config",
    "format_rife_status_report",
    
    # FILM processor functions
    "is_film_available",
    "validate_film_settings",
    "prepare_film_inference",
    "check_and_download_film_model",
    "get_film_model_requirements",
    "estimate_film_processing_time",
    "get_film_memory_requirements",
    "create_film_config",
    "format_film_status_report"
]

# Backward compatibility aliases for legacy imports
# These maintain the exact same function signatures as the original file

# Legacy function aliases
calculate_frames_to_add = calculate_frames_to_add
check_and_download_film_model = check_and_download_film_model
prepare_film_inference = prepare_film_inference

# Version info
__version__ = "2.10.2"
__author__ = "Deforum Team"
__description__ = "Modular frame interpolation system with RIFE and FILM support"
__refactored__ = "Phase 2.10.2 - Large file modularization completed"

# Module status information
def get_module_status() -> dict:
    """
    Pure function: -> module status information
    Get comprehensive status of the frame interpolation module
    """
    return {
        "version": __version__,
        "refactored": __refactored__,
        "rife_available": is_rife_available(),
        "film_available": is_film_available(),
        "total_engines": len(get_available_interpolation_engines()),
        "modular_architecture": True,
        "functional_programming": True,
        "backward_compatible": True
    }


def print_module_status():
    """Print a formatted status report for the frame interpolation module"""
    status = get_module_status()
    print(f"""
🎬 FRAME INTERPOLATION MODULE STATUS

📦 Module Info:
  • Version: {status['version']}
  • Architecture: {'Modular' if status['modular_architecture'] else 'Legacy'}
  • Programming Style: {'Functional' if status['functional_programming'] else 'Procedural'}
  • Backward Compatible: {'Yes' if status['backward_compatible'] else 'No'}

🚀 Engine Availability:
  • RIFE: {'Available' if status['rife_available'] else 'Not Available'}
  • FILM: {'Available' if status['film_available'] else 'Not Available'}
  • Total Engines: {status['total_engines']}

✅ Refactoring: {status['refactored']}
""")
