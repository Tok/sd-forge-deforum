"""
Depth processing with Depth-Anything-V2 (default), MiDaS, and video depth extraction.
"""

from .core_depth_analysis import *
from .depth_anything_v2_integration import *
from .midas_depth_estimation import *
from .video_depth_extraction import *

__all__ = [
    # Core depth functionality
    'process_depth',
    'depth_warp',
    
    # Depth-Anything-V2 integration (DEFAULT)
    'load_depth_anything_model',
    'estimate_depth_anything',
    'DepthAnything',
    'get_default_depth_estimator',
    
    # MiDaS integration (legacy support)
    'load_midas_model',
    'estimate_depth_midas',
    
    # Video depth extraction
    'extract_depth_from_video',
    'process_depth_vid_upload_logic',
]

# Default depth estimation method
DEFAULT_DEPTH_METHOD = 'depth_anything_v2' 