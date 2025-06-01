"""
External library integrations and adapters.
"""

from .controlnet import *
from .rife import *
from .film import *
from .midas import *
from .raft import *
from .wan import *
from .parseq_adapter import *
from .webui_pipeline import *

# External libraries are available in external_libs/ subdirectory
# These include: py3d_tools, clipseg, film_interpolation, etc.

__all__ = [
    # ControlNet integration
    'setup_controlnet_ui',
    'get_controlnet_script_args',
    'is_controlnet_enabled',
    
    # Frame interpolation
    'run_rife_new_video_infer',
    'run_film_interp_infer',
    
    # Depth estimation
    'run_midas_depth',
    
    # RAFT optical flow
    'run_raft_flow',
    
    # WAN AI integration
    'run_wan_generation',
    
    # External adapters
    'parseq_adapter',
    'webui_pipeline',
] 