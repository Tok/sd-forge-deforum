"""
WAN Utilities Module
Shared utility functions for WAN video generation
"""

from .model_discovery import WanModelDiscovery
from .video_utils import VideoProcessor
from .flow_matching import WanFlowMatchingPipeline

# Try to import other utilities, fallback if dependencies missing
try:
    from .vace_processor import VACEProcessor
    VACE_PROCESSOR_AVAILABLE = True
except ImportError:
    VACEProcessor = None
    VACE_PROCESSOR_AVAILABLE = False

try:
    from .fm_solvers import WanFlowMatchingScheduler
    from .fm_solvers_unipc import WanUniPCScheduler
    FM_SOLVERS_AVAILABLE = True
except ImportError:
    WanFlowMatchingScheduler = None
    WanUniPCScheduler = None
    FM_SOLVERS_AVAILABLE = False

try:
    from .prompt_extend import PromptExtender
    from .qwen_vl_utils import QwenVLProcessor
    from .tensor_adapter import TensorAdapter
    EXTENDED_UTILS_AVAILABLE = True
except ImportError:
    PromptExtender = None
    QwenVLProcessor = None  
    TensorAdapter = None
    EXTENDED_UTILS_AVAILABLE = False

try:
    from .core_utils import cache_video, cache_image, str2bool
    CORE_UTILS_AVAILABLE = True
except ImportError:
    cache_video = None
    cache_image = None
    str2bool = None
    CORE_UTILS_AVAILABLE = False

__all__ = [
    "WanModelDiscovery",
    "VideoProcessor", 
    "WanFlowMatchingPipeline",
    "VACEProcessor",
    "WanFlowMatchingScheduler",
    "WanUniPCScheduler",
    "PromptExtender",
    "QwenVLProcessor",
    "TensorAdapter",
    "cache_video",
    "cache_image",
    "str2bool",
    "VACE_PROCESSOR_AVAILABLE",
    "FM_SOLVERS_AVAILABLE",
    "EXTENDED_UTILS_AVAILABLE",
    "CORE_UTILS_AVAILABLE"
] 