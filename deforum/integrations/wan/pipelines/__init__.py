"""
WAN Pipelines Module
Different pipeline implementations for WAN video generation
"""

from .procedural_pipeline import WanProceduralPipeline

# Try to import other pipelines, fallback if dependencies missing
try:
    from .diffusers_pipeline import WanRealIntegration as WanDiffusersPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    WanDiffusersPipeline = None
    DIFFUSERS_AVAILABLE = False

try:
    from .vace_pipeline import WanWorkingIntegration as WanVACEPipeline
    VACE_AVAILABLE = True
except ImportError:
    WanVACEPipeline = None
    VACE_AVAILABLE = False

__all__ = [
    "WanProceduralPipeline",
    "WanDiffusersPipeline",
    "WanVACEPipeline",
    "DIFFUSERS_AVAILABLE",
    "VACE_AVAILABLE"
] 