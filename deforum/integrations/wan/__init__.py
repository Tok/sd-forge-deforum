"""
🎬 WAN Video Generation Module
Organized WAN (Wan-Video/Wan2.1) video generation implementation for Deforum

Directory Structure:
├── pipelines/          # Different pipeline implementations
│   ├── diffusers_pipeline.py    # Diffusers-based pipeline
│   ├── vace_pipeline.py         # VACE (Video and Content Editing) pipeline
│   └── procedural_pipeline.py   # Fallback procedural generation
├── models/             # Model components
│   ├── t5_encoder.py           # T5 text encoder
│   ├── vae.py                  # VAE encoder/decoder
│   └── dit.py                  # Diffusion Transformer
├── configs/            # Model configurations
│   ├── t2v_1_3b.py            # T2V 1.3B configuration
│   ├── t2v_14b.py             # T2V 14B configuration
│   └── shared_config.py        # Shared configurations
├── utils/              # Utility functions
│   ├── flow_matching.py        # Flow matching schedulers
│   ├── video_utils.py          # Video processing utilities
│   └── model_discovery.py      # Model discovery and validation
└── integration/        # Integration with Deforum
    ├── simple_integration.py   # Simple integration wrapper
    └── unified_integration.py  # Unified integration interface
"""

# Main public interface - import core components
from .utils.model_discovery import WanModelDiscovery
from .pipelines.procedural_pipeline import WanProceduralPipeline

# Import sub-modules
from . import utils
from . import pipelines
from . import configs
from . import models
from . import integration

# Try to import unified integration, fallback if dependencies missing
try:
    from .integration.unified_integration import WanUnifiedIntegration
    UNIFIED_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Unified integration not available: {e}")
    WanUnifiedIntegration = None
    UNIFIED_AVAILABLE = False

# Version information
__version__ = "1.0.0"
__author__ = "Deforum WAN Team"

# Public API
__all__ = [
    "WanUnifiedIntegration",
    "WanModelDiscovery", 
    "WanProceduralPipeline",
    'run_wan_generation',
    'enhance_with_wan',
    'wan_video_processing',
]

def create_wan_pipeline(model_path: str = None, pipeline_type: str = "auto"):
    """
    Factory function to create appropriate WAN pipeline
    
    Args:
        model_path: Path to WAN model directory (optional)
        pipeline_type: Type of pipeline ("auto", "diffusers", "vace", "procedural")
        
    Returns:
        WAN pipeline instance
    """
    if UNIFIED_AVAILABLE and WanUnifiedIntegration:
        integration = WanUnifiedIntegration()
        
        if model_path:
            integration.load_pipeline(model_path, pipeline_type)
        
        return integration
    else:
        # Fallback to procedural pipeline if unified not available
        print("🔄 Using procedural pipeline fallback")
        pipeline = WanProceduralPipeline()
        pipeline.load_components()
        return pipeline

def discover_wan_models():
    """
    Discover available WAN models
    
    Returns:
        List of discovered model dictionaries
    """
    discovery = WanModelDiscovery()
    return discovery.discover_models() 