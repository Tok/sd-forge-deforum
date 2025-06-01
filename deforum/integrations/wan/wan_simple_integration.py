#!/usr/bin/env python3
"""
WAN Simple Integration (Backward Compatibility Layer)
Main entry point that delegates to the new modular WAN system
"""

from typing import List, Dict, Optional, Any
from pathlib import Path

# Import new modular components
from .wan_core_integration import WanCoreIntegration, get_webui_root, wan_generate_video_main
from .wan_pipeline_manager import WanPipelineManager
from .wan_config_handler import WanConfigHandler
from .wan_utilities import WanUtilities


class WanSimpleIntegration:
    """Simplified Wan integration with auto-discovery and proper progress styling
    
    This is a backward compatibility wrapper that delegates to the new modular system.
    All functionality has been moved to focused modules:
    - WanCoreIntegration: Core initialization and model management
    - WanPipelineManager: Pipeline loading and video generation
    - WanConfigHandler: Configuration and metadata management
    - WanUtilities: Utility functions and frame processing
    """
    
    def __init__(self, device='cuda'):
        """Initialize WAN integration.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        # Initialize core integration
        self.core = WanCoreIntegration(device)
        
        # Initialize component managers
        self.pipeline_manager = WanPipelineManager(self.core)
        self.config_handler = WanConfigHandler(self.core)
        self.utilities = WanUtilities(self.core)
        
        # Expose core properties for backward compatibility
        self.device = self.core.device
        self.webui_root = self.core.webui_root
        self.models = self.core.models
        self.pipeline = self.core.pipeline
        self.model_size = self.core.model_size
        self.optimal_width = self.core.optimal_width
        self.optimal_height = self.core.optimal_height
        self.flash_attention_mode = self.core.flash_attention_mode
    
    # ============================================================================
    # Model Discovery and Management (delegated to PipelineManager)
    # ============================================================================
    
    def discover_models(self) -> List[Dict]:
        """Discover available Wan models."""
        models = self.pipeline_manager.discover_models()
        # Update local reference for backward compatibility
        self.models = self.core.models
        return models
    
    def get_best_model(self) -> Optional[Dict]:
        """Get the best available model."""
        return self.core.get_best_model()
    
    def load_simple_wan_pipeline(self, model_info: Dict, wan_args=None) -> bool:
        """Load a WAN pipeline."""
        success = self.pipeline_manager.load_simple_wan_pipeline(model_info, wan_args)
        # Update local reference for backward compatibility
        self.pipeline = self.core.pipeline
        self.model_size = self.core.model_size
        return success
    
    def unload_model(self):
        """Unload the current model."""
        self.core.unload_model()
        self.pipeline = self.core.pipeline  # Update reference
    
    def test_wan_setup(self) -> bool:
        """Test WAN setup."""
        return self.core.test_wan_setup()
    
    # ============================================================================
    # Configuration and Validation (delegated to ConfigHandler)
    # ============================================================================
    
    def _validate_vace_weights(self, model_path: Path) -> bool:
        """Validate VACE model weights."""
        return self.config_handler.validate_vace_weights(model_path)
    
    def _has_incomplete_models(self) -> bool:
        """Check for incomplete models."""
        return self.config_handler.has_incomplete_models()
    
    def _check_for_incomplete_models(self) -> List[Path]:
        """Check for incomplete model downloads."""
        return self.config_handler.check_for_incomplete_models()
    
    def _fix_incomplete_model(self, model_dir: Path, downloader=None) -> bool:
        """Fix incomplete model."""
        return self.config_handler.fix_incomplete_model(model_dir, downloader)
    
    def save_wan_settings_and_metadata(self, output_dir: str, timestring: str, clips: List[Dict], 
                                     model_info: Dict, wan_args=None, **kwargs) -> str:
        """Save WAN settings and metadata."""
        return self.config_handler.save_wan_settings_and_metadata(
            output_dir, timestring, clips, model_info, wan_args, **kwargs
        )
    
    def create_wan_srt_file(self, output_dir: str, timestring: str, clips: List[Dict], 
                          fps: float = 8.0) -> str:
        """Create SRT subtitle file."""
        return self.config_handler.create_wan_srt_file(output_dir, timestring, clips, fps)
    
    def download_and_cache_audio(self, audio_url: str, output_dir: str, timestring: str) -> str:
        """Download and cache audio."""
        return self.config_handler.download_and_cache_audio(audio_url, output_dir, timestring)
    
    # ============================================================================
    # Video Generation (delegated to PipelineManager)
    # ============================================================================
    
    def generate_video_with_i2v_chaining(self, clips, model_info, output_dir, wan_args=None, **kwargs):
        """Generate video with I2V chaining."""
        return self.pipeline_manager.generate_video_with_i2v_chaining(
            clips, model_info, output_dir, wan_args, **kwargs
        )
    
    # ============================================================================
    # Utility Functions (delegated to Utilities)
    # ============================================================================
    
    def _process_and_save_frames(self, result, clip_idx, output_dir, timestring, start_frame_idx, frame_progress=None):
        """Process and save frames."""
        return self.utilities.process_and_save_frames(
            result, clip_idx, output_dir, timestring, start_frame_idx, frame_progress
        )
    
    # ============================================================================
    # Private Helper Methods (maintained for compatibility)
    # ============================================================================
    
    def _analyze_model_directory(self, model_dir: Path) -> Optional[Dict]:
        """Analyze model directory (delegated to pipeline manager)."""
        return self.pipeline_manager._analyze_model_directory(model_dir)
    
    def _has_required_files(self, model_dir: Path) -> bool:
        """Check required files (delegated to pipeline manager)."""
        return self.pipeline_manager._has_required_files(model_dir)
    
    def _time_to_srt_format(self, seconds) -> str:
        """Convert time to SRT format (delegated to config handler)."""
        return self.config_handler._time_to_srt_format(seconds)
    
    def _get_deforum_version(self) -> str:
        """Get Deforum version (delegated to config handler)."""
        return self.config_handler._get_deforum_version()


# ============================================================================
# Standalone Functions (maintained for backward compatibility)
# ============================================================================

# Re-export the main generation function
__all__ = ['WanSimpleIntegration', 'wan_generate_video_main', 'get_webui_root'] 