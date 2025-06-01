#!/usr/bin/env python3
"""
WAN Core Integration
Core WAN integration functionality including initialization and basic model management
"""

from pathlib import Path
from typing import List, Dict, Optional
import torch
import os
import time

def get_webui_root() -> Path:
    """Get the WebUI root directory reliably"""
    # Start from this file's location and navigate up to find webui root
    current_path = Path(__file__).resolve()
    
    # Navigate up from extensions/sd-forge-deforum to webui root
    # Path is typically: webui/extensions/sd-forge-deforum/deforum/integrations/wan/wan_core_integration.py
    # So we go up 6 levels: wan -> integrations -> deforum -> sd-forge-deforum -> extensions -> webui
    webui_root = current_path.parent.parent.parent.parent.parent.parent
    
    # Validate that this looks like a webui directory
    expected_webui_files = ['launch.py', 'webui.py', 'modules']
    if all((webui_root / item).exists() for item in expected_webui_files if isinstance(item, str)) and (webui_root / 'modules').is_dir():
        return webui_root
    
    # Fallback: try to find webui root by looking for characteristic files
    search_path = current_path
    for _ in range(10):  # Max 10 levels up
        search_path = search_path.parent
        if all((search_path / item).exists() for item in expected_webui_files if isinstance(item, str)) and (search_path / 'modules').is_dir():
            return search_path
        if search_path.parent == search_path:  # Reached filesystem root
            break
    
    # Final fallback: use current working directory
    return Path.cwd()


class WanCoreIntegration:
    """Core WAN integration with auto-discovery and model management"""
    
    def __init__(self, device='cuda'):
        """Initialize WAN core integration.
        
        Args:
            device: Device to use for processing ('cuda' or 'cpu')
        """
        # Import progress utilities
        try:
            from .utils.wan_progress_utils import (
                print_wan_info, print_wan_success, print_wan_warning, print_wan_error
            )
            self.print_wan_info = print_wan_info
            self.print_wan_success = print_wan_success
            self.print_wan_warning = print_wan_warning
            self.print_wan_error = print_wan_error
        except ImportError:
            # Fallback printing functions
            self.print_wan_info = print
            self.print_wan_success = print
            self.print_wan_warning = print
            self.print_wan_error = print
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.webui_root = get_webui_root()
        self.models = []
        self.pipeline = None
        self.model_size = None
        self.optimal_width = 720
        self.optimal_height = 480
        self.flash_attention_mode = "auto"  # auto, enabled, disabled
        
        self.print_wan_info(f"WAN Core Integration initialized on {self.device}")
        self.print_wan_info(f"WebUI root detected: {self.webui_root}")
    
    def get_model_search_paths(self) -> List[Path]:
        """Get list of paths to search for WAN models.
        
        Returns:
            List of Path objects to search for models
        """
        webui_models = self.webui_root / "models"
        
        return [
            # PRIMARY: WebUI models directory (standard location)
            webui_models / "wan",
            webui_models / "Wan", 
            webui_models / "wan_models",
            webui_models / "video" / "wan",
            
            # Extension directory for development/testing
            Path(__file__).parent.parent.parent / "deforum" / "integrations" / "external_repos" / "wan2.1",
        ]
    
    def get_best_model(self) -> Optional[Dict]:
        """Get the best available model based on priority.
        
        Returns:
            Best model info dict or None if no models available
        """
        if not self.models:
            return None
        
        # Priority: VACE > T2V > I2V, and 14B > 1.3B
        def model_priority(model):
            type_priority = {'VACE': 3, 'T2V': 2, 'I2V': 1}.get(model['type'], 0)
            size_priority = {'14B': 2, '1.3B': 1}.get(model['size'], 0)
            return type_priority * 10 + size_priority
        
        return max(self.models, key=model_priority)
    
    def unload_model(self):
        """Unload the current model to free memory."""
        if self.pipeline is not None:
            try:
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Delete pipeline reference
                del self.pipeline
                self.pipeline = None
                
                self.print_wan_info("✅ WAN model unloaded successfully")
                
            except Exception as e:
                self.print_wan_error(f"⚠️ Error unloading WAN model: {e}")
    
    def test_wan_setup(self) -> bool:
        """Test basic WAN setup and model availability.
        
        Returns:
            True if setup is working, False otherwise
        """
        try:
            self.print_wan_info("🧪 Testing WAN setup...")
            
            # Check PyTorch
            if not torch.cuda.is_available() and self.device == 'cuda':
                self.print_wan_warning("⚠️ CUDA not available, falling back to CPU")
                self.device = 'cpu'
            
            # Test model discovery
            from .wan_pipeline_manager import WanPipelineManager
            pipeline_manager = WanPipelineManager(self)
            models = pipeline_manager.discover_models()
            
            if not models:
                self.print_wan_error("❌ No WAN models found")
                return False
            
            best_model = self.get_best_model()
            if not best_model:
                self.print_wan_error("❌ No suitable WAN model found")
                return False
            
            self.print_wan_success(f"✅ WAN setup test passed - found {len(models)} model(s)")
            self.print_wan_success(f"✅ Best model: {best_model['name']}")
            
            return True
            
        except Exception as e:
            self.print_wan_error(f"❌ WAN setup test failed: {e}")
            return False
    
    def get_webui_root_path(self) -> Path:
        """Get the WebUI root path.
        
        Returns:
            Path object pointing to WebUI root
        """
        return self.webui_root
    
    def get_device(self) -> str:
        """Get the current device setting.
        
        Returns:
            Device string ('cuda' or 'cpu')
        """
        return self.device
    
    def set_device(self, device: str) -> None:
        """Set the device for processing.
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        if device == 'cuda' and not torch.cuda.is_available():
            self.print_wan_warning("⚠️ CUDA not available, keeping CPU device")
            return
        
        self.device = device
        self.print_wan_info(f"🔧 Device set to: {self.device}")
    
    def get_flash_attention_mode(self) -> str:
        """Get current flash attention mode.
        
        Returns:
            Flash attention mode ('auto', 'enabled', 'disabled')
        """
        return self.flash_attention_mode
    
    def set_flash_attention_mode(self, mode: str) -> None:
        """Set flash attention mode.
        
        Args:
            mode: Flash attention mode ('auto', 'enabled', 'disabled')
        """
        if mode not in ['auto', 'enabled', 'disabled']:
            self.print_wan_warning(f"⚠️ Invalid flash attention mode: {mode}")
            return
        
        self.flash_attention_mode = mode
        self.print_wan_info(f"🔧 Flash attention mode set to: {mode}")
    
    def get_optimal_resolution(self) -> tuple:
        """Get optimal resolution for current model.
        
        Returns:
            Tuple of (width, height)
        """
        return (self.optimal_width, self.optimal_height)
    
    def set_optimal_resolution(self, width: int, height: int) -> None:
        """Set optimal resolution.
        
        Args:
            width: Optimal width
            height: Optimal height
        """
        self.optimal_width = width
        self.optimal_height = height
        self.print_wan_info(f"🔧 Optimal resolution set to: {width}x{height}")


def wan_generate_video_main(*args, **kwargs):
    """Main entry point for WAN video generation (compatibility function).
    
    This is the main function called by the UI to generate videos.
    It delegates to the appropriate pipeline manager.
    """
    try:
        # Initialize core integration
        core = WanCoreIntegration()
        
        # Import and use the pipeline manager
        from .wan_pipeline_manager import WanPipelineManager
        pipeline_manager = WanPipelineManager(core)
        
        return pipeline_manager.generate_video_main(*args, **kwargs)
        
    except Exception as e:
        print(f"❌ WAN video generation failed: {e}")
        return False 