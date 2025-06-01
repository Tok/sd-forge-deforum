#!/usr/bin/env python3
"""
WAN Configuration Handler
Handles WAN configuration, settings management, and metadata operations
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import os
import json
from decimal import Decimal
import time


class WanConfigHandler:
    """Handles WAN configuration and metadata management"""
    
    def __init__(self, core_integration):
        """Initialize configuration handler.
        
        Args:
            core_integration: WanCoreIntegration instance
        """
        self.core = core_integration
    
    def validate_vace_weights(self, model_path: Path) -> bool:
        """Validate that VACE model has required weights.
        
        Args:
            model_path: Path to VACE model directory
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            self.core.print_wan_info(f"🔍 Validating VACE model: {model_path.name}")
            
            # Check if the main model files exist
            diffusion_model = model_path / "diffusion_pytorch_model.safetensors"
            if not diffusion_model.exists():
                # Check for multi-part model
                diffusion_model = model_path / "diffusion_pytorch_model-00001-of-00007.safetensors"
                if not diffusion_model.exists():
                    self.core.print_wan_error("❌ No diffusion model file found for VACE validation")
                    return False
            
            # Basic file size check
            if diffusion_model.stat().st_size < 1_000_000:  # Less than 1MB is suspicious
                self.core.print_wan_error(f"❌ VACE model file too small: {diffusion_model.stat().st_size} bytes")
                return False
            
            # Check for required config files
            config_file = model_path / "config.json"
            if not config_file.exists():
                self.core.print_wan_error("❌ VACE model missing config.json")
                return False
            
            # Try to load and validate config
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                # Check if it's actually a VACE model
                model_type = config.get("model_type", "").lower()
                class_name = config.get("_class_name", "").lower()
                
                if "vace" not in model_type and "vace" not in class_name:
                    self.core.print_wan_warning(f"⚠️ Model config doesn't indicate VACE type: {model_type}, {class_name}")
                    # Don't fail - might still be VACE with different naming
                
                self.core.print_wan_success(f"✅ VACE model validation passed: {model_path.name}")
                return True
                
            except Exception as config_e:
                self.core.print_wan_warning(f"⚠️ VACE config validation failed: {config_e}")
                # Don't fail completely - file might still be valid
                return True
            
        except Exception as e:
            self.core.print_wan_error(f"❌ VACE validation error: {e}")
            return False
    
    def has_incomplete_models(self) -> bool:
        """Check if there are incomplete models.
        
        Returns:
            True if incomplete models found, False otherwise
        """
        try:
            incomplete_models = self.check_for_incomplete_models()
            return len(incomplete_models) > 0
        except Exception as e:
            self.core.print_wan_warning(f"⚠️ Error checking for incomplete models: {e}")
            return False
    
    def check_for_incomplete_models(self) -> List[Path]:
        """Check for incomplete model downloads.
        
        Returns:
            List of paths to incomplete models
        """
        try:
            incomplete_models = []
            
            # Check WebUI models directory
            search_paths = self.core.get_model_search_paths()
            
            for search_path in search_paths:
                if not search_path.exists():
                    continue
                
                for model_dir in search_path.iterdir():
                    if not model_dir.is_dir():
                        continue
                    
                    if model_dir.name.startswith(('Wan2.1', 'wan')):
                        # Check if model is incomplete
                        if self._is_model_incomplete(model_dir):
                            incomplete_models.append(model_dir)
            
            return incomplete_models
            
        except Exception as e:
            self.core.print_wan_error(f"❌ Error checking for incomplete models: {e}")
            return []
    
    def _is_model_incomplete(self, model_dir: Path) -> bool:
        """Check if a model directory appears incomplete.
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            True if model appears incomplete, False otherwise
        """
        try:
            # Check for required files
            required_files = ["config.json", "model_index.json"]
            
            for req_file in required_files:
                if not (model_dir / req_file).exists():
                    return True
            
            # Check for model weight files
            weight_files = list(model_dir.rglob("*.safetensors")) + list(model_dir.rglob("*.bin"))
            if not weight_files:
                return True
            
            # Check if any weight files are very small (indicating incomplete download)
            for weight_file in weight_files:
                if weight_file.stat().st_size < 100_000:  # Less than 100KB is suspicious
                    return True
            
            return False
            
        except Exception:
            return True  # Assume incomplete if we can't check
    
    def fix_incomplete_model(self, model_dir: Path, downloader=None) -> bool:
        """Fix incomplete model by re-downloading.
        
        Args:
            model_dir: Path to incomplete model directory
            downloader: Optional downloader function
            
        Returns:
            True if fixed successfully, False otherwise
        """
        try:
            self.core.print_wan_info(f"🔧 Attempting to fix incomplete model: {model_dir.name}")
            
            # For now, just report that manual re-download is needed
            self.core.print_wan_warning(f"⚠️ Model {model_dir.name} appears incomplete")
            self.core.print_wan_info("💡 Please re-download the model using huggingface-cli")
            
            return False  # Manual intervention required
            
        except Exception as e:
            self.core.print_wan_error(f"❌ Error fixing incomplete model: {e}")
            return False
    
    def save_wan_settings_and_metadata(self, output_dir: str, timestring: str, clips: List[Dict], 
                                     model_info: Dict, wan_args=None, **kwargs) -> str:
        """Save WAN generation settings and metadata.
        
        Args:
            output_dir: Output directory
            timestring: Timestamp string
            clips: List of clip dictionaries
            model_info: Model information
            wan_args: WAN arguments
            **kwargs: Additional settings
            
        Returns:
            Path to saved settings file
        """
        try:
            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare metadata
            metadata = {
                "generation_info": {
                    "timestamp": timestring,
                    "clips_count": len(clips),
                    "model_name": model_info.get('name', 'Unknown'),
                    "model_type": model_info.get('type', 'Unknown'),
                    "model_size": model_info.get('size', 'Unknown'),
                    "device": self.core.device,
                    "flash_attention_mode": self.core.flash_attention_mode
                },
                "clips": clips,
                "wan_args": wan_args or {},
                "additional_settings": kwargs,
                "deforum_version": self._get_deforum_version()
            }
            
            # Save metadata to JSON file
            metadata_file = output_path / f"{timestring}_wan_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            self.core.print_wan_success(f"✅ WAN metadata saved: {metadata_file}")
            
            # Import and save Deforum settings if available
            try:
                from ..settings import save_settings_from_animation_run
                save_settings_from_animation_run(output_dir, timestring, wan_args, clips)
                self.core.print_wan_success(f"✅ Deforum settings saved")
            except ImportError:
                self.core.print_wan_warning("⚠️ Could not import Deforum settings saver")
            except Exception as e:
                self.core.print_wan_warning(f"⚠️ Could not save Deforum settings: {e}")
            
            return str(metadata_file)
            
        except Exception as e:
            self.core.print_wan_error(f"❌ Error saving WAN settings and metadata: {e}")
            return ""
    
    def create_wan_srt_file(self, output_dir: str, timestring: str, clips: List[Dict], 
                          fps: float = 8.0) -> str:
        """Create SRT subtitle file for WAN generation.
        
        Args:
            output_dir: Output directory
            timestring: Timestamp string
            clips: List of clip dictionaries
            fps: Frames per second
            
        Returns:
            Path to created SRT file
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            srt_file = output_path / f"{timestring}_wan_subtitles.srt"
            
            self.core.print_wan_info(f"📝 Creating SRT file: {srt_file}")
            
            with open(srt_file, 'w', encoding='utf-8') as f:
                for i, clip in enumerate(clips):
                    # Calculate timing
                    start_frame = clip.get('start_frame', i * 30)  # Default 30 frames per clip
                    end_frame = clip.get('end_frame', start_frame + 30)
                    
                    start_time = self._frame_to_srt_time(start_frame, fps)
                    end_time = self._frame_to_srt_time(end_frame, fps)
                    
                    # Write subtitle entry
                    f.write(f"{i + 1}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{clip.get('prompt', 'Generated video clip')}\n\n")
            
            self.core.print_wan_success(f"✅ SRT file created: {srt_file}")
            return str(srt_file)
            
        except Exception as e:
            self.core.print_wan_error(f"❌ Error creating SRT file: {e}")
            return ""
    
    def _frame_to_srt_time(self, frame: int, fps: float) -> str:
        """Convert frame number to SRT time format.
        
        Args:
            frame: Frame number
            fps: Frames per second
            
        Returns:
            Time in SRT format (HH:MM:SS,mmm)
        """
        seconds = Decimal(frame) / Decimal(fps)
        return self._time_to_srt_format(seconds)
    
    def _time_to_srt_format(self, seconds: Decimal) -> str:
        """Convert seconds to SRT time format.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Time in SRT format (HH:MM:SS,mmm)
        """
        total_seconds = int(seconds)
        milliseconds = int((seconds - total_seconds) * 1000)
        
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _get_deforum_version(self) -> str:
        """Get Deforum version information.
        
        Returns:
            Version string
        """
        try:
            # Try to get version from package info
            version_file = Path(__file__).parent.parent.parent / "VERSION"
            if version_file.exists():
                return version_file.read_text().strip()
            
            # Fallback to git info if available
            git_dir = Path(__file__).parent.parent.parent.parent / ".git"
            if git_dir.exists():
                return "git-development"
            
            return "unknown"
            
        except Exception:
            return "unknown"
    
    def download_and_cache_audio(self, audio_url: str, output_dir: str, timestring: str) -> str:
        """Download and cache audio file for video generation.
        
        Args:
            audio_url: URL to audio file
            output_dir: Output directory
            timestring: Timestamp string
            
        Returns:
            Path to downloaded audio file
        """
        try:
            if not audio_url:
                return ""
            
            self.core.print_wan_info(f"🎵 Downloading audio: {audio_url}")
            
            # Import video audio utilities
            try:
                from ..video_audio_utilities import download_audio
                audio_path = download_audio(audio_url, output_dir, timestring)
                
                if audio_path:
                    self.core.print_wan_success(f"✅ Audio downloaded: {audio_path}")
                    return audio_path
                else:
                    self.core.print_wan_warning("⚠️ Audio download failed")
                    return ""
                    
            except ImportError:
                self.core.print_wan_warning("⚠️ Audio download utilities not available")
                return ""
            
        except Exception as e:
            self.core.print_wan_error(f"❌ Error downloading audio: {e}")
            return ""
    
    def create_model_config(self, model_info: Dict) -> Dict:
        """Create configuration dictionary for model.
        
        Args:
            model_info: Model information dictionary
            
        Returns:
            Configuration dictionary
        """
        try:
            config = {
                "model_name": model_info.get('name', 'Unknown'),
                "model_type": model_info.get('type', 'Unknown'),
                "model_size": model_info.get('size', 'Unknown'),
                "model_path": model_info.get('path', ''),
                "device": self.core.device,
                "flash_attention_mode": self.core.flash_attention_mode,
                "optimal_resolution": self.core.get_optimal_resolution(),
                "supported_features": {
                    "text_to_video": True,
                    "image_to_video": model_info.get('type') in ['VACE', 'I2V'],
                    "i2v_chaining": model_info.get('type') in ['VACE', 'I2V']
                }
            }
            
            return config
            
        except Exception as e:
            self.core.print_wan_error(f"❌ Error creating model config: {e}")
            return {}
    
    def validate_generation_args(self, clips: List[Dict], **kwargs) -> tuple:
        """Validate generation arguments.
        
        Args:
            clips: List of clip dictionaries
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not clips:
                return False, "No clips provided for generation"
            
            for i, clip in enumerate(clips):
                if not clip.get('prompt'):
                    return False, f"Clip {i+1} missing prompt"
                
                # Validate frame counts
                num_frames = clip.get('num_frames', 30)
                if num_frames <= 0 or num_frames > 1000:
                    return False, f"Clip {i+1} has invalid frame count: {num_frames}"
                
                # Validate dimensions
                width = clip.get('width', 720)
                height = clip.get('height', 480)
                if width <= 0 or height <= 0 or width > 4096 or height > 4096:
                    return False, f"Clip {i+1} has invalid dimensions: {width}x{height}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Validation error: {e}" 