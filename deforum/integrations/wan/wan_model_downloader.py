#!/usr/bin/env python3
"""
Wan Model Downloader
Automatically downloads Wan models from HuggingFace when needed
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, List
import time
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()


class WanModelDownloader:
    """Handles automatic downloading of Wan models"""
    
    def __init__(self):
        # Dynamically detect the models directory
        self.models_dir = self._detect_models_directory()
        self.available_models = {
            # Official Wan-AI Models (Diffusers format - Compatible with Deforum)
            "TI2V-5B": {
                "repo_id": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                "local_dir": str(self.models_dir / "Wan2.2-TI2V-5B"),
                "description": "Wan 2.2 TI2V-5B (Unified T2V + I2V, 720P@24fps, Works with 16GB VRAM using CPU offload)",
                "size_gb": 30,
                "quantization": "FP16",
                "vram": "~16GB (with CPU offload) / ~24GB (full GPU)",
                "source": "Official",
                "recommended": True
            },
            "TI2V-A14B": {
                "repo_id": "Wan-AI/Wan2.2-TI2V-A14B-Diffusers",
                "local_dir": str(self.models_dir / "Wan2.2-TI2V-A14B"),
                "description": "Wan 2.2 TI2V-A14B (MoE, Unified T2V + I2V, Highest Quality, Requires 24GB+ VRAM)",
                "size_gb": 60,
                "quantization": "FP16",
                "vram": "~32GB",
                "source": "Official",
                "recommended": False
            }
            # Note: Kijai's FP8/GGUF models are ComfyUI format only and incompatible with diffusers pipeline
        }
    
    def _detect_models_directory(self) -> Path:
        """Detect the best models directory for the current installation"""
        # Try to find the webui models directory
        extension_root = Path(__file__).parent.parent.parent.parent
        
        # Option 1: webui/models/Deforum/wan (standard installation)
        webui_models = extension_root.parent.parent / "models" / "wan"
        if webui_models.parent.exists():
            webui_models.mkdir(exist_ok=True)
            return webui_models
        
        # Option 2: Current working directory models/Deforum/wan
        local_models = Path("models/Deforum/wan")
        if local_models.parent.exists() or Path("models").exists():
            local_models.mkdir(parents=True, exist_ok=True)
            return local_models
        
        # Option 3: Extension directory models (fallback)
        extension_models = extension_root / "models" / "wan"
        extension_models.mkdir(parents=True, exist_ok=True)
        return extension_models
    
    def check_huggingface_cli(self) -> bool:
        """Check if huggingface-cli is available"""
        try:
            result = subprocess.run(
                ["huggingface-cli", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def install_huggingface_hub(self) -> bool:
        """Install huggingface_hub if not available"""
        try:
            logger.info("📦 Installing huggingface_hub...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "huggingface_hub"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                logger.info("✅ huggingface_hub installed successfully")
                return True
            else:
                logger.error(f"Failed to install huggingface_hub: {result.stderr}", emoji='off')
                return False
                
        except subprocess.TimeoutExpired:
            logger.info("Installation timed out", emoji='off')
            return False
        except Exception as e:
            logger.error(f"Installation error: {e}", emoji='off')
            return False
    
    def download_model(self, model_key: str, progress_callback=None) -> bool:
        """Download a specific model"""
        if model_key not in self.available_models:
            logger.info(f"Unknown model: {model_key}", emoji='off')
            return False

        model_info = self.available_models[model_key]
        local_dir = Path(model_info["local_dir"])

        # Check if model already exists
        if self.is_model_downloaded(model_key):
            logger.info(f"✅ Model {model_key} already exists at {local_dir}")
            if progress_callback:
                progress_callback(f"✅ Model already downloaded at {local_dir}")
            return True

        # Install huggingface_hub if needed
        if not self.check_huggingface_cli():
            logger.warning("⚠️ huggingface_hub not found, installing...")
            if progress_callback:
                progress_callback("📦 Installing huggingface_hub...")
            if not self.install_huggingface_hub():
                error_msg = "❌ Failed to install huggingface_hub"
                logger.info(error_msg)
                if progress_callback:
                    progress_callback(error_msg)
                return False

        logger.info(f"📥 Downloading {model_key} ({model_info['description']})...")
        logger.info(f"   📂 From: {model_info['repo_id']}")
        logger.info(f"   📁 To: {local_dir}")
        logger.info(f"   💾 Size: ~{model_info['size_gb']}GB")

        if progress_callback:
            progress_callback(f"📥 Downloading {model_key} from {model_info['repo_id']}\nSize: ~{model_info['size_gb']}GB\nThis may take a while...")

        # Create directory
        local_dir.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Work around Forge's huggingface_hub monkey-patching issues
            # Use CLI with ulimit workaround instead
            import resource

            # Get current file descriptor limit
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            logger.info(f"Current file descriptor limit: {soft_limit}/{hard_limit}", emoji='distribution')

            # Temporarily increase soft limit to hard limit
            try:
                new_limit = min(hard_limit, 8192)  # Cap at 8192 for safety
                resource.setrlimit(resource.RLIMIT_NOFILE, (new_limit, hard_limit))
                logger.info(f"✅ Temporarily increased file descriptor limit to {new_limit}")
                if progress_callback:
                    progress_callback(f"🔧 Increased file descriptor limit to {new_limit} for download")
            except Exception as limit_e:
                logger.error(f"⚠️ Could not increase file limit: {limit_e}")
                # Continue anyway

            # Use CLI with increased limits
            cmd = [
                "huggingface-cli", "download",
                model_info["repo_id"],
                "--local-dir", str(local_dir),
                "--local-dir-use-symlinks", "False",
                "--resume-download"  # Allow resuming
            ]

            logger.info(f"🚀 Running: {' '.join(cmd)}")
            if progress_callback:
                progress_callback(f"🚀 Starting download (this may take 15-30 minutes)...\nDownloading to: {local_dir}")

            # Run with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # Stream output
            last_progress = ""
            for line in process.stdout:
                line = line.strip()
                if line:
                    logger.info(f"   {line}")
                    # Only update callback every 10 lines to avoid UI spam
                    if "Fetching" in line or "Downloading" in line or "%" in line:
                        last_progress = line
                        if progress_callback:
                            progress_callback(f"📥 {line}")

            process.wait()

            # Restore original file descriptor limit
            try:
                resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))
                logger.info(f"✅ Restored file descriptor limit to {soft_limit}")
            except:
                pass

            if process.returncode == 0:
                success_msg = f"✅ Successfully downloaded {model_key} to {local_dir}"
                logger.info(success_msg)
                if progress_callback:
                    progress_callback(success_msg)
                return True
            else:
                error_msg = f"❌ Download failed with return code {process.returncode}"
                logger.info(error_msg)
                if progress_callback:
                    progress_callback(error_msg)
                return False

        except Exception as e:
            error_msg = f"❌ Download error: {e}"
            logger.info(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            import traceback
            traceback.print_exc()
            return False
    
    def is_model_downloaded(self, model_key: str) -> bool:
        """Check if a model is already downloaded"""
        if model_key not in self.available_models:
            return False
        
        model_info = self.available_models[model_key]
        local_dir = Path(model_info["local_dir"])
        
        # Check for essential files
        required_files = [
            "config.json"
        ]
        
        for file in required_files:
            if not (local_dir / file).exists():
                return False
        
        # Check for diffusion model files (single file OR multi-part)
        single_diffusion_file = local_dir / "diffusion_pytorch_model.safetensors"
        multi_part_index = local_dir / "diffusion_pytorch_model.safetensors.index.json"
        
        if single_diffusion_file.exists():
            # Single file model (1.3B)
            pass
        elif multi_part_index.exists():
            # Multi-part model (14B) - verify at least some files exist
            multi_part_files = list(local_dir.glob("diffusion_pytorch_model-*-of-*.safetensors"))
            if not multi_part_files:
                return False
        else:
            # No diffusion model found
            return False
        
        return True
    
    def get_model_path(self, model_key: str) -> Optional[str]:
        """Get the local path for a downloaded model"""
        if not self.is_model_downloaded(model_key):
            return None
        
        model_info = self.available_models[model_key]
        return str(Path(model_info["local_dir"]).absolute())
    
    def list_available_models(self) -> List[Dict]:
        """List all available models with download status"""
        models = []
        for key, info in self.available_models.items():
            models.append({
                "key": key,
                "description": info["description"],
                "size_gb": info["size_gb"],
                "downloaded": self.is_model_downloaded(key),
                "path": self.get_model_path(key) if self.is_model_downloaded(key) else None
            })
        return models
    
    def auto_download_recommended(self) -> Dict[str, str]:
        """Auto-download recommended model (TI2V-5B with CPU offload for 16GB VRAM)"""
        results = {}

        # Download TI2V-5B (Recommended - works with 16GB VRAM using aggressive CPU offload)
        logger.info("🎯 Auto-downloading Wan 2.2 TI2V-5B (Unified T2V+I2V, 720p@24fps)")
        logger.info("This model works with 16GB VRAM using automatic CPU offload optimizations", emoji='bulb')
        if self.download_model("TI2V-5B"):
            model_path = self.get_model_path("TI2V-5B")
            results["t2v"] = model_path
            results["i2v"] = model_path  # TI2V handles both T2V and I2V
            logger.info("✅ TI2V-5B ready for unified T2V and I2V generation")
            logger.info("Will auto-enable CPU offload, attention slicing, and VAE optimizations for 16GB VRAM", emoji='wrench')
        else:
            logger.error("Failed to download TI2V-5B model", emoji='off')

        return results
    
    def download_by_preference(self, prefer_size: str = "TI2V-5B", download_i2v: bool = False) -> Dict[str, str]:
        """Download Wan 2.2 TI2V model based on size preference"""
        results = {}

        # Determine which TI2V model to download (Wan 2.2 only)
        if "A14B" in prefer_size or "14B" in prefer_size:
            primary_model = "TI2V-A14B"
            logger.info("✅ Wan 2.2 TI2V-A14B: MoE, unified T2V+I2V, 720p, highest quality")
        else:
            # Default to TI2V-5B (recommended)
            primary_model = "TI2V-5B"
            logger.info("✅ Wan 2.2 TI2V-5B: Unified T2V+I2V, 720p@24fps, RTX 4090")

        # Download the TI2V model
        logger.info(f"📥 Downloading {primary_model}...")
        if self.download_model(primary_model):
            model_path = self.get_model_path(primary_model)
            results["t2v"] = model_path
            results["i2v"] = model_path  # TI2V handles both T2V and I2V
            logger.info(f"✅ {primary_model} ready for unified T2V and I2V generation")
        else:
            logger.error(f"Failed to download {primary_model}", emoji='off')

        return results

def download_wan_model(model_key: str) -> bool:
    """Convenience function to download a single model"""
    downloader = WanModelDownloader()
    return downloader.download_model(model_key)

def auto_setup_wan_models(prefer_size: str = "TI2V-5B") -> Dict[str, str]:
    """Auto-setup Wan models with size preference"""
    downloader = WanModelDownloader()
    return downloader.download_by_preference(prefer_size)

if __name__ == "__main__":
    # Test the downloader
    downloader = WanModelDownloader()
    
    logger.info("🔍 Available models:")
    for model in downloader.list_available_models():
        status = "✅ Downloaded" if model["downloaded"] else "❌ Not downloaded"
        logger.info(f"   {model['key']}: {model['description']} ({model['size_gb']}GB) - {status}")
    
    # Test download (uncomment to actually download)
    # print("\n📥 Testing download...")
    # downloader.download_model("1.3B T2V") 