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

class WanModelDownloader:
    """Handles automatic downloading of Wan models"""
    
    def __init__(self):
        self.models_dir = Path("models/wan")
        self.available_models = {
            "1.3B T2V": {
                "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
                "local_dir": str(self.models_dir / "Wan2.1-T2V-1.3B"),
                "description": "1.3B Text-to-Video model (480P)",
                "size_gb": 8
            },
            "14B T2V": {
                "repo_id": "Wan-AI/Wan2.1-T2V-14B", 
                "local_dir": str(self.models_dir / "Wan2.1-T2V-14B"),
                "description": "14B Text-to-Video model (480P/720P)",
                "size_gb": 75
            },
            "14B I2V 720P": {
                "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
                "local_dir": str(self.models_dir / "Wan2.1-I2V-14B-720P"),
                "description": "14B Image-to-Video model (720P)",
                "size_gb": 75
            },
            "14B I2V 480P": {
                "repo_id": "Wan-AI/Wan2.1-I2V-14B-480P", 
                "local_dir": str(self.models_dir / "Wan2.1-I2V-14B-480P"),
                "description": "14B Image-to-Video model (480P)",
                "size_gb": 75
            }
        }
    
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
            print("📦 Installing huggingface_hub...")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "huggingface_hub"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print("✅ huggingface_hub installed successfully")
                return True
            else:
                print(f"❌ Failed to install huggingface_hub: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Installation timed out")
            return False
        except Exception as e:
            print(f"❌ Installation error: {e}")
            return False
    
    def download_model(self, model_key: str, progress_callback=None) -> bool:
        """Download a specific model"""
        if model_key not in self.available_models:
            print(f"❌ Unknown model: {model_key}")
            return False
        
        model_info = self.available_models[model_key]
        local_dir = Path(model_info["local_dir"])
        
        # Check if model already exists
        if self.is_model_downloaded(model_key):
            print(f"✅ Model {model_key} already exists at {local_dir}")
            return True
        
        # Ensure huggingface-cli is available
        if not self.check_huggingface_cli():
            print("⚠️ huggingface-cli not found, installing huggingface_hub...")
            if not self.install_huggingface_hub():
                print("❌ Failed to install huggingface_hub")
                return False
        
        print(f"📥 Downloading {model_key} ({model_info['description']})...")
        print(f"   📂 From: {model_info['repo_id']}")
        print(f"   📁 To: {local_dir}")
        print(f"   💾 Size: ~{model_info['size_gb']}GB")
        
        # Create directory
        local_dir.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use huggingface-cli to download
            cmd = [
                "huggingface-cli", "download",
                model_info["repo_id"],
                "--local-dir", str(local_dir),
                "--local-dir-use-symlinks", "False"
            ]
            
            print(f"🚀 Running: {' '.join(cmd)}")
            
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
            for line in process.stdout:
                line = line.strip()
                if line:
                    print(f"   {line}")
                    if progress_callback:
                        progress_callback(line)
            
            process.wait()
            
            if process.returncode == 0:
                print(f"✅ Successfully downloaded {model_key}")
                return True
            else:
                print(f"❌ Download failed with return code {process.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ Download error: {e}")
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
        """Auto-download recommended models (1.3B T2V)"""
        results = {}
        
        # Download 1.3B T2V as primary model
        print("🎯 Auto-downloading recommended model: 1.3B T2V")
        if self.download_model("1.3B T2V"):
            results["t2v"] = self.get_model_path("1.3B T2V")
            results["i2v"] = results["t2v"]  # Use same model for I2V (no 1.3B I2V exists)
            print("⚠️ Warning: Using T2V model for I2V (no 1.3B I2V model exists). This will break continuity.")
        else:
            print("❌ Failed to download recommended model")
        
        return results
    
    def download_by_preference(self, prefer_size: str = "1.3B", download_i2v: bool = False) -> Dict[str, str]:
        """Download models based on size preference"""
        results = {}
        
        # Determine which models to download
        if prefer_size.startswith("1.3B"):
            t2v_model = "1.3B T2V"
            if download_i2v:
                print("⚠️ WARNING: No 1.3B I2V model exists!")
                print("💡 RECOMMENDATION: Either use 1.3B T2V only (T2V for all clips)")
                print("                   OR upgrade to 14B T2V + 14B I2V for real I2V chaining")
                print("🔄 Will use 1.3B T2V for both T2V and I2V (breaks continuity)")
                i2v_model = None  # Will use T2V
            else:
                i2v_model = None
        else:
            t2v_model = "14B T2V"
            i2v_model = "14B I2V 720P" if download_i2v else None
            if download_i2v:
                print("✅ Using proper 14B T2V + 14B I2V combination for best results")
        
        # Download T2V model
        print(f"📥 Downloading T2V model: {t2v_model}")
        if self.download_model(t2v_model):
            results["t2v"] = self.get_model_path(t2v_model)
        
        # Download I2V model if requested and exists
        if i2v_model:
            print(f"📥 Downloading I2V model: {i2v_model}")
            if self.download_model(i2v_model):
                results["i2v"] = self.get_model_path(i2v_model)
            else:
                print(f"❌ Failed to download {i2v_model}, will use T2V for I2V")
                results["i2v"] = results["t2v"]
        
        # Use T2V for I2V if no separate I2V model
        if "i2v" not in results and "t2v" in results:
            results["i2v"] = results["t2v"]
            if download_i2v:
                print("⚠️ Using T2V model for I2V - this will break visual continuity between clips")
        
        return results

def download_wan_model(model_key: str) -> bool:
    """Convenience function to download a single model"""
    downloader = WanModelDownloader()
    return downloader.download_model(model_key)

def auto_setup_wan_models(prefer_size: str = "1.3B") -> Dict[str, str]:
    """Auto-setup Wan models with size preference"""
    downloader = WanModelDownloader()
    return downloader.download_by_preference(prefer_size)

if __name__ == "__main__":
    # Test the downloader
    downloader = WanModelDownloader()
    
    print("🔍 Available models:")
    for model in downloader.list_available_models():
        status = "✅ Downloaded" if model["downloaded"] else "❌ Not downloaded"
        print(f"   {model['key']}: {model['description']} ({model['size_gb']}GB) - {status}")
    
    # Test download (uncomment to actually download)
    # print("\n📥 Testing download...")
    # downloader.download_model("1.3B T2V") 