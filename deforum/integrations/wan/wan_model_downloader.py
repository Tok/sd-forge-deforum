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

def get_webui_root() -> Path:
    """Get the WebUI root directory reliably"""
    # Start from this file's location and navigate up to find webui root
    current_path = Path(__file__).resolve()
    
    # Navigate up from extensions/sd-forge-deforum to webui root
    # Path is typically: webui/extensions/sd-forge-deforum/deforum/integrations/wan/wan_model_downloader.py
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

class WanModelDownloader:
    """Handles automatic downloading of Wan models"""
    
    def __init__(self):
        # Detect WebUI root and use proper models directory
        self.webui_root = get_webui_root()
        self.models_dir = self._detect_models_directory()
        self.available_models = {
            "1.3B VACE": {
                "repo_id": "Wan-AI/Wan2.1-VACE-1.3B",
                "local_dir": str(self.models_dir / "Wan2.1-VACE-1.3B"),
                "description": "VACE 1.3B - All-in-One (T2V + I2V + FLF2V)",
                "size_gb": 3.2
            },
            "14B VACE": {
                "repo_id": "Wan-AI/Wan2.1-VACE-14B",
                "local_dir": str(self.models_dir / "Wan2.1-VACE-14B"),
                "description": "VACE 14B - All-in-One (T2V + I2V + FLF2V) - High Quality",
                "size_gb": 28.0
            },
            "1.3B T2V": {
                "repo_id": "Wan-AI/Wan2.1-T2V-1.3B",
                "local_dir": str(self.models_dir / "Wan2.1-T2V-1.3B"),
                "description": "Text-to-Video 1.3B - Fast Generation",
                "size_gb": 3.0
            },
            "14B T2V": {
                "repo_id": "Wan-AI/Wan2.1-T2V-14B",
                "local_dir": str(self.models_dir / "Wan2.1-T2V-14B"),
                "description": "Text-to-Video 14B - High Quality",
                "size_gb": 26.0
            },
            "14B I2V-720P": {
                "repo_id": "Wan-AI/Wan2.1-I2V-14B-720P",
                "local_dir": str(self.models_dir / "Wan2.1-I2V-14B-720P"),
                "description": "Image-to-Video 14B - 720P Resolution",
                "size_gb": 26.0
            },
            "14B I2V-480P": {
                "repo_id": "Wan-AI/Wan2.1-I2V-14B-480P",
                "local_dir": str(self.models_dir / "Wan2.1-I2V-14B-480P"),
                "description": "Image-to-Video 14B - 480P Resolution", 
                "size_gb": 26.0
            },
        }
        
        print(f"🎯 Wan Model Downloader initialized")
        print(f"📁 WebUI root: {self.webui_root}")
        print(f"📁 Models directory: {self.models_dir}")
    
    def _detect_models_directory(self) -> Path:
        """Detect the best models directory for the current installation"""
        # Primary: WebUI models directory
        webui_models = self.webui_root / "models" / "wan"
        
        try:
            # Create the directory if it doesn't exist
            webui_models.mkdir(parents=True, exist_ok=True)
            print(f"✅ Using WebUI models directory: {webui_models}")
            return webui_models
        except Exception as e:
            print(f"⚠️ Cannot create WebUI models directory: {e}")
            
            # Fallback: Current working directory models/wan
            local_models = Path("models/wan")
            try:
                local_models.mkdir(parents=True, exist_ok=True)
                print(f"✅ Using local models directory: {local_models}")
                return local_models
            except Exception as e2:
                print(f"⚠️ Cannot create local models directory: {e2}")
                
                # Final fallback: Extension directory models
                extension_models = Path(__file__).parent.parent.parent / "models" / "wan"
                extension_models.mkdir(parents=True, exist_ok=True)
                print(f"✅ Using extension models directory: {extension_models}")
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
        """Auto-download recommended models (14B VACE)"""
        results = {}
        
        # Download 14B VACE as the new recommended model
        print("🎯 Auto-downloading recommended model: 14B VACE (All-in-One)")
        if self.download_model("14B VACE"):
            model_path = self.get_model_path("14B VACE")
            results["t2v"] = model_path
            results["i2v"] = model_path  # VACE handles both T2V and I2V
            print("✅ VACE model ready for both T2V and I2V generation")
        else:
            print("❌ Failed to download VACE model, trying legacy fallback...")
            # Fallback to legacy models
            if self.download_model("1.3B T2V"):
                results["t2v"] = self.get_model_path("1.3B T2V")
                results["i2v"] = results["t2v"]
                print("⚠️ Warning: Using legacy T2V model for I2V. This will break continuity.")
            else:
                print("❌ Failed to download any models")
        
        return results
    
    def download_by_preference(self, prefer_size: str = "14B", download_i2v: bool = False) -> Dict[str, str]:
        """Download models based on size preference - VACE models are all-in-one"""
        results = {}
        
        # Determine which VACE model to download based on preference
        if prefer_size.startswith("14B") or "14B VACE" in prefer_size:
            primary_model = "14B VACE"
            print("✅ Using 14B VACE - All-in-One model (T2V + I2V, 480P + 720P)")
        elif prefer_size.startswith("1.3B") or "1.3B VACE" in prefer_size:
            primary_model = "1.3B VACE" 
            print("✅ Using 1.3B VACE - All-in-One model (T2V + I2V, 480P)")
        elif "Legacy" in prefer_size:
            # Fallback to legacy models for backwards compatibility
            return self._download_legacy_models(download_i2v)
        else:
            # Default to 14B VACE
            primary_model = "14B VACE"
            print("✅ Defaulting to 14B VACE - All-in-One model (T2V + I2V)")
        
        # Download the VACE model
        print(f"📥 Downloading VACE model: {primary_model}")
        if self.download_model(primary_model):
            model_path = self.get_model_path(primary_model)
            results["t2v"] = model_path
            results["i2v"] = model_path  # VACE handles both T2V and I2V
            print(f"✅ VACE model downloaded and ready for both T2V and I2V")
        else:
            print(f"❌ Failed to download {primary_model}, trying legacy fallback...")
            return self._download_legacy_models(download_i2v)
        
        return results
    
    def _download_legacy_models(self, download_i2v: bool = False) -> Dict[str, str]:
        """Fallback to legacy separate T2V/I2V models"""
        results = {}
        print("🔄 Using legacy separate T2V/I2V models...")
        
        # Download legacy T2V
        t2v_model = "14B T2V"
        print(f"📥 Downloading legacy T2V model: {t2v_model}")
        if self.download_model(t2v_model):
            results["t2v"] = self.get_model_path(t2v_model)
        
        # Download legacy I2V if requested
        if download_i2v:
            i2v_model = "14B I2V-720P"
            print(f"📥 Downloading legacy I2V model: {i2v_model}")
            if self.download_model(i2v_model):
                results["i2v"] = self.get_model_path(i2v_model)
            else:
                print("❌ Failed to download legacy I2V, using T2V for I2V")
                results["i2v"] = results.get("t2v")
        
        # Use T2V for I2V if no separate I2V downloaded
        if "i2v" not in results and "t2v" in results:
            results["i2v"] = results["t2v"]
            print("⚠️ Using legacy T2V model for I2V - may break continuity")
        
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