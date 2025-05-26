#!/usr/bin/env python3
"""
WAN Setup Helper for Deforum Extension
Helps users properly set up WAN integration
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if basic requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda}")
        else:
            print("⚠️ CUDA not available - WAN will be very slow")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    return True

def setup_wan_repository():
    """Set up the WAN repository"""
    print("\n📦 Setting up WAN repository...")
    
    extension_root = Path(__file__).parent
    wan_dir = extension_root / "Wan2.1"
    
    if not wan_dir.exists():
        print("❌ Wan2.1 directory not found!")
        print("💡 Please ensure the WAN repository is cloned in Wan2.1/")
        return False
    
    print(f"✅ WAN repository found: {wan_dir}")
    
    # Install WAN package
    print("📥 Installing WAN package...")
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", str(wan_dir)
        ], capture_output=True, text=True, cwd=wan_dir)
        
        if result.returncode == 0:
            print("✅ WAN package installed successfully")
        else:
            print(f"⚠️ WAN package installation warning: {result.stderr}")
            print("🔄 Continuing anyway - package might be already installed")
    
    except Exception as e:
        print(f"⚠️ Could not install WAN package: {e}")
        print("🔄 Continuing anyway - WAN might still work")
    
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📥 Installing dependencies...")
    
    dependencies = [
        "imageio",
        "imageio-ffmpeg", 
        "easydict",
        "transformers",
        "accelerate",
        "diffusers"
    ]
    
    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✅ {dep} already installed")
        except ImportError:
            print(f"📥 Installing {dep}...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], check=True, capture_output=True)
                print(f"✅ {dep} installed")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {dep}: {e}")
                return False
    
    return True

def setup_flash_attention():
    """Set up flash attention (optional)"""
    print("\n⚡ Setting up Flash Attention (optional for better performance)...")
    
    # Check if flash attention is already available
    try:
        import flash_attn
        print("✅ Flash Attention 2 already available")
        return True
    except ImportError:
        pass
    
    try:
        import flash_attn_interface  
        print("✅ Flash Attention 3 already available")
        return True
    except ImportError:
        pass
    
    print("⚠️ Flash Attention not available")
    print("💡 WAN will use PyTorch native attention (slower but compatible)")
    print("📖 To install Flash Attention manually:")
    print("   pip install flash-attn --no-build-isolation")
    print("   Note: Requires CUDA and proper compilation environment")
    
    return True

def download_models():
    """Help user download WAN models"""
    print("\n🤖 Model Download Helper...")
    
    models_dir = Path.home() / "Documents" / "workspace" / "webui-forge" / "webui" / "models" / "wan"
    
    print(f"💡 Recommended model directory: {models_dir}")
    print("📥 To download WAN models, run:")
    print("   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan")
    print("   OR")
    print("   git lfs clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B models/wan")
    
    if models_dir.exists():
        files = list(models_dir.glob("*"))
        if files:
            print(f"✅ Model directory exists with {len(files)} files")
            
            # Check for key files
            key_files = ["diffusion_pytorch_model.safetensors", "config.json"]
            for key_file in key_files:
                if (models_dir / key_file).exists():
                    print(f"   ✅ {key_file}")
                else:
                    print(f"   ❌ {key_file} missing")
        else:
            print("⚠️ Model directory exists but is empty")
    else:
        print("❌ Model directory does not exist")
    
    return True

def test_setup():
    """Test the WAN setup"""
    print("\n🧪 Testing WAN setup...")
    
    try:
        # Test compatibility layer
        scripts_dir = Path(__file__).parent / "scripts" / "deforum_helpers"
        sys.path.insert(0, str(scripts_dir))
        
        from wan_compatibility import ensure_wan_compatibility
        ensure_wan_compatibility()
        print("✅ Compatibility layer working")
        
        # Test WAN import
        import wan
        print("✅ WAN import successful")
        
        # Test model discovery
        from wan_model_discovery import WanModelDiscovery
        discovery = WanModelDiscovery()
        models = discovery.discover_models()
        
        if models:
            print(f"✅ Found {len(models)} WAN model(s)")
            for model in models:
                print(f"   • {model['name']} ({model['size']})")
        else:
            print("⚠️ No WAN models found - download models first")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 WAN Setup Helper for Deforum Extension")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        return False
    
    # Set up WAN repository  
    if not setup_wan_repository():
        print("\n❌ WAN repository setup failed!")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed!")
        return False
    
    # Set up flash attention (optional)
    setup_flash_attention()
    
    # Download models helper
    download_models()
    
    # Test setup
    if test_setup():
        print("\n" + "=" * 50)
        print("🎉 WAN setup completed successfully!")
        print("✅ WAN is ready to use with Deforum")
        print("\n📋 Next steps:")
        print("   1. Download WAN models if you haven't already")
        print("   2. Test WAN generation in Deforum")
        print("   3. Enjoy creating videos!")
        return True
    else:
        print("\n❌ Setup test failed - please check the errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
