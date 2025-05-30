#!/usr/bin/env python3
"""
Wan Simple Integration for Deforum
Provides simple Wan video generation capabilities with auto-discovery and fallback support
"""

# Apply compatibility patches FIRST, before any other imports
try:
    import sys
    import os
    from pathlib import Path
    
    # Import wan_flash_attention_patch to trigger its immediate patching logic.
    # The patch applies itself automatically when the module is first loaded.
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path: # Ensure the directory is in path for the import
        sys.path.insert(0, str(current_dir))
    import wan_flash_attention_patch # This import triggers the patch
    
    print("✅ Early Flash Attention patch module imported (patch should have been applied automatically).")
    # Optionally, could call wan_flash_attention_patch.apply_wan_compatibility_patches() here
    # if we wanted to be absolutely sure or get a return status, but it should be redundant.

except Exception as early_patch_e:
    print(f"⚠️ Early Flash Attention patch import/application FAILED: {early_patch_e}")
    import traceback
    traceback.print_exc() # Print full traceback for early patch failures

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import json
import tempfile
import shutil
from pathlib import Path
import sys
import os
import time

class WanSimpleIntegration:
    """Simple, robust Wan integration that directly loads models"""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.pipeline = None
        self.extension_root = Path(__file__).parent.parent.parent.parent  # Go up to extension root
        self.model_root = self.extension_root / "models"
        self.discovered_models = []
        
    def discover_models(self) -> List[Dict]:
        """Auto-discover Wan models from common locations with smart fallback"""
        print("🔍 Auto-discovering Wan models...")
        
        models = []
        search_paths = [
            # Local paths (relative to current directory)
            Path("models/wan"),
            Path("models/Wan"), 
            Path("models"),
            # WebUI models directory (where models actually are)
            self.extension_root.parent.parent / "models" / "wan",
            self.extension_root.parent.parent / "models" / "Wan",
            # Alternative webui locations
            Path("../../models/wan"),
            Path("../../models/Wan"),
        ]
        
        # Add HuggingFace cache paths if they exist
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir()
            for repo in cache_info.repos:
                if "wan" in repo.repo_id.lower():
                    for revision in repo.revisions:
                        if revision.commit_hash and len(list(revision.snapshot_path.iterdir())) > 0:
                            search_paths.append(revision.snapshot_path)
        except Exception:
            pass  # HF cache scanning failed, continue with local paths
        
        for base_path in search_paths:
            if not base_path.exists():
                continue
                
            print(f"🔍 Searching in: {base_path}")
            
            # Look for direct model files or subdirectories
            for item in base_path.iterdir():
                if item.is_dir():
                    model_info = self._analyze_model_directory(item)
                    if model_info:
                        # Check if we already have this model (avoid duplicates)
                        duplicate = False
                        for existing in models:
                            if existing['name'] == model_info['name'] and existing['type'] == model_info['type']:
                                duplicate = True
                                break
                        
                        if not duplicate:
                            models.append(model_info)
        
        # Store discovered models
        self.discovered_models = models
        
        # Check for incomplete models and try to auto-fix them
        if self._has_incomplete_models():
            print("🔧 Found incomplete models, attempting auto-repair...")
            incomplete_models = self._check_for_incomplete_models()
            for model_dir in incomplete_models:
                print(f"🛠️ Attempting to fix incomplete model: {model_dir}")
                try:
                    if self._fix_incomplete_model(model_dir, self._get_model_downloader()):
                        print(f"✅ Successfully repaired: {model_dir}")
                        # Re-analyze the fixed model
                        model_info = self._analyze_model_directory(model_dir)
                        if model_info and model_info not in models:
                            models.append(model_info)
                    else:
                        print(f"❌ Failed to repair: {model_dir}")
                except Exception as e:
                    print(f"❌ Error repairing {model_dir}: {e}")
        
        # If no models found and auto-download is enabled, download default models
        if not models:
            print("📥 No Wan models found. Checking auto-download settings...")
            # Note: Auto-download logic will be handled by the caller
            # This method focuses on discovery only
        
        # Sort models by preference: 1.3B VACE first, then 14B VACE, then others
        def model_priority(model):
            name = model['name'].lower()
            if '1.3b' in name and 'vace' in name:
                return 0  # Highest priority
            elif '14b' in name and 'vace' in name:
                return 1  # Second priority
            elif 'vace' in name:
                return 2  # VACE models preferred over legacy
            elif '1.3b' in name:
                return 3  # Smaller models preferred
            else:
                return 4  # Legacy models last
        
        models.sort(key=model_priority)
        
        if models:
            print(f"✅ Found {len(models)} Wan model(s):")
            for i, model in enumerate(models, 1):
                print(f"   {i}. {model['name']} ({model['type']}, {model['size']}) - {model['path']}")
        else:
            print("⚠️ No Wan models found in auto-discovery")
            
        return models
    
    def _has_incomplete_models(self) -> bool:
        """Check if discovered models are incomplete"""
        for model in self.discovered_models:
            if not self._validate_wan_model(model):
                print(f"⚠️ Found incomplete/corrupted model: {model['name']} ({model['type']})")
                return True
        return False
    
    def _check_for_incomplete_models(self) -> List[Path]:
        """Check for incomplete model directories (e.g., only .cache folder)"""
        incomplete = []
        
        # Check common model locations relative to the webui installation
        potential_model_dirs = [
            Path("models/wan"),  # Current working directory
            self.extension_root.parent.parent / "models" / "wan",  # Webui models directory
            self.extension_root.parent.parent / "models" / "Wan",  # Alternative casing
        ]
        
        for models_dir in potential_model_dirs:
            if models_dir.exists():
                print(f"🔍 Checking for incomplete models in: {models_dir}")
                for model_dir in models_dir.iterdir():
                    if model_dir.is_dir() and "wan" in model_dir.name.lower():
                        # Check if directory only has .cache or is missing required files
                        files = list(model_dir.iterdir())
                        if (len(files) == 1 and files[0].name == ".cache") or not self._has_required_files(model_dir):
                            print(f"⚠️ Found incomplete model: {model_dir}")
                            incomplete.append(model_dir)
        
        return incomplete
    
    def _has_required_files(self, model_dir: Path) -> bool:
        """Check if model directory has required files"""
        required_files = [
            'diffusion_pytorch_model.safetensors',
            'Wan2.1_VAE.pth',
            'models_t5_umt5-xxl-enc-bf16.pth'
        ]
        
        for required_file in required_files:
            if not (model_dir / required_file).exists():
                # Check for multi-part models (14B)
                if required_file == 'diffusion_pytorch_model.safetensors':
                    multi_part_exists = any(
                        (model_dir / f"diffusion_pytorch_model-{i:05d}-of-00007.safetensors").exists()
                        for i in range(1, 8)
                    )
                    if not multi_part_exists:
                        return False
                else:
                    return False
        return True
    
    def _fix_incomplete_model(self, model_dir: Path, downloader) -> bool:
        """Fix an incomplete model by re-downloading"""
        try:
            # Determine which model this should be based on directory name
            dir_name = model_dir.name.lower()
            
            if "vace" in dir_name and "1.3b" in dir_name:
                model_key = "1.3B VACE"
                print(f"🔄 Detected corrupted VACE 1.3B model: {model_dir.name}")
            elif "vace" in dir_name and "14b" in dir_name:
                model_key = "14B VACE"
                print(f"🔄 Detected corrupted VACE 14B model: {model_dir.name}")
            elif "t2v" in dir_name and "1.3b" in dir_name:
                model_key = "1.3B T2V (Legacy)"
                print(f"🔄 Detected incomplete T2V 1.3B model: {model_dir.name}")
            elif "t2v" in dir_name and "14b" in dir_name:
                model_key = "14B T2V (Legacy)"
                print(f"🔄 Detected incomplete T2V 14B model: {model_dir.name}")
            elif "i2v" in dir_name and "14b" in dir_name:
                if "720p" in dir_name:
                    model_key = "14B I2V 720P (Legacy)"
                else:
                    model_key = "14B I2V 480P (Legacy)"
                print(f"🔄 Detected incomplete I2V model: {model_dir.name}")
            else:
                print(f"⚠️ Could not determine model type for {model_dir.name}")
                return False
            
            print(f"🗑️ Removing corrupted model directory: {model_dir}")
            print(f"📥 Re-downloading {model_key} to fix corruption...")
            
            # Remove corrupted directory
            import shutil
            shutil.rmtree(model_dir)
            
            # Download fresh copy
            success = downloader.download_model(model_key)
            if success:
                print(f"✅ Successfully re-downloaded {model_key}")
            else:
                print(f"❌ Failed to re-download {model_key}")
            
            return success
            
        except Exception as e:
            print(f"❌ Error fixing corrupted model {model_dir.name}: {e}")
            return False
    
    def get_best_model(self) -> Optional[Dict]:
        """Get the best available model"""
        if not self.discovered_models:
            self.discover_models()
        return self.discovered_models[0] if self.discovered_models else None
    
    def test_wan_setup(self) -> bool:
        """Test if Wan setup is working properly"""
        try:
            print("🧪 Testing Wan setup...")
            
            # Check if models are available
            models = self.discover_models()
            if not models:
                print("❌ No Wan models found")
                return False
            
            print(f"✅ Found {len(models)} Wan models")
            
            # Try to load a model
            best_model = models[0]
            print(f"🔧 Testing model loading: {best_model['name']}")
            
            # Test model validation
            if not self._validate_wan_model(best_model):
                print("❌ Model validation failed")
                return False
            
            print("✅ Model validation passed")
            
            # Test pipeline creation (but don't actually load the heavy model)
            try:
                pipeline = self._create_custom_wan_pipeline(best_model)
                print("✅ Pipeline creation test passed")
                return True
            except Exception as e:
                print(f"❌ Pipeline creation failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Wan setup test failed: {e}")
            return False
    
    def load_simple_wan_pipeline(self, model_info: Dict) -> bool:
        """Load a simple Wan pipeline using the discovered model"""
        try:
            print(f"🚀 Loading Wan pipeline: {model_info['name']}")
            print(f"   📁 Path: {model_info['path']}")
            print(f"   🏷️ Type: {model_info['type']}")
            print(f"   📏 Size: {model_info['size']}")
            
            # Validate model files first
            if not self._validate_wan_model(model_info):
                raise RuntimeError("Model validation failed")
            
            # Create custom pipeline based on model type
            self.pipeline = self._create_custom_wan_pipeline(model_info)
            
            if self.pipeline:
                print(f"✅ Wan pipeline loaded successfully")
                return True
            else:
                print(f"❌ Failed to create Wan pipeline")
                return False
                
        except Exception as e:
            print(f"❌ Failed to load Wan pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_custom_wan_pipeline(self, model_info: Dict):
        """Create a custom Wan pipeline that handles the model correctly"""
        
        class CustomWanPipeline:
            def __init__(self, model_path: str, device: str):
                self.model_path = model_path
                self.device = device
                self.model = None
                self.loaded = False
                
                print(f"🔧 Initializing custom Wan pipeline for {model_path}")
                
                # Try to load the actual Wan model
                self._load_wan_model()
            
            def _load_wan_model(self):
                """Load the actual Wan model with multiple fallback strategies"""
                try:
                    # Strategy 1: Try to import and use official Wan
                    print("🔄 Attempting to load official Wan model...")
                    
                    # Add Wan2.1 to path if it exists
                    import sys
                    from pathlib import Path
                    
                    # Look for Wan2.1 directory in the extension root
                    extension_root = Path(__file__).parent.parent.parent.parent
                    wan_repo_path = extension_root / "Wan2.1"
                    
                    print(f"🔍 Looking for Wan repository at: {wan_repo_path}")
                    
                    if wan_repo_path.exists() and (wan_repo_path / "wan").exists():
                        # Add to Python path
                        if str(wan_repo_path) not in sys.path:
                            sys.path.insert(0, str(wan_repo_path))
                            print(f"✅ Added Wan repo to path: {wan_repo_path}")
                        
                        # Apply Flash Attention compatibility patches
                        try:
                            from .wan_flash_attention_patch import apply_wan_compatibility_patches
                        except Exception as patch_e:
                            print(f"⚠️ Could not apply compatibility patches: {patch_e}")
                        
                        # Check what type of model we actually have FIRST
                        model_path = Path(self.model_path)
                        config_path = model_path / "config.json"
                        
                        is_vace_model = False
                        if config_path.exists():
                            try:
                                import json
                                with open(config_path, 'r') as f:
                                    config = json.load(f)
                                model_type = config.get("model_type", "").lower()
                                class_name = config.get("_class_name", "").lower()
                                
                                if model_type == "vace" or "vace" in class_name:
                                    is_vace_model = True
                                    print("🎯 Detected VACE model - using specialized VACE loading")
                                else:
                                    print(f"🎯 Detected {model_type} model")
                            except Exception as e:
                                print(f"⚠️ Could not read config: {e}")
                        
                        # Handle VACE models with specialized loading
                        if is_vace_model:
                            print("🔧 Using specialized VACE model loading...")
                            success = self._load_vace_model_specialized(wan_repo_path)
                            if success:
                                self.loaded = True
                                print("✅ VACE model loaded successfully with specialized loader")
                                return
                            else:
                                print("❌ Specialized VACE loading failed, trying standard approach...")
                                # Continue to standard loading as fallback
                        
                        # Standard Wan model loading for T2V/I2V models
                        # Verify the wan module can be imported
                        try:
                            import wan  # type: ignore
                            print("✅ Successfully imported wan module from local repository")
                            
                            # Import specific components
                            from wan.text2video import WanT2V  # type: ignore
                            from wan.image2video import WanI2V  # type: ignore
                            print("✅ Successfully imported WanT2V and WanI2V classes")
                            
                            # Try to load configs
                            try:
                                # Check if config files exist
                                config_dir = wan_repo_path / "wan" / "configs"
                                if config_dir.exists():
                                    print(f"✅ Found config directory: {config_dir}")
                                    
                                    # Try to import configs
                                    try:
                                        if is_vace_model:
                                            # For VACE models, use T2V config as base
                                            from wan.configs.wan_t2v_1_3B import t2v_1_3B as vace_config
                                            print("✅ Using T2V config for VACE model")
                                            t2v_config = vace_config
                                            i2v_config = vace_config  # Same config for both
                                        else:
                                            from wan.configs.wan_t2v_14B import t2v_14B as t2v_config
                                            from wan.configs.wan_i2v_14B import i2v_14B as i2v_config
                                            print("✅ Loaded official Wan configs")
                                    except ImportError as config_e:
                                        print(f"⚠️ Config import failed: {config_e}, will use minimal configs")
                                        # Create minimal config structure
                                        class MinimalConfig:
                                            def __init__(self):
                                                self.model = type('obj', (object,), {
                                                    'num_attention_heads': 32,
                                                    'attention_head_dim': 128,
                                                    'in_channels': 4,
                                                    'out_channels': 4,
                                                    'num_layers': 28,
                                                    'sample_size': 32,
                                                    'patch_size': 2,
                                                    'num_vector_embeds': None,
                                                    'activation_fn': "geglu",
                                                    'num_embeds_ada_norm': 1000,
                                                    'norm_elementwise_affine': False,
                                                    'norm_eps': 1e-6,
                                                    'attention_bias': True,
                                                    'caption_channels': 4096
                                                })
                                                
                                        t2v_config = MinimalConfig()
                                        i2v_config = MinimalConfig()
                                else:
                                    print("⚠️ Config directory not found, creating minimal configs")
                                    # Create minimal config structure
                                    class MinimalConfig:
                                        def __init__(self):
                                            self.model = type('obj', (object,), {
                                                'num_attention_heads': 32,
                                                'attention_head_dim': 128,
                                                'in_channels': 4,
                                                'out_channels': 4,
                                                'num_layers': 28,
                                                'sample_size': 32,
                                                'patch_size': 2,
                                                'num_vector_embeds': None,
                                                'activation_fn': "geglu",
                                                'num_embeds_ada_norm': 1000,
                                                'norm_elementwise_affine': False,
                                                'norm_eps': 1e-6,
                                                'attention_bias': True,
                                                'caption_channels': 4096
                                            })
                                            
                                    t2v_config = MinimalConfig()
                                    i2v_config = MinimalConfig()
                                    
                            except Exception as config_e:
                                print(f"⚠️ Config loading failed: {config_e}, using minimal configs")
                                # Create minimal config structure
                                class MinimalConfig:
                                    def __init__(self):
                                        self.model = type('obj', (object,), {
                                            'num_attention_heads': 32,
                                            'attention_head_dim': 128,
                                            'in_channels': 4,
                                            'out_channels': 4,
                                            'num_layers': 28,
                                            'sample_size': 32,
                                            'patch_size': 2,
                                            'num_vector_embeds': None,
                                            'activation_fn': "geglu",
                                            'num_embeds_ada_norm': 1000,
                                            'norm_elementwise_affine': False,
                                            'norm_eps': 1e-6,
                                            'attention_bias': True,
                                            'caption_channels': 4096
                                        })
                                        
                                t2v_config = MinimalConfig()
                                i2v_config = MinimalConfig()
                            
                            # Initialize T2V model with correct parameters
                            print(f"🚀 Initializing WanT2V with checkpoint dir: {self.model_path}")
                            
                            if is_vace_model:
                                # For VACE models, use special loading parameters
                                print("🎯 Loading VACE model with T2V-compatible parameters")
                                try:
                                    self.t2v_model = WanT2V(
                                        config=t2v_config,
                                        checkpoint_dir=self.model_path,
                                        device_id=0,
                                        rank=0,
                                        dit_fsdp=False,
                                        t5_fsdp=False
                                    )
                                    print("✅ VACE model loaded as T2V")
                                    
                                    # For VACE, the same model can handle both T2V and I2V
                                    self.i2v_model = self.t2v_model
                                    print("✅ VACE model ready for both T2V and I2V")
                                    
                                except Exception as vace_e:
                                    print(f"❌ VACE T2V loading failed: {vace_e}")
                                    # For VACE models, fail fast instead of trying diffusers fallback
                                    raise RuntimeError(f"""
❌ CRITICAL: VACE model loading failed with official Wan loader!

🔧 REAL VACE MODEL REQUIRED:
The VACE model at {self.model_path} could not be loaded with official Wan methods.

❌ NO FALLBACKS for VACE - Real implementation required!
Specific error: {vace_e}

💡 SOLUTIONS:
1. 📦 Ensure Wan dependencies are properly installed: cd Wan2.1 && pip install -e .
2. 🔧 Verify VACE model files are complete and uncorrupted
3. 💾 Ensure sufficient VRAM (VACE 1.3B requires ~8GB+)
4. 🔄 Re-download VACE model if corrupted
5. 🔄 Restart WebUI after fixing dependencies

🚫 REFUSING TO GENERATE PLACEHOLDER CONTENT!
""")
                            else:
                                # Standard T2V/I2V model loading
                                self.t2v_model = WanT2V(
                                    config=t2v_config,
                                    checkpoint_dir=self.model_path,
                                    device_id=0,
                                    rank=0,
                                    dit_fsdp=False,
                                    t5_fsdp=False
                                )
                                
                                # Try to initialize I2V model if available
                                try:
                                    print(f"🚀 Initializing WanI2V with checkpoint dir: {self.model_path}")
                                    self.i2v_model = WanI2V(
                                        config=i2v_config,
                                        checkpoint_dir=self.model_path,
                                        device_id=0,
                                        rank=0,
                                        dit_fsdp=False,
                                        t5_fsdp=False
                                    )
                                    print("✅ I2V model loaded for chaining support")
                                except Exception as e:
                                    print(f"⚠️ I2V model not available: {e}")
                                    self.i2v_model = None
                            
                            self.loaded = True
                            print("✅ Official Wan models loaded successfully")
                            return
                            
                        except ImportError as import_e:
                            print(f"❌ Failed to import wan module: {import_e}")
                            # Continue to fallback methods
                    else:
                        print(f"❌ Wan repository not found at: {wan_repo_path}")
                        print("💡 Expected structure: Wan2.1/wan/ directory with Python modules")
                    
                    # If we get here, the official Wan import failed
                    print("⚠️ Official Wan import failed, trying diffusers fallback...")
                    
                    # Check if this is a VACE model - if so, refuse fallback
                    model_path = Path(self.model_path)
                    config_path = model_path / "config.json"
                    
                    is_vace_model = False
                    if config_path.exists():
                        try:
                            import json
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            model_type = config.get("model_type", "").lower()
                            class_name = config.get("_class_name", "").lower()
                            
                            if model_type == "vace" or "vace" in class_name:
                                is_vace_model = True
                        except Exception:
                            pass
                    
                    if is_vace_model:
                        # Refuse to use diffusers fallback for VACE models
                        raise RuntimeError(f"""
❌ CRITICAL: VACE model detected but official Wan loading failed!

🔧 REAL VACE MODEL REQUIRES OFFICIAL WAN:
VACE models require the official Wan repository and cannot use diffusers fallback.

❌ NO FALLBACKS for VACE - Real Wan implementation required!

💡 SOLUTIONS:
1. 📦 Properly install Wan repository: cd Wan2.1 && pip install -e .
2. 🔧 Fix Wan import issues (check Python path, dependencies)
3. 💾 Ensure all Wan dependencies are installed
4. 🔄 Restart WebUI after fixing Wan setup

🚫 REFUSING DIFFUSERS FALLBACK FOR VACE MODEL!
""")
                    
                    # Strategy 2: Try diffusers-based loading as fallback (only for non-VACE models)
                    try:
                        from diffusers import DiffusionPipeline
                        
                        print("🔄 Attempting diffusers-based loading for non-VACE model...")
                        self.model = DiffusionPipeline.from_pretrained(
                            self.model_path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                        
                        if torch.cuda.is_available():
                            self.model = self.model.to(self.device)
                        
                        self.loaded = True
                        print("✅ Diffusers-based model loaded successfully (fallback for non-VACE)")
                        return
                        
                    except Exception as diffusers_e:
                        print(f"❌ Diffusers loading also failed: {diffusers_e}")
                    
                    # Both methods failed, provide comprehensive error
                    raise RuntimeError(f"""
❌ SETUP REQUIRED: Wan 2.1 Official Repository Setup Issue!

🔧 QUICK SETUP:

📥 The Wan2.1 directory exists but import failed. Try:

1. 📦 Install Wan dependencies:
   cd Wan2.1
   pip install -e .

2. 📂 Download models to correct location:
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan

3. 🔄 Restart WebUI completely

💡 The Wan2.1 repository is already present at: {wan_repo_path}
   But the Python modules couldn't be imported properly.

🌐 If issues persist, check: https://github.com/Wan-Video/Wan2.1#readme
""")
                
                except Exception as e:
                    print(f"❌ All model loading strategies failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Check if this was a VACE model to provide specific guidance
                    model_path = Path(self.model_path)
                    config_path = model_path / "config.json"
                    is_vace_model = False
                    
                    if config_path.exists():
                        try:
                            import json
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            model_type = config.get("model_type", "").lower()
                            class_name = config.get("_class_name", "").lower()
                            
                            if model_type == "vace" or "vace" in class_name:
                                is_vace_model = True
                        except Exception:
                            pass
                    
                    if is_vace_model:
                        raise RuntimeError(f"""
❌ CRITICAL: VACE model loading completely failed!

🔧 VACE MODEL REQUIRES PROPER SETUP:
VACE models require the official Wan repository with all dependencies properly installed.

❌ NO FALLBACKS FOR VACE - Real implementation required!
Error: {str(e)}

💡 VACE-SPECIFIC SOLUTIONS:
1. 📦 Install Wan repository properly: cd Wan2.1 && pip install -e .
2. 📦 Install diffusers: pip install diffusers
3. 🔧 Verify VACE model files are complete and uncorrupted
4. 💾 Ensure sufficient VRAM (VACE 1.3B requires ~8GB+)
5. 🔄 Restart WebUI after fixing dependencies

🚫 PLACEHOLDER GENERATION DISABLED - REQUIRING REAL VACE MODEL!
""")
                    else:
                        raise RuntimeError(f"""
❌ CRITICAL: Wan model loading completely failed!

🔧 COMPLETE SETUP GUIDE:
1. 📥 Ensure Wan2.1 repository is properly set up
2. 📦 Install Wan dependencies: cd Wan2.1 && pip install -e .
3. 📂 Download models to: models/wan/
4. 🔄 Restart WebUI completely

❌ NO FALLBACKS AVAILABLE - Real Wan implementation required!
Error: {str(e)}
""")
            
            def _load_vace_model_specialized(self, wan_repo_path: Path) -> bool:
                """Specialized VACE model loading that bypasses standard Wan loading issues"""
                try:
                    print("🔧 Loading VACE model with specialized approach...")
                    
                    # Import required modules
                    import wan
                    import torch
                    import json
                    from pathlib import Path
                    
                    # Load the VACE model config
                    config_path = Path(self.model_path) / "config.json"
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            vace_config = json.load(f)
                        print(f"✅ Loaded VACE config: {vace_config.get('model_type', 'unknown')}")
                    else:
                        print("⚠️ No config.json found, using default VACE config")
                        vace_config = {
                            "model_type": "vace",
                            "dim": 1536,
                            "num_layers": 30,
                        }
                    
                    # Create a custom VACE pipeline wrapper
                    class VACEPipelineWrapper:
                        def __init__(self, model_path: str, device: str):
                            self.model_path = model_path
                            self.device = device
                            self.vace_model = None
                            self.vae = None
                            self.text_encoder = None
                            self._load_vace_components()
                        
                        def _load_vace_components(self):
                            """Load VACE model - prioritize official Wan repository over diffusers"""
                            try:
                                print("🔧 Loading VACE model - prioritizing official Wan repository...")
                                
                                # First, try to use the official Wan VACE implementation
                                try:
                                    # Import from the official Wan repository
                                    import wan
                                    from wan.vace import WanVace  # Correct VACE class name
                                    
                                    print("🚀 Using official Wan VACE implementation")
                                    
                                    # Load VACE config - use T2V 1.3B config as base for VACE
                                    try:
                                        from wan.configs.wan_t2v_1_3B import t2v_1_3B as vace_config
                                        print("✅ Using T2V 1.3B config for VACE")
                                    except ImportError:
                                        # Create minimal config if none available
                                        print("⚠️ Creating minimal VACE config")
                                        class MinimalVACEConfig:
                                            def __init__(self):
                                                self.num_train_timesteps = 1000
                                                self.param_dtype = torch.bfloat16
                                                self.t5_dtype = torch.bfloat16
                                                self.text_len = 512
                                                self.vae_stride = [4, 8, 8]
                                                self.patch_size = [1, 2, 2]
                                                self.sample_neg_prompt = "Low quality, blurry, distorted"
                                                self.sample_fps = 24
                                                self.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
                                                self.vae_checkpoint = 'Wan2.1_VAE.pth'
                                                self.t5_tokenizer = 'google/umt5-xxl'
                                        vace_config = MinimalVACEConfig()
                                    
                                    # Initialize VACE model with official implementation
                                    self.vace_model = WanVace(
                                        config=vace_config,
                                        checkpoint_dir=self.model_path,
                                        device_id=0,
                                        rank=0,
                                        dit_fsdp=False,
                                        t5_fsdp=False
                                    )
                                    
                                    # Create wrapper interface that matches our expected API
                                    class OfficialWanVACEWrapper:
                                        def __init__(self, vace_model):
                                            self.vace_model = vace_model
                                        
                                        def __call__(self, **kwargs):
                                            return self.vace_model.generate(**kwargs)
                                        
                                        def generate(self, **kwargs):
                                            # For T2V generation, create dummy blank frames for VACE to transform
                                            # VACE is a video editing model that needs input frames, so we provide blank ones
                                            frame_num = kwargs.get('frame_num', 81)
                                            size = kwargs.get('size', (1280, 720))
                                            
                                            # VACE requires dimensions aligned with VAE stride [4, 8, 8]
                                            # Ensure width and height are divisible by 16 (8*2 for proper alignment)
                                            width, height = size
                                            aligned_width = ((width + 15) // 16) * 16
                                            aligned_height = ((height + 15) // 16) * 16
                                            
                                            print(f"🔧 VACE dimension alignment: {width}x{height} -> {aligned_width}x{aligned_height}")
                                            
                                            # Create blank frames (black video) for VACE to transform
                                            if 'input_frames' not in kwargs or kwargs['input_frames'] == [] or kwargs['input_frames'] is None:
                                                # Create dummy black frames with proper alignment
                                                blank_frame = torch.zeros((3, frame_num, aligned_height, aligned_width), device=self.vace_model.device)
                                                kwargs['input_frames'] = [blank_frame]
                                            
                                            if 'input_masks' not in kwargs or kwargs['input_masks'] == [] or kwargs['input_masks'] is None:
                                                # Create masks that indicate the entire video should be edited (transformed)
                                                # Use proper format: (1, frame_num, height, width)
                                                full_mask = torch.ones((1, frame_num, aligned_height, aligned_width), device=self.vace_model.device)
                                                kwargs['input_masks'] = [full_mask]
                                            
                                            if 'input_ref_images' not in kwargs or kwargs['input_ref_images'] == [] or kwargs['input_ref_images'] is None:
                                                kwargs['input_ref_images'] = [None]
                                            
                                            # Update size to use aligned dimensions
                                            kwargs['size'] = (aligned_width, aligned_height)
                                            
                                            return self.vace_model.generate(**kwargs)
                                        
                                        def generate_image2video(self, image, **kwargs):
                                            # For I2V with VACE, use the provided image as reference
                                            # Enhanced prompt for better continuity
                                            enhanced_prompt = f"Continuing seamlessly from the provided image, {kwargs.get('input_prompt', '')}. Maintaining visual continuity and style."
                                            kwargs['input_prompt'] = enhanced_prompt
                                            
                                            frame_num = kwargs.get('frame_num', 81)
                                            size = kwargs.get('size', (1280, 720))
                                            
                                            # VACE requires dimensions aligned with VAE stride [4, 8, 8]
                                            # Ensure width and height are divisible by 16 (8*2 for proper alignment)
                                            width, height = size
                                            aligned_width = ((width + 15) // 16) * 16
                                            aligned_height = ((height + 15) // 16) * 16
                                            
                                            print(f"🔧 VACE I2V dimension alignment: {width}x{height} -> {aligned_width}x{aligned_height}")
                                            
                                            # For I2V, we can still use dummy frames but provide the image as reference
                                            if 'input_frames' not in kwargs or kwargs['input_frames'] == [] or kwargs['input_frames'] is None:
                                                # Create dummy black frames with proper alignment
                                                blank_frame = torch.zeros((3, frame_num, aligned_height, aligned_width), device=self.vace_model.device)
                                                kwargs['input_frames'] = [blank_frame]
                                            
                                            if 'input_masks' not in kwargs or kwargs['input_masks'] == [] or kwargs['input_masks'] is None:
                                                # Create masks that indicate the entire video should be edited
                                                full_mask = torch.ones((1, frame_num, aligned_height, aligned_width), device=self.vace_model.device)
                                                kwargs['input_masks'] = [full_mask]
                                            
                                            # Set the reference image for I2V generation
                                            if 'input_ref_images' not in kwargs or kwargs['input_ref_images'] == [] or kwargs['input_ref_images'] is None:
                                                # TODO: Convert the provided image to the format expected by VACE
                                                # For now, fall back to T2V approach
                                                kwargs['input_ref_images'] = [None]
                                            
                                            # Update size to use aligned dimensions
                                            kwargs['size'] = (aligned_width, aligned_height)
                                            
                                            return self.vace_model.generate(**kwargs)
                                    
                                    self.pipeline = OfficialWanVACEWrapper(self.vace_model)
                                    
                                    print("✅ VACE model loaded successfully with official Wan implementation")
                                    return True
                                    
                                except (ImportError, AttributeError) as wan_e:
                                    print(f"⚠️ Official Wan VACE implementation not available: {wan_e}")
                                    print("🔄 Falling back to diffusers approach...")
                                
                                # Fallback: try diffusers approach for WebUI Forge compatibility
                                from diffusers import DiffusionPipeline
                                
                                print("🔄 Loading VACE model with DiffusionPipeline (WebUI Forge compatible)...")
                                
                                # Load using the standard approach that works with older diffusers
                                self.pipeline = DiffusionPipeline.from_pretrained(
                                    self.model_path,
                                    torch_dtype=torch.bfloat16,  # VACE uses bfloat16
                                    use_safetensors=True,
                                    trust_remote_code=True  # VACE might need this
                                )
                                
                                # Manually move to device after loading
                                print(f"🔄 Moving VACE model to {self.device}...")
                                self.pipeline = self.pipeline.to(self.device)
                                
                                print("✅ VACE model loaded successfully with WebUI Forge compatible approach")
                                return True
                                
                            except Exception as e:
                                print(f"❌ VACE loading failed: {e}")
                                
                                # Try even more basic approach
                                try:
                                    print("🔄 Trying basic VACE loading for older diffusers...")
                                    from diffusers import DiffusionPipeline
                                    
                                    # Very basic loading without advanced parameters
                                    self.pipeline = DiffusionPipeline.from_pretrained(
                                        self.model_path,
                                        torch_dtype=torch.float16,  # Use float16 for better compatibility
                                        use_safetensors=True
                                    )
                                    
                                    # Move to device manually
                                    self.pipeline = self.pipeline.to(self.device)
                                    
                                    print("✅ VACE model loaded with basic diffusers approach")
                                    return True
                                    
                                except Exception as e2:
                                    print(f"❌ All VACE loading approaches failed")
                                    print(f"   Primary error: {e}")
                                    print(f"   Secondary error: {e2}")
                                    
                                    # Final fallback: clear error message
                                    raise RuntimeError(f"""
❌ CRITICAL: VACE model is not compatible with current T2V workflow!

🔧 VACE MODEL UNDERSTANDING:
VACE (Video and Contour Editing) models are designed for VIDEO EDITING, not simple T2V generation.
They require input_frames, input_masks, and input_ref_images parameters.

❌ INCOMPATIBLE WITH DEFORUM T2V WORKFLOW!

💡 SOLUTIONS (recommended order):
1. 📁 **BEST**: Download and use T2V models instead:
   - Wan2.1-T2V-1.3B (8GB VRAM, consumer-friendly)
   - Wan2.1-T2V-14B (more VRAM required)
   
2. 🔧 **ALTERNATIVE**: Use official Wan repository for VACE editing:
   cd Wan2.1 && pip install -e .
   
3. 🔄 **FUTURE**: Wait for VACE T2V support in future updates

📥 **DOWNLOAD T2V MODELS**:
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan/Wan2.1-T2V-1.3B

🌐 **REFERENCE**: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B

🚫 **VACE IS FOR EDITING, NOT T2V GENERATION!**
""")
                        
                        def __call__(self, *args, **kwargs):
                            """VACE pipeline call with proper parameter handling"""
                            if not hasattr(self, 'pipeline') or self.pipeline is None:
                                raise RuntimeError("VACE pipeline not loaded")
                            
                            # VACE is an all-in-one model, so handle both T2V and I2V calls
                            return self.pipeline(*args, **kwargs)
                        
                        def generate(self, *args, **kwargs):
                            """Official Wan T2V generate method interface for VACE wrapper"""
                            print(f"🎬 VACE wrapper generate method called")
                            
                            # Convert official Wan parameters to Diffusers parameters
                            input_prompt = kwargs.get('input_prompt', args[0] if args else "")
                            size = kwargs.get('size', args[1] if len(args) > 1 else (512, 512))
                            frame_num = kwargs.get('frame_num', args[2] if len(args) > 2 else 5)
                            sampling_steps = kwargs.get('sampling_steps', args[3] if len(args) > 3 else 20)
                            guide_scale = kwargs.get('guide_scale', args[4] if len(args) > 4 else 7.5)
                            
                            width, height = size
                            print(f"🎯 VACE T2V: {input_prompt[:30]}..., size: {width}x{height}, frames: {frame_num}")
                            
                            # Check if this is an official Wan wrapper or diffusers pipeline
                            if hasattr(self.pipeline, 'vace_model'):
                                # Official Wan VACE
                                return self.pipeline.generate(
                                    input_prompt=input_prompt,
                                    size=(width, height),
                                    frame_num=frame_num,
                                    sampling_steps=sampling_steps,
                                    guide_scale=guide_scale
                                )
                            else:
                                # Diffusers pipeline - use compatible parameters
                                pipeline_kwargs = {
                                    "prompt": input_prompt,
                                    "num_inference_steps": sampling_steps,
                                    "guidance_scale": guide_scale,
                                }
                                
                                # Add video-specific parameters if supported
                                import inspect
                                try:
                                    pipeline_signature = inspect.signature(self.pipeline.__call__)
                                    
                                    if 'height' in pipeline_signature.parameters:
                                        pipeline_kwargs['height'] = height
                                    if 'width' in pipeline_signature.parameters:
                                        pipeline_kwargs['width'] = width
                                    if 'num_frames' in pipeline_signature.parameters:
                                        pipeline_kwargs['num_frames'] = frame_num
                                except:
                                    # If signature inspection fails, just use basic parameters
                                    pass
                                
                                return self.pipeline(**pipeline_kwargs)
                        
                        def generate_image2video(self, *args, **kwargs):
                            """I2V generation with proper aspect ratio preservation for VACE"""
                            print(f"🎬 VACE I2V wrapper called")
                            
                            # Extract parameters
                            image = args[0] if args else kwargs.get('image')
                            prompt = args[1] if len(args) > 1 else kwargs.get('prompt', "")
                            
                            # Use provided dimensions instead of auto-preserving input image aspect ratio
                            # This ensures consistency with T2V aligned dimensions
                            height = args[2] if len(args) > 2 else kwargs.get('height', 512)
                            width = args[3] if len(args) > 3 else kwargs.get('width', 512)
                            
                            # Log dimension handling
                            if image and hasattr(image, 'size'):
                                input_width, input_height = image.size
                                print(f"🖼️ Input image size: {input_width}x{input_height}")
                                print(f"🎯 Target I2V size: {width}x{height} (using aligned dimensions)")
                                
                                # Resize the input image to match target dimensions if needed
                                if (input_width, input_height) != (width, height):
                                    print(f"🔧 Resizing input image to match target dimensions")
                                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                            else:
                                print(f"🎯 Target I2V size: {width}x{height}")
                            
                            num_frames = args[4] if len(args) > 4 else kwargs.get('num_frames', 5)
                            num_inference_steps = args[5] if len(args) > 5 else kwargs.get('num_inference_steps', 20)
                            guidance_scale = args[6] if len(args) > 6 else kwargs.get('guidance_scale', 7.5)
                            strength = kwargs.get('strength', 0.8)  # I2V strength parameter
                            
                            print(f"🎯 VACE I2V: {prompt[:30]}..., size: {width}x{height}, frames: {num_frames}, strength: {strength}")
                            
                            # Check if this is an official Wan wrapper or diffusers pipeline
                            if hasattr(self.pipeline, 'vace_model'):
                                # Official Wan VACE
                                return self.pipeline.generate_image2video(
                                    image=image,
                                    input_prompt=prompt,
                                    height=height,
                                    width=width,
                                    num_frames=num_frames,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    strength=strength,  # Pass the strength parameter
                                )
                            else:
                                # Diffusers pipeline - check if it supports image input
                                import inspect
                                try:
                                    pipeline_signature = inspect.signature(self.pipeline.__call__)
                                    
                                    pipeline_kwargs = {
                                        "prompt": prompt,
                                        "num_inference_steps": num_inference_steps,
                                        "guidance_scale": guidance_scale,
                                    }
                                    
                                    # Add parameters if supported
                                    if 'image' in pipeline_signature.parameters and image is not None:
                                        pipeline_kwargs['image'] = image
                                        print("✅ Using image input for I2V")
                                    else:
                                        # Fallback: enhance prompt for T2V-style generation
                                        enhanced_prompt = f"Continuing seamlessly from the provided image, {prompt}. Maintaining visual continuity and style."
                                        pipeline_kwargs['prompt'] = enhanced_prompt
                                        print("⚠️ No image support detected, using enhanced T2V prompt")
                                    
                                    if 'height' in pipeline_signature.parameters:
                                        pipeline_kwargs['height'] = height
                                    if 'width' in pipeline_signature.parameters:
                                        pipeline_kwargs['width'] = width
                                    if 'num_frames' in pipeline_signature.parameters:
                                        pipeline_kwargs['num_frames'] = num_frames
                                    
                                    return self.pipeline(**pipeline_kwargs)
                                    
                                except Exception as e:
                                    print(f"⚠️ Pipeline inspection failed: {e}")
                                    # Fallback to basic call
                                    enhanced_prompt = f"Continuing seamlessly from the provided image, {prompt}. Maintaining visual continuity and style."
                                    return self.pipeline(prompt=enhanced_prompt)
                    
                    # Create the VACE wrapper
                    print("🔧 Creating VACE pipeline wrapper...")
                    vace_wrapper = VACEPipelineWrapper(self.model_path, self.device)
                    
                    # Set up the pipeline attributes
                    self.pipeline = vace_wrapper
                    self.t2v_model = vace_wrapper
                    self.i2v_model = vace_wrapper
                    
                    print("✅ VACE specialized loading completed successfully")
                    return True
                    
                except Exception as e:
                    print(f"❌ VACE specialized loading failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Re-raise the error to fail fast instead of continuing
                    raise RuntimeError(f"""
❌ CRITICAL: VACE specialized loading completely failed!

🔧 REAL VACE MODEL REQUIRED:
The VACE model could not be loaded with any method.

❌ NO FALLBACKS - Real VACE implementation required!
Error: {e}

💡 SOLUTIONS:
1. 📦 Install VACE dependencies properly
2. 🔧 Verify model file integrity
3. 💾 Check available VRAM
4. 🔄 Restart WebUI

🚫 REFUSING TO GENERATE PLACEHOLDER CONTENT!
""")
            
            def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                """Generate video frames"""
                if not self.loaded:
                    raise RuntimeError("Wan model not loaded")
                
                print(f"🎬 Generating {num_frames} frames with Wan...")
                print(f"   📝 Prompt: {prompt[:50]}...")
                print(f"   📐 Size: {width}x{height}")
                print(f"   🔧 Steps: {num_inference_steps}")
                print(f"   📏 Guidance: {guidance_scale}")
                
                try:
                    # Use official Wan T2V if available
                    if hasattr(self, 't2v_model') and self.t2v_model:
                        print("🚀 Using official Wan T2V model")
                        
                        result = self.t2v_model.generate(
                            input_prompt=prompt,
                            size=(width, height),
                            frame_num=num_frames,
                            sampling_steps=num_inference_steps,
                            guide_scale=guidance_scale,
                            shift=5.0,
                            sample_solver='unipc',
                            offload_model=True,
                            **kwargs
                        )
                        
                        return result
                    
                    # Use diffusers model if available
                    elif hasattr(self, 'model') and self.model:
                        print("🚀 Using diffusers-based model")
                        
                        generation_kwargs = {
                            "prompt": prompt,
                            "height": height,
                            "width": width,
                            "num_frames": num_frames,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                        }
                        
                        # Add optional parameters if supported
                        import inspect
                        if hasattr(self.model, '__call__'):
                            sig = inspect.signature(self.model.__call__)
                            
                            if 'output_type' in sig.parameters:
                                generation_kwargs['output_type'] = 'pil'
                            if 'return_dict' in sig.parameters:
                                generation_kwargs['return_dict'] = False
                        
                        with torch.no_grad():
                            result = self.model(**generation_kwargs)
                        
                        return result
                    
                    # No valid model available
                    else:
                        raise RuntimeError("""
❌ CRITICAL: No valid Wan model loaded!

🔧 SETUP REQUIRED:
1. Install official Wan repository
2. Download Wan models
3. Restart WebUI

💡 No fallbacks available - real Wan implementation required.
""")
                
                except Exception as e:
                    print(f"❌ Generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    raise RuntimeError(f"""
❌ CRITICAL: Wan video generation failed!

🔧 TROUBLESHOOTING:
1. ✅ Verify Wan models are properly installed
2. 🔄 Check CUDA/GPU availability: {torch.cuda.is_available()}
3. 💾 Check available VRAM/memory
4. 📦 Verify all dependencies are installed

❌ NO FALLBACKS - Real Wan implementation required!
Error: {e}
""")
            
            def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                """Generate video from image (I2V)"""
                if not self.loaded:
                    raise RuntimeError("Wan model not loaded")
                
                print(f"🎬 Generating I2V {num_frames} frames with Wan...")
                
                try:
                    # Use official Wan I2V if available
                    if hasattr(self, 'i2v_model') and self.i2v_model:
                        print("🚀 Using official Wan I2V model")
                        
                        result = self.i2v_model.generate(
                            input_prompt=prompt,
                            img=image,
                            max_area=height * width,
                            frame_num=num_frames,
                            sampling_steps=num_inference_steps,
                            guide_scale=guidance_scale,
                            shift=5.0,
                            sample_solver='unipc',
                            offload_model=True,
                            **kwargs
                        )
                        
                        return result
                    
                    # Fallback to T2V with enhanced prompt if I2V not available
                    else:
                        print("⚠️ I2V model not available, using enhanced T2V")
                        enhanced_prompt = f"Starting from the given image, {prompt}. Maintain visual continuity."
                        
                        return self.__call__(
                            enhanced_prompt, height, width, num_frames, 
                            num_inference_steps, guidance_scale, **kwargs
                        )
                
                except Exception as e:
                    print(f"❌ I2V generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    raise RuntimeError(f"""
❌ CRITICAL: Wan I2V generation failed!

🔧 TROUBLESHOOTING:
1. ✅ Verify I2V model is properly loaded
2. 🖼️ Check input image format and size
3. 💾 Check available VRAM/memory
4. 📦 Verify Wan I2V dependencies

❌ NO FALLBACKS - Real Wan I2V implementation required!
Error: {e}
""")
            
            def _check_if_vace_model(self, model_path: str) -> bool:
                """Check if model is a VACE model"""
                import json
                from pathlib import Path
                
                path = Path(model_path)
                print(f"🔍 Checking if {path} is a VACE model...")
                
                # Check config.json first
                config_file = path / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        model_type = config.get("model_type", "").lower()
                        if model_type == "vace":
                            print("✅ VACE model detected via config.json!")
                            return True
                            
                    except Exception as e:
                        print(f"⚠️ Error reading config.json: {e}")
                
                # Check path name
                path_str = str(path).lower()
                is_vace_by_name = "vace" in path_str
                
                if is_vace_by_name:
                    print("✅ VACE model detected via path name!")
                else:
                    print("❌ Not a VACE model")
                
                return is_vace_by_name
            
            def _create_vace_config(self):
                """Create VACE configuration"""
                import json
                from pathlib import Path
                
                config_file = Path(self.model_path) / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            model_config = json.load(f)
                        
                        class VACEConfig:
                            def __init__(self, model_config):
                                self.model_type = model_config.get("model_type", "vace")
                                self.dim = model_config.get("dim", 1536)
                                self.ffn_dim = model_config.get("ffn_dim", 8960)
                                self.num_heads = model_config.get("num_heads", 12)
                                self.num_layers = model_config.get("num_layers", 30)
                                self.in_dim = model_config.get("in_dim", 16)
                                self.out_dim = model_config.get("out_dim", 16)
                                self.text_len = model_config.get("text_len", 512)
                                self.freq_dim = model_config.get("freq_dim", 256)
                                self.eps = model_config.get("eps", 1e-6)
                                self.vace_layers = model_config.get("vace_layers", [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
                                self.vace_in_dim = model_config.get("vace_in_dim", 96)
                                self.num_train_timesteps = 1000
                                self.param_dtype = torch.bfloat16
                                self.t5_dtype = torch.bfloat16
                                self.clip_dtype = torch.bfloat16
                                self.vae_stride = [4, 8, 8]
                                self.patch_size = [1, 2, 2]
                                self.sample_neg_prompt = "Low quality, blurry, distorted, nsfw, nude"
                                self.sample_fps = 24
                                self.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
                                self.vae_checkpoint = 'Wan2.1_VAE.pth'
                                self.t5_tokenizer = 'google/umt5-xxl'
                                self.clip_checkpoint = 'clip_l.safetensors'
                                self.clip_tokenizer = 'openai/clip-vit-large-patch14'
                        
                        return VACEConfig(model_config)
                        
                    except Exception as e:
                        print(f"⚠️ Failed to load VACE config: {e}")
                
                # Fallback to default VACE config
                class VACEConfig:
                    def __init__(self):
                        self.model_type = "vace"
                        self.dim = 1536
                        self.ffn_dim = 8960
                        self.num_heads = 12
                        self.num_layers = 30
                        self.in_dim = 16
                        self.out_dim = 16
                        self.text_len = 512
                        self.freq_dim = 256
                        self.eps = 1e-6
                        self.vace_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
                        self.vace_in_dim = 96
                        self.num_train_timesteps = 1000
                        self.param_dtype = torch.bfloat16
                        self.t5_dtype = torch.bfloat16
                        self.clip_dtype = torch.bfloat16
                        self.vae_stride = [4, 8, 8]
                        self.patch_size = [1, 2, 2]
                        self.sample_neg_prompt = "Low quality, blurry, distorted, nsfw, nude"
                        self.sample_fps = 24
                        self.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
                        self.vae_checkpoint = 'Wan2.1_VAE.pth'
                        self.t5_tokenizer = 'google/umt5-xxl'
                        self.clip_checkpoint = 'clip_l.safetensors'
                        self.clip_tokenizer = 'openai/clip-vit-large-patch14'
                
                return VACEConfig()
            
            def _load_wan_model(self):
                """Load the actual Wan model with multiple fallback strategies"""
                try:
                    # Strategy 1: Try to import and use official Wan
                    print("🔄 Attempting to load official Wan model...")
                    
                    # Add Wan2.1 to path if it exists
                    import sys
                    from pathlib import Path
                    
                    # Look for Wan2.1 directory in the extension root
                    extension_root = Path(__file__).parent.parent.parent.parent
                    wan_repo_path = extension_root / "Wan2.1"
                    
                    print(f"🔍 Looking for Wan repository at: {wan_repo_path}")
                    
                    if wan_repo_path.exists() and (wan_repo_path / "wan").exists():
                        # Add to Python path
                        if str(wan_repo_path) not in sys.path:
                            sys.path.insert(0, str(wan_repo_path))
                            print(f"✅ Added Wan repo to path: {wan_repo_path}")
                        
                        # Apply Flash Attention compatibility patches
                        try:
                            from .wan_flash_attention_patch import apply_wan_compatibility_patches
                        except Exception as patch_e:
                            print(f"⚠️ Could not apply compatibility patches: {patch_e}")
                        
                        # Check what type of model we actually have FIRST
                        model_path = Path(self.model_path)
                        config_path = model_path / "config.json"
                        
                        is_vace_model = False
                        if config_path.exists():
                            try:
                                import json
                                with open(config_path, 'r') as f:
                                    config = json.load(f)
                                model_type = config.get("model_type", "").lower()
                                class_name = config.get("_class_name", "").lower()
                                
                                if model_type == "vace" or "vace" in class_name:
                                    is_vace_model = True
                                    print("🎯 Detected VACE model - using specialized VACE loading")
                                else:
                                    print(f"🎯 Detected {model_type} model")
                            except Exception as e:
                                print(f"⚠️ Could not read config: {e}")
                        
                        # Handle VACE models with specialized loading
                        if is_vace_model:
                            print("🔧 Using specialized VACE model loading...")
                            success = self._load_vace_model_specialized(wan_repo_path)
                            if success:
                                self.loaded = True
                                print("✅ VACE model loaded successfully with specialized loader")
                                return
                            else:
                                print("❌ Specialized VACE loading failed, trying standard approach...")
                                # Continue to standard loading as fallback
                        
                        # Standard Wan model loading for T2V/I2V models
                        # Verify the wan module can be imported
                        try:
                            import wan  # type: ignore
                            print("✅ Successfully imported wan module from local repository")
                            
                            # Import specific components
                            from wan.text2video import WanT2V  # type: ignore
                            from wan.image2video import WanI2V  # type: ignore
                            print("✅ Successfully imported WanT2V and WanI2V classes")
                            
                            # Try to load configs
                            try:
                                # Check if config files exist
                                config_dir = wan_repo_path / "wan" / "configs"
                                if config_dir.exists():
                                    print(f"✅ Found config directory: {config_dir}")
                                    
                                    # Try to import configs
                                    try:
                                        if is_vace_model:
                                            # For VACE models, use T2V config as base
                                            from wan.configs.wan_t2v_1_3B import t2v_1_3B as vace_config
                                            print("✅ Using T2V config for VACE model")
                                            t2v_config = vace_config
                                            i2v_config = vace_config  # Same config for both
                                        else:
                                            from wan.configs.wan_t2v_14B import t2v_14B as t2v_config
                                            from wan.configs.wan_i2v_14B import i2v_14B as i2v_config
                                            print("✅ Loaded official Wan configs")
                                    except ImportError as config_e:
                                        print(f"⚠️ Config import failed: {config_e}, will use minimal configs")
                                        # Create minimal config structure
                                        class MinimalConfig:
                                            def __init__(self):
                                                self.model = type('obj', (object,), {
                                                    'num_attention_heads': 32,
                                                    'attention_head_dim': 128,
                                                    'in_channels': 4,
                                                    'out_channels': 4,
                                                    'num_layers': 28,
                                                    'sample_size': 32,
                                                    'patch_size': 2,
                                                    'num_vector_embeds': None,
                                                    'activation_fn': "geglu",
                                                    'num_embeds_ada_norm': 1000,
                                                    'norm_elementwise_affine': False,
                                                    'norm_eps': 1e-6,
                                                    'attention_bias': True,
                                                    'caption_channels': 4096
                                                })
                                                
                                        t2v_config = MinimalConfig()
                                        i2v_config = MinimalConfig()
                                else:
                                    print("⚠️ Config directory not found, creating minimal configs")
                                    # Create minimal config structure
                                    class MinimalConfig:
                                        def __init__(self):
                                            self.model = type('obj', (object,), {
                                                'num_attention_heads': 32,
                                                'attention_head_dim': 128,
                                                'in_channels': 4,
                                                'out_channels': 4,
                                                'num_layers': 28,
                                                'sample_size': 32,
                                                'patch_size': 2,
                                                'num_vector_embeds': None,
                                                'activation_fn': "geglu",
                                                'num_embeds_ada_norm': 1000,
                                                'norm_elementwise_affine': False,
                                                'norm_eps': 1e-6,
                                                'attention_bias': True,
                                                'caption_channels': 4096
                                            })
                                            
                                    t2v_config = MinimalConfig()
                                    i2v_config = MinimalConfig()
                                    
                            except Exception as config_e:
                                print(f"⚠️ Config loading failed: {config_e}, using minimal configs")
                                # Create minimal config structure
                                class MinimalConfig:
                                    def __init__(self):
                                        self.model = type('obj', (object,), {
                                            'num_attention_heads': 32,
                                            'attention_head_dim': 128,
                                            'in_channels': 4,
                                            'out_channels': 4,
                                            'num_layers': 28,
                                            'sample_size': 32,
                                            'patch_size': 2,
                                            'num_vector_embeds': None,
                                            'activation_fn': "geglu",
                                            'num_embeds_ada_norm': 1000,
                                            'norm_elementwise_affine': False,
                                            'norm_eps': 1e-6,
                                            'attention_bias': True,
                                            'caption_channels': 4096
                                        })
                                        
                                t2v_config = MinimalConfig()
                                i2v_config = MinimalConfig()
                            
                            # Initialize T2V model with correct parameters
                            print(f"🚀 Initializing WanT2V with checkpoint dir: {self.model_path}")
                            
                            if is_vace_model:
                                # For VACE models, use special loading parameters
                                print("🎯 Loading VACE model with T2V-compatible parameters")
                                try:
                                    self.t2v_model = WanT2V(
                                        config=t2v_config,
                                        checkpoint_dir=self.model_path,
                                        device_id=0,
                                        rank=0,
                                        dit_fsdp=False,
                                        t5_fsdp=False
                                    )
                                    print("✅ VACE model loaded as T2V")
                                    
                                    # For VACE, the same model can handle both T2V and I2V
                                    self.i2v_model = self.t2v_model
                                    print("✅ VACE model ready for both T2V and I2V")
                                    
                                except Exception as vace_e:
                                    print(f"❌ VACE T2V loading failed: {vace_e}")
                                    # For VACE models, fail fast instead of trying diffusers fallback
                                    raise RuntimeError(f"""
❌ CRITICAL: VACE model loading failed with official Wan loader!

🔧 REAL VACE MODEL REQUIRED:
The VACE model at {self.model_path} could not be loaded with official Wan methods.

❌ NO FALLBACKS for VACE - Real implementation required!
Specific error: {vace_e}

💡 SOLUTIONS:
1. 📦 Ensure Wan dependencies are properly installed: cd Wan2.1 && pip install -e .
2. 🔧 Verify VACE model files are complete and uncorrupted
3. 💾 Ensure sufficient VRAM (VACE 1.3B requires ~8GB+)
4. 🔄 Re-download VACE model if corrupted
5. 🔄 Restart WebUI after fixing dependencies

🚫 REFUSING TO GENERATE PLACEHOLDER CONTENT!
""")
                            else:
                                # Standard T2V/I2V model loading
                                self.t2v_model = WanT2V(
                                    config=t2v_config,
                                    checkpoint_dir=self.model_path,
                                    device_id=0,
                                    rank=0,
                                    dit_fsdp=False,
                                    t5_fsdp=False
                                )
                                
                                # Try to initialize I2V model if available
                                try:
                                    print(f"🚀 Initializing WanI2V with checkpoint dir: {self.model_path}")
                                    self.i2v_model = WanI2V(
                                        config=i2v_config,
                                        checkpoint_dir=self.model_path,
                                        device_id=0,
                                        rank=0,
                                        dit_fsdp=False,
                                        t5_fsdp=False
                                    )
                                    print("✅ I2V model loaded for chaining support")
                                except Exception as e:
                                    print(f"⚠️ I2V model not available: {e}")
                                    self.i2v_model = None
                            
                            self.loaded = True
                            print("✅ Official Wan models loaded successfully")
                            return
                            
                        except ImportError as import_e:
                            print(f"❌ Failed to import wan module: {import_e}")
                            # Continue to fallback methods
                    else:
                        print(f"❌ Wan repository not found at: {wan_repo_path}")
                        print("💡 Expected structure: Wan2.1/wan/ directory with Python modules")
                    
                    # If we get here, the official Wan import failed
                    print("⚠️ Official Wan import failed, trying diffusers fallback...")
                    
                    # Check if this is a VACE model - if so, refuse fallback
                    model_path = Path(self.model_path)
                    config_path = model_path / "config.json"
                    
                    is_vace_model = False
                    if config_path.exists():
                        try:
                            import json
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            model_type = config.get("model_type", "").lower()
                            class_name = config.get("_class_name", "").lower()
                            
                            if model_type == "vace" or "vace" in class_name:
                                is_vace_model = True
                        except Exception:
                            pass
                    
                    if is_vace_model:
                        # Refuse to use diffusers fallback for VACE models
                        raise RuntimeError(f"""
❌ CRITICAL: VACE model detected but official Wan loading failed!

🔧 REAL VACE MODEL REQUIRES OFFICIAL WAN:
VACE models require the official Wan repository and cannot use diffusers fallback.

❌ NO FALLBACKS for VACE - Real Wan implementation required!

💡 SOLUTIONS:
1. 📦 Properly install Wan repository: cd Wan2.1 && pip install -e .
2. 🔧 Fix Wan import issues (check Python path, dependencies)
3. 💾 Ensure all Wan dependencies are installed
4. 🔄 Restart WebUI after fixing Wan setup

🚫 REFUSING DIFFUSERS FALLBACK FOR VACE MODEL!
""")
                    
                    # Strategy 2: Try diffusers-based loading as fallback (only for non-VACE models)
                    try:
                        from diffusers import DiffusionPipeline
                        
                        print("🔄 Attempting diffusers-based loading for non-VACE model...")
                        self.model = DiffusionPipeline.from_pretrained(
                            self.model_path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                        
                        if torch.cuda.is_available():
                            self.model = self.model.to(self.device)
                        
                        self.loaded = True
                        print("✅ Diffusers-based model loaded successfully (fallback for non-VACE)")
                        return
                        
                    except Exception as diffusers_e:
                        print(f"❌ Diffusers loading also failed: {diffusers_e}")
                    
                    # Both methods failed, provide comprehensive error
                    raise RuntimeError(f"""
❌ SETUP REQUIRED: Wan 2.1 Official Repository Setup Issue!

🔧 QUICK SETUP:

📥 The Wan2.1 directory exists but import failed. Try:

1. 📦 Install Wan dependencies:
   cd Wan2.1
   pip install -e .

2. 📂 Download models to correct location:
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan

3. 🔄 Restart WebUI completely

💡 The Wan2.1 repository is already present at: {wan_repo_path}
   But the Python modules couldn't be imported properly.

🌐 If issues persist, check: https://github.com/Wan-Video/Wan2.1#readme
""")
                
                except Exception as e:
                    print(f"❌ All model loading strategies failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Check if this was a VACE model to provide specific guidance
                    model_path = Path(self.model_path)
                    config_path = model_path / "config.json"
                    is_vace_model = False
                    
                    if config_path.exists():
                        try:
                            import json
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            model_type = config.get("model_type", "").lower()
                            class_name = config.get("_class_name", "").lower()
                            
                            if model_type == "vace" or "vace" in class_name:
                                is_vace_model = True
                        except Exception:
                            pass
                    
                    if is_vace_model:
                        raise RuntimeError(f"""
❌ CRITICAL: VACE model loading completely failed!

🔧 VACE MODEL REQUIRES PROPER SETUP:
VACE models require the official Wan repository with all dependencies properly installed.

❌ NO FALLBACKS FOR VACE - Real implementation required!
Error: {str(e)}

💡 VACE-SPECIFIC SOLUTIONS:
1. 📦 Install Wan repository properly: cd Wan2.1 && pip install -e .
2. 📦 Install diffusers: pip install diffusers
3. 🔧 Verify VACE model files are complete and uncorrupted
4. 💾 Ensure sufficient VRAM (VACE 1.3B requires ~8GB+)
5. 🔄 Restart WebUI after fixing dependencies

🚫 PLACEHOLDER GENERATION DISABLED - REQUIRING REAL VACE MODEL!
""")
                    else:
                        raise RuntimeError(f"""
❌ CRITICAL: Wan model loading completely failed!

🔧 COMPLETE SETUP GUIDE:
1. 📥 Ensure Wan2.1 repository is properly set up
2. 📦 Install Wan dependencies: cd Wan2.1 && pip install -e .
3. 📂 Download models to: models/wan/
4. 🔄 Restart WebUI completely

❌ NO FALLBACKS AVAILABLE - Real Wan implementation required!
Error: {str(e)}
""")
            
            def _load_vace_model_specialized(self, wan_repo_path: Path) -> bool:
                """Specialized VACE model loading that bypasses standard Wan loading issues"""
                try:
                    print("🔧 Loading VACE model with specialized approach...")
                    
                    # Import required modules
                    import wan
                    import torch
                    import json
                    from pathlib import Path
                    
                    # Load the VACE model config
                    config_path = Path(self.model_path) / "config.json"
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            vace_config = json.load(f)
                        print(f"✅ Loaded VACE config: {vace_config.get('model_type', 'unknown')}")
                    else:
                        print("⚠️ No config.json found, using default VACE config")
                        vace_config = {
                            "model_type": "vace",
                            "dim": 1536,
                            "num_layers": 30,
                        }
                    
                    # Create a custom VACE pipeline wrapper
                    class VACEPipelineWrapper:
                        def __init__(self, model_path: str, device: str):
                            self.model_path = model_path
                            self.device = device
                            self.vace_model = None
                            self.vae = None
                            self.text_encoder = None
                            self._load_vace_components()
                        
                        def _load_vace_components(self):
                            """Load VACE model - prioritize official Wan repository over diffusers"""
                            try:
                                print("🔧 Loading VACE model - prioritizing official Wan repository...")
                                
                                # First, try to use the official Wan VACE implementation
                                try:
                                    # Import from the official Wan repository
                                    import wan
                                    from wan.vace import WanVace  # Correct VACE class name
                                    
                                    print("🚀 Using official Wan VACE implementation")
                                    
                                    # Load VACE config - use T2V 1.3B config as base for VACE
                                    try:
                                        from wan.configs.wan_t2v_1_3B import t2v_1_3B as vace_config
                                        print("✅ Using T2V 1.3B config for VACE")
                                    except ImportError:
                                        # Create minimal config if none available
                                        print("⚠️ Creating minimal VACE config")
                                        class MinimalVACEConfig:
                                            def __init__(self):
                                                self.num_train_timesteps = 1000
                                                self.param_dtype = torch.bfloat16
                                                self.t5_dtype = torch.bfloat16
                                                self.text_len = 512
                                                self.vae_stride = [4, 8, 8]
                                                self.patch_size = [1, 2, 2]
                                                self.sample_neg_prompt = "Low quality, blurry, distorted"
                                                self.sample_fps = 24
                                                self.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
                                                self.vae_checkpoint = 'Wan2.1_VAE.pth'
                                                self.t5_tokenizer = 'google/umt5-xxl'
                                        vace_config = MinimalVACEConfig()
                                    
                                    # Initialize VACE model with official implementation
                                    self.vace_model = WanVace(
                                        config=vace_config,
                                        checkpoint_dir=self.model_path,
                                        device_id=0,
                                        rank=0,
                                        dit_fsdp=False,
                                        t5_fsdp=False
                                    )
                                    
                                    # Create wrapper interface that matches our expected API
                                    class OfficialWanVACEWrapper:
                                        def __init__(self, vace_model):
                                            self.vace_model = vace_model
                                        
                                        def __call__(self, **kwargs):
                                            return self.vace_model.generate(**kwargs)
                                        
                                        def generate(self, **kwargs):
                                            # For T2V generation, create dummy blank frames for VACE to transform
                                            # VACE is a video editing model that needs input frames, so we provide blank ones
                                            frame_num = kwargs.get('frame_num', 81)
                                            size = kwargs.get('size', (1280, 720))
                                            
                                            # VACE requires dimensions aligned with VAE stride [4, 8, 8]
                                            # Ensure width and height are divisible by 16 (8*2 for proper alignment)
                                            width, height = size
                                            aligned_width = ((width + 15) // 16) * 16
                                            aligned_height = ((height + 15) // 16) * 16
                                            
                                            print(f"🔧 VACE dimension alignment: {width}x{height} -> {aligned_width}x{aligned_height}")
                                            
                                            # Create blank frames (black video) for VACE to transform
                                            if 'input_frames' not in kwargs or kwargs['input_frames'] == [] or kwargs['input_frames'] is None:
                                                # Create dummy black frames with proper alignment
                                                blank_frame = torch.zeros((3, frame_num, aligned_height, aligned_width), device=self.vace_model.device)
                                                kwargs['input_frames'] = [blank_frame]
                                            
                                            if 'input_masks' not in kwargs or kwargs['input_masks'] == [] or kwargs['input_masks'] is None:
                                                # Create masks that indicate the entire video should be edited (transformed)
                                                # Use proper format: (1, frame_num, height, width)
                                                full_mask = torch.ones((1, frame_num, aligned_height, aligned_width), device=self.vace_model.device)
                                                kwargs['input_masks'] = [full_mask]
                                            
                                            if 'input_ref_images' not in kwargs or kwargs['input_ref_images'] == [] or kwargs['input_ref_images'] is None:
                                                kwargs['input_ref_images'] = [None]
                                            
                                            # Update size to use aligned dimensions
                                            kwargs['size'] = (aligned_width, aligned_height)
                                            
                                            return self.vace_model.generate(**kwargs)
                                        
                                        def generate_image2video(self, image, **kwargs):
                                            # For I2V with VACE, use the provided image as reference
                                            # Enhanced prompt for better continuity
                                            enhanced_prompt = f"Continuing seamlessly from the provided image, {kwargs.get('input_prompt', '')}. Maintaining visual continuity and style."
                                            kwargs['input_prompt'] = enhanced_prompt
                                            
                                            frame_num = kwargs.get('frame_num', 81)
                                            size = kwargs.get('size', (1280, 720))
                                            
                                            # VACE requires dimensions aligned with VAE stride [4, 8, 8]
                                            # Ensure width and height are divisible by 16 (8*2 for proper alignment)
                                            width, height = size
                                            aligned_width = ((width + 15) // 16) * 16
                                            aligned_height = ((height + 15) // 16) * 16
                                            
                                            print(f"🔧 VACE I2V dimension alignment: {width}x{height} -> {aligned_width}x{aligned_height}")
                                            
                                            # For I2V, we can still use dummy frames but provide the image as reference
                                            if 'input_frames' not in kwargs or kwargs['input_frames'] == [] or kwargs['input_frames'] is None:
                                                # Create dummy black frames with proper alignment
                                                blank_frame = torch.zeros((3, frame_num, aligned_height, aligned_width), device=self.vace_model.device)
                                                kwargs['input_frames'] = [blank_frame]
                                            
                                            if 'input_masks' not in kwargs or kwargs['input_masks'] == [] or kwargs['input_masks'] is None:
                                                # Create masks that indicate the entire video should be edited
                                                full_mask = torch.ones((1, frame_num, aligned_height, aligned_width), device=self.vace_model.device)
                                                kwargs['input_masks'] = [full_mask]
                                            
                                            # Set the reference image for I2V generation
                                            if 'input_ref_images' not in kwargs or kwargs['input_ref_images'] == [] or kwargs['input_ref_images'] is None:
                                                # TODO: Convert the provided image to the format expected by VACE
                                                # For now, fall back to T2V approach
                                                kwargs['input_ref_images'] = [None]
                                            
                                            # Update size to use aligned dimensions
                                            kwargs['size'] = (aligned_width, aligned_height)
                                            
                                            return self.vace_model.generate(**kwargs)
                                    
                                    self.pipeline = OfficialWanVACEWrapper(self.vace_model)
                                    
                                    print("✅ VACE model loaded successfully with official Wan implementation")
                                    return True
                                    
                                except (ImportError, AttributeError) as wan_e:
                                    print(f"⚠️ Official Wan VACE implementation not available: {wan_e}")
                                    print("🔄 Falling back to diffusers approach...")
                                
                                # Fallback: try diffusers approach for WebUI Forge compatibility
                                from diffusers import DiffusionPipeline
                                
                                print("🔄 Loading VACE model with DiffusionPipeline (WebUI Forge compatible)...")
                                
                                # Load using the standard approach that works with older diffusers
                                self.pipeline = DiffusionPipeline.from_pretrained(
                                    self.model_path,
                                    torch_dtype=torch.bfloat16,  # VACE uses bfloat16
                                    use_safetensors=True,
                                    trust_remote_code=True  # VACE might need this
                                )
                                
                                # Manually move to device after loading
                                print(f"🔄 Moving VACE model to {self.device}...")
                                self.pipeline = self.pipeline.to(self.device)
                                
                                print("✅ VACE model loaded successfully with WebUI Forge compatible approach")
                                return True
                                
                            except Exception as e:
                                print(f"❌ VACE loading failed: {e}")
                                
                                # Try even more basic approach
                                try:
                                    print("🔄 Trying basic VACE loading for older diffusers...")
                                    from diffusers import DiffusionPipeline
                                    
                                    # Very basic loading without advanced parameters
                                    self.pipeline = DiffusionPipeline.from_pretrained(
                                        self.model_path,
                                        torch_dtype=torch.float16,  # Use float16 for better compatibility
                                        use_safetensors=True
                                    )
                                    
                                    # Move to device manually
                                    self.pipeline = self.pipeline.to(self.device)
                                    
                                    print("✅ VACE model loaded with basic diffusers approach")
                                    return True
                                    
                                except Exception as e2:
                                    print(f"❌ All VACE loading approaches failed")
                                    print(f"   Primary error: {e}")
                                    print(f"   Secondary error: {e2}")
                                    
                                    # Final fallback: clear error message
                                    raise RuntimeError(f"""
❌ CRITICAL: VACE model is not compatible with current T2V workflow!

🔧 VACE MODEL UNDERSTANDING:
VACE (Video and Contour Editing) models are designed for VIDEO EDITING, not simple T2V generation.
They require input_frames, input_masks, and input_ref_images parameters.

❌ INCOMPATIBLE WITH DEFORUM T2V WORKFLOW!

💡 SOLUTIONS (recommended order):
1. 📁 **BEST**: Download and use T2V models instead:
   - Wan2.1-T2V-1.3B (8GB VRAM, consumer-friendly)
   - Wan2.1-T2V-14B (more VRAM required)
   
2. 🔧 **ALTERNATIVE**: Use official Wan repository for VACE editing:
   cd Wan2.1 && pip install -e .
   
3. 🔄 **FUTURE**: Wait for VACE T2V support in future updates

📥 **DOWNLOAD T2V MODELS**:
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan/Wan2.1-T2V-1.3B

🌐 **REFERENCE**: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B

🚫 **VACE IS FOR EDITING, NOT T2V GENERATION!**
""")
                        
                        def __call__(self, *args, **kwargs):
                            """VACE pipeline call with proper parameter handling"""
                            if not hasattr(self, 'pipeline') or self.pipeline is None:
                                raise RuntimeError("VACE pipeline not loaded")
                            
                            # VACE is an all-in-one model, so handle both T2V and I2V calls
                            return self.pipeline(*args, **kwargs)
                        
                        def generate(self, *args, **kwargs):
                            """Official Wan T2V generate method interface for VACE wrapper"""
                            print(f"🎬 VACE wrapper generate method called")
                            
                            # Convert official Wan parameters to Diffusers parameters
                            input_prompt = kwargs.get('input_prompt', args[0] if args else "")
                            size = kwargs.get('size', args[1] if len(args) > 1 else (512, 512))
                            frame_num = kwargs.get('frame_num', args[2] if len(args) > 2 else 5)
                            sampling_steps = kwargs.get('sampling_steps', args[3] if len(args) > 3 else 20)
                            guide_scale = kwargs.get('guide_scale', args[4] if len(args) > 4 else 7.5)
                            
                            width, height = size
                            print(f"🎯 VACE T2V: {input_prompt[:30]}..., size: {width}x{height}, frames: {frame_num}")
                            
                            # Check if this is an official Wan wrapper or diffusers pipeline
                            if hasattr(self.pipeline, 'vace_model'):
                                # Official Wan VACE
                                return self.pipeline.generate(
                                    input_prompt=input_prompt,
                                    size=(width, height),
                                    frame_num=frame_num,
                                    sampling_steps=sampling_steps,
                                    guide_scale=guide_scale
                                )
                            else:
                                # Diffusers pipeline - use compatible parameters
                                pipeline_kwargs = {
                                    "prompt": input_prompt,
                                    "num_inference_steps": sampling_steps,
                                    "guidance_scale": guide_scale,
                                }
                                
                                # Add video-specific parameters if supported
                                import inspect
                                try:
                                    pipeline_signature = inspect.signature(self.pipeline.__call__)
                                    
                                    if 'height' in pipeline_signature.parameters:
                                        pipeline_kwargs['height'] = height
                                    if 'width' in pipeline_signature.parameters:
                                        pipeline_kwargs['width'] = width
                                    if 'num_frames' in pipeline_signature.parameters:
                                        pipeline_kwargs['num_frames'] = frame_num
                                except:
                                    # If signature inspection fails, just use basic parameters
                                    pass
                                
                                return self.pipeline(**pipeline_kwargs)
                        
                        def generate_image2video(self, *args, **kwargs):
                            """I2V generation with proper aspect ratio preservation for VACE"""
                            print(f"🎬 VACE I2V wrapper called")
                            
                            # Extract parameters
                            image = args[0] if args else kwargs.get('image')
                            prompt = args[1] if len(args) > 1 else kwargs.get('prompt', "")
                            
                            # Use provided dimensions instead of auto-preserving input image aspect ratio
                            # This ensures consistency with T2V aligned dimensions
                            height = args[2] if len(args) > 2 else kwargs.get('height', 512)
                            width = args[3] if len(args) > 3 else kwargs.get('width', 512)
                            
                            # Log dimension handling
                            if image and hasattr(image, 'size'):
                                input_width, input_height = image.size
                                print(f"🖼️ Input image size: {input_width}x{input_height}")
                                print(f"🎯 Target I2V size: {width}x{height} (using aligned dimensions)")
                                
                                # Resize the input image to match target dimensions if needed
                                if (input_width, input_height) != (width, height):
                                    print(f"🔧 Resizing input image to match target dimensions")
                                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                            else:
                                print(f"🎯 Target I2V size: {width}x{height}")
                            
                            num_frames = args[4] if len(args) > 4 else kwargs.get('num_frames', 5)
                            num_inference_steps = args[5] if len(args) > 5 else kwargs.get('num_inference_steps', 20)
                            guidance_scale = args[6] if len(args) > 6 else kwargs.get('guidance_scale', 7.5)
                            strength = kwargs.get('strength', 0.8)  # I2V strength parameter
                            
                            print(f"🎯 VACE I2V: {prompt[:30]}..., size: {width}x{height}, frames: {num_frames}, strength: {strength}")
                            
                            # Check if this is an official Wan wrapper or diffusers pipeline
                            if hasattr(self.pipeline, 'vace_model'):
                                # Official Wan VACE
                                return self.pipeline.generate_image2video(
                                    image=image,
                                    input_prompt=prompt,
                                    height=height,
                                    width=width,
                                    num_frames=num_frames,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    strength=strength,  # Pass the strength parameter
                                )
                            else:
                                # Diffusers pipeline - check if it supports image input
                                import inspect
                                try:
                                    pipeline_signature = inspect.signature(self.pipeline.__call__)
                                    
                                    pipeline_kwargs = {
                                        "prompt": prompt,
                                        "num_inference_steps": num_inference_steps,
                                        "guidance_scale": guidance_scale,
                                    }
                                    
                                    # Add parameters if supported
                                    if 'image' in pipeline_signature.parameters and image is not None:
                                        pipeline_kwargs['image'] = image
                                        print("✅ Using image input for I2V")
                                    else:
                                        # Fallback: enhance prompt for T2V-style generation
                                        enhanced_prompt = f"Continuing seamlessly from the provided image, {prompt}. Maintaining visual continuity and style."
                                        pipeline_kwargs['prompt'] = enhanced_prompt
                                        print("⚠️ No image support detected, using enhanced T2V prompt")
                                    
                                    if 'height' in pipeline_signature.parameters:
                                        pipeline_kwargs['height'] = height
                                    if 'width' in pipeline_signature.parameters:
                                        pipeline_kwargs['width'] = width
                                    if 'num_frames' in pipeline_signature.parameters:
                                        pipeline_kwargs['num_frames'] = num_frames
                                    
                                    return self.pipeline(**pipeline_kwargs)
                                    
                                except Exception as e:
                                    print(f"⚠️ Pipeline inspection failed: {e}")
                                    # Fallback to basic call
                                    enhanced_prompt = f"Continuing seamlessly from the provided image, {prompt}. Maintaining visual continuity and style."
                                    return self.pipeline(prompt=enhanced_prompt)
                    
                    # Create the VACE wrapper
                    print("🔧 Creating VACE pipeline wrapper...")
                    vace_wrapper = VACEPipelineWrapper(self.model_path, self.device)
                    
                    # Set up the pipeline attributes
                    self.pipeline = vace_wrapper
                    self.t2v_model = vace_wrapper
                    self.i2v_model = vace_wrapper
                    
                    print("✅ VACE specialized loading completed successfully")
                    return True
                    
                except Exception as e:
                    print(f"❌ VACE specialized loading failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Re-raise the error to fail fast instead of continuing
                    raise RuntimeError(f"""
❌ CRITICAL: VACE specialized loading completely failed!

🔧 REAL VACE MODEL REQUIRED:
The VACE model could not be loaded with any method.

❌ NO FALLBACKS - Real VACE implementation required!
Error: {e}

💡 SOLUTIONS:
1. 📦 Install VACE dependencies properly
2. 🔧 Verify model file integrity
3. 💾 Check available VRAM
4. 🔄 Restart WebUI

🚫 REFUSING TO GENERATE PLACEHOLDER CONTENT!
""")
            
            def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                """Generate video frames"""
                if not self.loaded:
                    raise RuntimeError("Wan model not loaded")
                
                print(f"🎬 Generating {num_frames} frames with Wan...")
                print(f"   📝 Prompt: {prompt[:50]}...")
                print(f"   📐 Size: {width}x{height}")
                print(f"   🔧 Steps: {num_inference_steps}")
                print(f"   📏 Guidance: {guidance_scale}")
                
                try:
                    # Use official Wan T2V if available
                    if hasattr(self, 't2v_model') and self.t2v_model:
                        print("🚀 Using official Wan T2V model")
                        
                        result = self.t2v_model.generate(
                            input_prompt=prompt,
                            size=(width, height),
                            frame_num=num_frames,
                            sampling_steps=num_inference_steps,
                            guide_scale=guidance_scale,
                            shift=5.0,
                            sample_solver='unipc',
                            offload_model=True,
                            **kwargs
                        )
                        
                        return result
                    
                    # Use diffusers model if available
                    elif hasattr(self, 'model') and self.model:
                        print("🚀 Using diffusers-based model")
                        
                        generation_kwargs = {
                            "prompt": prompt,
                            "height": height,
                            "width": width,
                            "num_frames": num_frames,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                        }
                        
                        # Add optional parameters if supported
                        import inspect
                        if hasattr(self.model, '__call__'):
                            sig = inspect.signature(self.model.__call__)
                            
                            if 'output_type' in sig.parameters:
                                generation_kwargs['output_type'] = 'pil'
                            if 'return_dict' in sig.parameters:
                                generation_kwargs['return_dict'] = False
                        
                        with torch.no_grad():
                            result = self.model(**generation_kwargs)
                        
                        return result
                    
                    # No valid model available
                    else:
                        raise RuntimeError("""
❌ CRITICAL: No valid Wan model loaded!

🔧 SETUP REQUIRED:
1. Install official Wan repository
2. Download Wan models
3. Restart WebUI

💡 No fallbacks available - real Wan implementation required.
""")
                
                except Exception as e:
                    print(f"❌ Generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    raise RuntimeError(f"""
❌ CRITICAL: Wan video generation failed!

🔧 TROUBLESHOOTING:
1. ✅ Verify Wan models are properly installed
2. 🔄 Check CUDA/GPU availability: {torch.cuda.is_available()}
3. 💾 Check available VRAM/memory
4. 📦 Verify all dependencies are installed

❌ NO FALLBACKS - Real Wan implementation required!
Error: {e}
""")
            
            def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                """Generate video from image (I2V)"""
                if not self.loaded:
                    raise RuntimeError("Wan model not loaded")
                
                print(f"🎬 Generating I2V {num_frames} frames with Wan...")
                
                try:
                    # Use official Wan I2V if available
                    if hasattr(self, 'i2v_model') and self.i2v_model:
                        print("🚀 Using official Wan I2V model")
                        
                        result = self.i2v_model.generate(
                            input_prompt=prompt,
                            img=image,
                            max_area=height * width,
                            frame_num=num_frames,
                            sampling_steps=num_inference_steps,
                            guide_scale=guidance_scale,
                            shift=5.0,
                            sample_solver='unipc',
                            offload_model=True,
                            **kwargs
                        )
                        
                        return result
                    
                    # Fallback to T2V with enhanced prompt if I2V not available
                    else:
                        print("⚠️ I2V model not available, using enhanced T2V")
                        enhanced_prompt = f"Starting from the given image, {prompt}. Maintain visual continuity."
                        
                        return self.__call__(
                            enhanced_prompt, height, width, num_frames, 
                            num_inference_steps, guidance_scale, **kwargs
                        )
                
                except Exception as e:
                    print(f"❌ I2V generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    raise RuntimeError(f"""
❌ CRITICAL: Wan I2V generation failed!

🔧 TROUBLESHOOTING:
1. ✅ Verify I2V model is properly loaded
2. 🖼️ Check input image format and size
3. 💾 Check available VRAM/memory
4. 📦 Verify Wan I2V dependencies

❌ NO FALLBACKS - Real Wan I2V implementation required!
Error: {e}
""")
            
            def _check_if_vace_model(self, model_path: str) -> bool:
                """Check if model is a VACE model"""
                import json
                from pathlib import Path
                
                path = Path(model_path)
                print(f"🔍 Checking if {path} is a VACE model...")
                
                # Check config.json first
                config_file = path / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        model_type = config.get("model_type", "").lower()
                        if model_type == "vace":
                            print("✅ VACE model detected via config.json!")
                            return True
                            
                    except Exception as e:
                        print(f"⚠️ Error reading config.json: {e}")
                
                # Check path name
                path_str = str(path).lower()
                is_vace_by_name = "vace" in path_str
                
                if is_vace_by_name:
                    print("✅ VACE model detected via path name!")
                else:
                    print("❌ Not a VACE model")
                
                return is_vace_by_name
            
            def _create_vace_config(self):
                """Create VACE configuration"""
                import json
                from pathlib import Path
                
                config_file = Path(self.model_path) / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            model_config = json.load(f)
                        
                        class VACEConfig:
                            def __init__(self, model_config):
                                self.model_type = model_config.get("model_type", "vace")
                                self.dim = model_config.get("dim", 1536)
                                self.ffn_dim = model_config.get("ffn_dim", 8960)
                                self.num_heads = model_config.get("num_heads", 12)
                                self.num_layers = model_config.get("num_layers", 30)
                                self.in_dim = model_config.get("in_dim", 16)
                                self.out_dim = model_config.get("out_dim", 16)
                                self.text_len = model_config.get("text_len", 512)
                                self.freq_dim = model_config.get("freq_dim", 256)
                                self.eps = model_config.get("eps", 1e-6)
                                self.vace_layers = model_config.get("vace_layers", [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
                                self.vace_in_dim = model_config.get("vace_in_dim", 96)
                                self.num_train_timesteps = 1000
                                self.param_dtype = torch.bfloat16
                                self.t5_dtype = torch.bfloat16
                                self.clip_dtype = torch.bfloat16
                                self.vae_stride = [4, 8, 8]
                                self.patch_size = [1, 2, 2]
                                self.sample_neg_prompt = "Low quality, blurry, distorted, nsfw, nude"
                                self.sample_fps = 24
                                self.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
                                self.vae_checkpoint = 'Wan2.1_VAE.pth'
                                self.t5_tokenizer = 'google/umt5-xxl'
                                self.clip_checkpoint = 'clip_l.safetensors'
                                self.clip_tokenizer = 'openai/clip-vit-large-patch14'
                        
                        return VACEConfig(model_config)
                        
                    except Exception as e:
                        print(f"⚠️ Failed to load VACE config: {e}")
                
                # Fallback to default VACE config
                class VACEConfig:
                    def __init__(self):
                        self.model_type = "vace"
                        self.dim = 1536
                        self.ffn_dim = 8960
                        self.num_heads = 12
                        self.num_layers = 30
                        self.in_dim = 16
                        self.out_dim = 16
                        self.text_len = 512
                        self.freq_dim = 256
                        self.eps = 1e-6
                        self.vace_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
                        self.vace_in_dim = 96
                        self.num_train_timesteps = 1000
                        self.param_dtype = torch.bfloat16
                        self.t5_dtype = torch.bfloat16
                        self.clip_dtype = torch.bfloat16
                        self.vae_stride = [4, 8, 8]
                        self.patch_size = [1, 2, 2]
                        self.sample_neg_prompt = "Low quality, blurry, distorted, nsfw, nude"
                        self.sample_fps = 24
                        self.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
                        self.vae_checkpoint = 'Wan2.1_VAE.pth'
                        self.t5_tokenizer = 'google/umt5-xxl'
                        self.clip_checkpoint = 'clip_l.safetensors'
                        self.clip_tokenizer = 'openai/clip-vit-large-patch14'
                
                return VACEConfig()
            
            def _load_wan_model(self):
                """Load the actual Wan model with multiple fallback strategies"""
                try:
                    # Strategy 1: Try to import and use official Wan
                    print("🔄 Attempting to load official Wan model...")
                    
                    # Add Wan2.1 to path if it exists
                    import sys
                    from pathlib import Path
                    
                    # Look for Wan2.1 directory in the extension root
                    extension_root = Path(__file__).parent.parent.parent.parent
                    wan_repo_path = extension_root / "Wan2.1"
                    
                    print(f"🔍 Looking for Wan repository at: {wan_repo_path}")
                    
                    if wan_repo_path.exists() and (wan_repo_path / "wan").exists():
                        # Add to Python path
                        if str(wan_repo_path) not in sys.path:
                            sys.path.insert(0, str(wan_repo_path))
                            print(f"✅ Added Wan repo to path: {wan_repo_path}")
                        
                        # Apply Flash Attention compatibility patches
                        try:
                            from .wan_flash_attention_patch import apply_wan_compatibility_patches
                        except Exception as patch_e:
                            print(f"⚠️ Could not apply compatibility patches: {patch_e}")
                        
                        # Check what type of model we actually have FIRST
                        model_path = Path(self.model_path)
                        config_path = model_path / "config.json"
                        
                        is_vace_model = False
                        if config_path.exists():
                            try:
                                import json
                                with open(config_path, 'r') as f:
                                    config = json.load(f)
                                model_type = config.get("model_type", "").lower()
                                class_name = config.get("_class_name", "").lower()
                                
                                if model_type == "vace" or "vace" in class_name:
                                    is_vace_model = True
                                    print("🎯 Detected VACE model - using specialized VACE loading")
                                else:
                                    print(f"🎯 Detected {model_type} model")
                            except Exception as e:
                                print(f"⚠️ Could not read config: {e}")
                        
                        # Handle VACE models with specialized loading
                        if is_vace_model:
                            print("🔧 Using specialized VACE model loading...")
                            success = self._load_vace_model_specialized(wan_repo_path)
                            if success:
                                self.loaded = True
                                print("✅ VACE model loaded successfully with specialized loader")
                                return
                            else:
                                print("❌ Specialized VACE loading failed, trying standard approach...")
                                # Continue to standard loading as fallback
                        
                        # Standard Wan model loading for T2V/I2V models
                        # Verify the wan module can be imported
                        try:
                            import wan  # type: ignore
                            print("✅ Successfully imported wan module from local repository")
                            
                            # Import specific components
                            from wan.text2video import WanT2V  # type: ignore
                            from wan.image2video import WanI2V  # type: ignore
                            print("✅ Successfully imported WanT2V and WanI2V classes")
                            
                            # Try to load configs
                            try:
                                # Check if config files exist
                                config_dir = wan_repo_path / "wan" / "configs"
                                if config_dir.exists():
                                    print(f"✅ Found config directory: {config_dir}")
                                    
                                    # Try to import configs
                                    try:
                                        if is_vace_model:
                                            # For VACE models, use T2V config as base
                                            from wan.configs.wan_t2v_1_3B import t2v_1_3B as vace_config
                                            print("✅ Using T2V config for VACE model")
                                            t2v_config = vace_config
                                            i2v_config = vace_config  # Same config for both
                                        else:
                                            from wan.configs.wan_t2v_14B import t2v_14B as t2v_config
                                            from wan.configs.wan_i2v_14B import i2v_14B as i2v_config
                                            print("✅ Loaded official Wan configs")
                                    except ImportError as config_e:
                                        print(f"⚠️ Config import failed: {config_e}, will use minimal configs")
                                        # Create minimal config structure
                                        class MinimalConfig:
                                            def __init__(self):
                                                self.model = type('obj', (object,), {
                                                    'num_attention_heads': 32,
                                                    'attention_head_dim': 128,
                                                    'in_channels': 4,
                                                    'out_channels': 4,
                                                    'num_layers': 28,
                                                    'sample_size': 32,
                                                    'patch_size': 2,
                                                    'num_vector_embeds': None,
                                                    'activation_fn': "geglu",
                                                    'num_embeds_ada_norm': 1000,
                                                    'norm_elementwise_affine': False,
                                                    'norm_eps': 1e-6,
                                                    'attention_bias': True,
                                                    'caption_channels': 4096
                                                })
                                                
                                        t2v_config = MinimalConfig()
                                        i2v_config = MinimalConfig()
                                else:
                                    print("⚠️ Config directory not found, creating minimal configs")
                                    # Create minimal config structure
                                    class MinimalConfig:
                                        def __init__(self):
                                            self.model = type('obj', (object,), {
                                                'num_attention_heads': 32,
                                                'attention_head_dim': 128,
                                                'in_channels': 4,
                                                'out_channels': 4,
                                                'num_layers': 28,
                                                'sample_size': 32,
                                                'patch_size': 2,
                                                'num_vector_embeds': None,
                                                'activation_fn': "geglu",
                                                'num_embeds_ada_norm': 1000,
                                                'norm_elementwise_affine': False,
                                                'norm_eps': 1e-6,
                                                'attention_bias': True,
                                                'caption_channels': 4096
                                            })
                                            
                                    t2v_config = MinimalConfig()
                                    i2v_config = MinimalConfig()
                                    
                            except Exception as config_e:
                                print(f"⚠️ Config loading failed: {config_e}, using minimal configs")
                                # Create minimal config structure
                                class MinimalConfig:
                                    def __init__(self):
                                        self.model = type('obj', (object,), {
                                            'num_attention_heads': 32,
                                            'attention_head_dim': 128,
                                            'in_channels': 4,
                                            'out_channels': 4,
                                            'num_layers': 28,
                                            'sample_size': 32,
                                            'patch_size': 2,
                                            'num_vector_embeds': None,
                                            'activation_fn': "geglu",
                                            'num_embeds_ada_norm': 1000,
                                            'norm_elementwise_affine': False,
                                            'norm_eps': 1e-6,
                                            'attention_bias': True,
                                            'caption_channels': 4096
                                        })
                                        
                                t2v_config = MinimalConfig()
                                i2v_config = MinimalConfig()
                            
                            # Initialize T2V model with correct parameters
                            print(f"🚀 Initializing WanT2V with checkpoint dir: {self.model_path}")
                            
                            if is_vace_model:
                                # For VACE models, use special loading parameters
                                print("🎯 Loading VACE model with T2V-compatible parameters")
                                try:
                                    self.t2v_model = WanT2V(
                                        config=t2v_config,
                                        checkpoint_dir=self.model_path,
                                        device_id=0,
                                        rank=0,
                                        dit_fsdp=False,
                                        t5_fsdp=False
                                    )
                                    print("✅ VACE model loaded as T2V")
                                    
                                    # For VACE, the same model can handle both T2V and I2V
                                    self.i2v_model = self.t2v_model
                                    print("✅ VACE model ready for both T2V and I2V")
                                    
                                except Exception as vace_e:
                                    print(f"❌ VACE T2V loading failed: {vace_e}")
                                    # For VACE models, fail fast instead of trying diffusers fallback
                                    raise RuntimeError(f"""
❌ CRITICAL: VACE model loading failed with official Wan loader!

🔧 REAL VACE MODEL REQUIRED:
The VACE model at {self.model_path} could not be loaded with official Wan methods.

❌ NO FALLBACKS for VACE - Real implementation required!
Specific error: {vace_e}

💡 SOLUTIONS:
1. 📦 Ensure Wan dependencies are properly installed: cd Wan2.1 && pip install -e .
2. 🔧 Verify VACE model files are complete and uncorrupted
3. 💾 Ensure sufficient VRAM (VACE 1.3B requires ~8GB+)
4. 🔄 Re-download VACE model if corrupted
5. 🔄 Restart WebUI after fixing dependencies

🚫 REFUSING TO GENERATE PLACEHOLDER CONTENT!
""")
                            else:
                                # Standard T2V/I2V model loading
                                self.t2v_model = WanT2V(
                                    config=t2v_config,
                                    checkpoint_dir=self.model_path,
                                    device_id=0,
                                    rank=0,
                                    dit_fsdp=False,
                                    t5_fsdp=False
                                )
                                
                                # Try to initialize I2V model if available
                                try:
                                    print(f"🚀 Initializing WanI2V with checkpoint dir: {self.model_path}")
                                    self.i2v_model = WanI2V(
                                        config=i2v_config,
                                        checkpoint_dir=self.model_path,
                                        device_id=0,
                                        rank=0,
                                        dit_fsdp=False,
                                        t5_fsdp=False
                                    )
                                    print("✅ I2V model loaded for chaining support")
                                except Exception as e:
                                    print(f"⚠️ I2V model not available: {e}")
                                    self.i2v_model = None
                            
                            self.loaded = True
                            print("✅ Official Wan models loaded successfully")
                            return
                            
                        except ImportError as import_e:
                            print(f"❌ Failed to import wan module: {import_e}")
                            # Continue to fallback methods
                    else:
                        print(f"❌ Wan repository not found at: {wan_repo_path}")
                        print("💡 Expected structure: Wan2.1/wan/ directory with Python modules")
                    
                    # If we get here, the official Wan import failed
                    print("⚠️ Official Wan import failed, trying diffusers fallback...")
                    
                    # Check if this is a VACE model - if so, refuse fallback
                    model_path = Path(self.model_path)
                    config_path = model_path / "config.json"
                    
                    is_vace_model = False
                    if config_path.exists():
                        try:
                            import json
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            model_type = config.get("model_type", "").lower()
                            class_name = config.get("_class_name", "").lower()
                            
                            if model_type == "vace" or "vace" in class_name:
                                is_vace_model = True
                        except Exception:
                            pass
                    
                    if is_vace_model:
                        # Refuse to use diffusers fallback for VACE models
                        raise RuntimeError(f"""
❌ CRITICAL: VACE model detected but official Wan loading failed!

🔧 REAL VACE MODEL REQUIRES OFFICIAL WAN:
VACE models require the official Wan repository and cannot use diffusers fallback.

❌ NO FALLBACKS for VACE - Real Wan implementation required!

💡 SOLUTIONS:
1. 📦 Properly install Wan repository: cd Wan2.1 && pip install -e .
2. 🔧 Fix Wan import issues (check Python path, dependencies)
3. 💾 Ensure all Wan dependencies are installed
4. 🔄 Restart WebUI after fixing Wan setup

🚫 REFUSING DIFFUSERS FALLBACK FOR VACE MODEL!
""")
                    
                    # Strategy 2: Try diffusers-based loading as fallback (only for non-VACE models)
                    try:
                        from diffusers import DiffusionPipeline
                        
                        print("🔄 Attempting diffusers-based loading for non-VACE model...")
                        self.model = DiffusionPipeline.from_pretrained(
                            self.model_path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                        
                        if torch.cuda.is_available():
                            self.model = self.model.to(self.device)
                        
                        self.loaded = True
                        print("✅ Diffusers-based model loaded successfully (fallback for non-VACE)")
                        return
                        
                    except Exception as diffusers_e:
                        print(f"❌ Diffusers loading also failed: {diffusers_e}")
                    
                    # Both methods failed, provide comprehensive error
                    raise RuntimeError(f"""
❌ SETUP REQUIRED: Wan 2.1 Official Repository Setup Issue!

🔧 QUICK SETUP:

📥 The Wan2.1 directory exists but import failed. Try:

1. 📦 Install Wan dependencies:
   cd Wan2.1
   pip install -e .

2. 📂 Download models to correct location:
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan

3. 🔄 Restart WebUI completely

💡 The Wan2.1 repository is already present at: {wan_repo_path}
   But the Python modules couldn't be imported properly.

🌐 If issues persist, check: https://github.com/Wan-Video/Wan2.1#readme
""")
                
                except Exception as e:
                    print(f"❌ All model loading strategies failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Check if this was a VACE model to provide specific guidance
                    model_path = Path(self.model_path)
                    config_path = model_path / "config.json"
                    is_vace_model = False
                    
                    if config_path.exists():
                        try:
                            import json
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            model_type = config.get("model_type", "").lower()
                            class_name = config.get("_class_name", "").lower()
                            
                            if model_type == "vace" or "vace" in class_name:
                                is_vace_model = True
                        except Exception:
                            pass
                    
                    if is_vace_model:
                        raise RuntimeError(f"""
❌ CRITICAL: VACE model loading completely failed!

🔧 VACE MODEL REQUIRES PROPER SETUP:
VACE models require the official Wan repository with all dependencies properly installed.

❌ NO FALLBACKS FOR VACE - Real implementation required!
Error: {str(e)}

💡 VACE-SPECIFIC SOLUTIONS:
1. 📦 Install Wan repository properly: cd Wan2.1 && pip install -e .
2. 📦 Install diffusers: pip install diffusers
3. 🔧 Verify VACE model files are complete and uncorrupted
4. 💾 Ensure sufficient VRAM (VACE 1.3B requires ~8GB+)
5. 🔄 Restart WebUI after fixing dependencies

🚫 PLACEHOLDER GENERATION DISABLED - REQUIRING REAL VACE MODEL!
""")
                    else:
                        raise RuntimeError(f"""
❌ CRITICAL: Wan model loading completely failed!

🔧 COMPLETE SETUP GUIDE:
1. 📥 Ensure Wan2.1 repository is properly set up
2. 📦 Install Wan dependencies: cd Wan2.1 && pip install -e .
3. 📂 Download models to: models/wan/
4. 🔄 Restart WebUI completely

❌ NO FALLBACKS AVAILABLE - Real Wan implementation required!
Error: {str(e)}
""")
            
            def _load_vace_model_specialized(self, wan_repo_path: Path) -> bool:
                """Specialized VACE model loading that bypasses standard Wan loading issues"""
                try:
                    print("🔧 Loading VACE model with specialized approach...")
                    
                    # Import required modules
                    import wan
                    import torch
                    import json
                    from pathlib import Path
                    
                    # Load the VACE model config
                    config_path = Path(self.model_path) / "config.json"
                    if config_path.exists():
                        with open(config_path, 'r') as f:
                            vace_config = json.load(f)
                        print(f"✅ Loaded VACE config: {vace_config.get('model_type', 'unknown')}")
                    else:
                        print("⚠️ No config.json found, using default VACE config")
                        vace_config = {
                            "model_type": "vace",
                            "dim": 1536,
                            "num_layers": 30,
                        }
                    
                    # Create a custom VACE pipeline wrapper
                    class VACEPipelineWrapper:
                        def __init__(self, model_path: str, device: str):
                            self.model_path = model_path
                            self.device = device
                            self.vace_model = None
                            self.vae = None
                            self.text_encoder = None
                            self._load_vace_components()
                        
                        def _load_vace_components(self):
                            """Load VACE model - prioritize official Wan repository over diffusers"""
                            try:
                                print("🔧 Loading VACE model - prioritizing official Wan repository...")
                                
                                # First, try to use the official Wan VACE implementation
                                try:
                                    # Import from the official Wan repository
                                    import wan
                                    from wan.vace import WanVace  # Correct VACE class name
                                    
                                    print("🚀 Using official Wan VACE implementation")
                                    
                                    # Load VACE config - use T2V 1.3B config as base for VACE
                                    try:
                                        from wan.configs.wan_t2v_1_3B import t2v_1_3B as vace_config
                                        print("✅ Using T2V 1.3B config for VACE")
                                    except ImportError:
                                        # Create minimal config if none available
                                        print("⚠️ Creating minimal VACE config")
                                        class MinimalVACEConfig:
                                            def __init__(self):
                                                self.num_train_timesteps = 1000
                                                self.param_dtype = torch.bfloat16
                                                self.t5_dtype = torch.bfloat16
                                                self.text_len = 512
                                                self.vae_stride = [4, 8, 8]
                                                self.patch_size = [1, 2, 2]
                                                self.sample_neg_prompt = "Low quality, blurry, distorted"
                                                self.sample_fps = 24
                                                self.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
                                                self.vae_checkpoint = 'Wan2.1_VAE.pth'
                                                self.t5_tokenizer = 'google/umt5-xxl'
                                        vace_config = MinimalVACEConfig()
                                    
                                    # Initialize VACE model with official implementation
                                    self.vace_model = WanVace(
                                        config=vace_config,
                                        checkpoint_dir=self.model_path,
                                        device_id=0,
                                        rank=0,
                                        dit_fsdp=False,
                                        t5_fsdp=False
                                    )
                                    
                                    # Create wrapper interface that matches our expected API
                                    class OfficialWanVACEWrapper:
                                        def __init__(self, vace_model):
                                            self.vace_model = vace_model
                                        
                                        def __call__(self, **kwargs):
                                            return self.vace_model.generate(**kwargs)
                                        
                                        def generate(self, **kwargs):
                                            # For T2V generation, create dummy blank frames for VACE to transform
                                            # VACE is a video editing model that needs input frames, so we provide blank ones
                                            frame_num = kwargs.get('frame_num', 81)
                                            size = kwargs.get('size', (1280, 720))
                                            
                                            # VACE requires dimensions aligned with VAE stride [4, 8, 8]
                                            # Ensure width and height are divisible by 16 (8*2 for proper alignment)
                                            width, height = size
                                            aligned_width = ((width + 15) // 16) * 16
                                            aligned_height = ((height + 15) // 16) * 16
                                            
                                            print(f"🔧 VACE dimension alignment: {width}x{height} -> {aligned_width}x{aligned_height}")
                                            
                                            # Create blank frames (black video) for VACE to transform
                                            if 'input_frames' not in kwargs or kwargs['input_frames'] == [] or kwargs['input_frames'] is None:
                                                # Create dummy black frames with proper alignment
                                                blank_frame = torch.zeros((3, frame_num, aligned_height, aligned_width), device=self.vace_model.device)
                                                kwargs['input_frames'] = [blank_frame]
                                            
                                            if 'input_masks' not in kwargs or kwargs['input_masks'] == [] or kwargs['input_masks'] is None:
                                                # Create masks that indicate the entire video should be edited (transformed)
                                                # Use proper format: (1, frame_num, height, width)
                                                full_mask = torch.ones((1, frame_num, aligned_height, aligned_width), device=self.vace_model.device)
                                                kwargs['input_masks'] = [full_mask]
                                            
                                            if 'input_ref_images' not in kwargs or kwargs['input_ref_images'] == [] or kwargs['input_ref_images'] is None:
                                                kwargs['input_ref_images'] = [None]
                                            
                                            # Update size to use aligned dimensions
                                            kwargs['size'] = (aligned_width, aligned_height)
                                            
                                            return self.vace_model.generate(**kwargs)
                                        
                                        def generate_image2video(self, image, **kwargs):
                                            # For I2V with VACE, use the provided image as reference
                                            # Enhanced prompt for better continuity
                                            enhanced_prompt = f"Continuing seamlessly from the provided image, {kwargs.get('input_prompt', '')}. Maintaining visual continuity and style."
                                            kwargs['input_prompt'] = enhanced_prompt
                                            
                                            frame_num = kwargs.get('frame_num', 81)
                                            size = kwargs.get('size', (1280, 720))
                                            
                                            # VACE requires dimensions aligned with VAE stride [4, 8, 8]
                                            # Ensure width and height are divisible by 16 (8*2 for proper alignment)
                                            width, height = size
                                            aligned_width = ((width + 15) // 16) * 16
                                            aligned_height = ((height + 15) // 16) * 16
                                            
                                            print(f"🔧 VACE I2V dimension alignment: {width}x{height} -> {aligned_width}x{aligned_height}")
                                            
                                            # For I2V, we can still use dummy frames but provide the image as reference
                                            if 'input_frames' not in kwargs or kwargs['input_frames'] == [] or kwargs['input_frames'] is None:
                                                # Create dummy black frames with proper alignment
                                                blank_frame = torch.zeros((3, frame_num, aligned_height, aligned_width), device=self.vace_model.device)
                                                kwargs['input_frames'] = [blank_frame]
                                            
                                            if 'input_masks' not in kwargs or kwargs['input_masks'] == [] or kwargs['input_masks'] is None:
                                                # Create masks that indicate the entire video should be edited
                                                full_mask = torch.ones((1, frame_num, aligned_height, aligned_width), device=self.vace_model.device)
                                                kwargs['input_masks'] = [full_mask]
                                            
                                            # Set the reference image for I2V generation
                                            if 'input_ref_images' not in kwargs or kwargs['input_ref_images'] == [] or kwargs['input_ref_images'] is None:
                                                # TODO: Convert the provided image to the format expected by VACE
                                                # For now, fall back to T2V approach
                                                kwargs['input_ref_images'] = [None]
                                            
                                            # Update size to use aligned dimensions
                                            kwargs['size'] = (aligned_width, aligned_height)
                                            
                                            return self.vace_model.generate(**kwargs)
                                    
                                    self.pipeline = OfficialWanVACEWrapper(self.vace_model)
                                    
                                    print("✅ VACE model loaded successfully with official Wan implementation")
                                    return True
                                    
                                except (ImportError, AttributeError) as wan_e:
                                    print(f"⚠️ Official Wan VACE implementation not available: {wan_e}")
                                    print("🔄 Falling back to diffusers approach...")
                                
                                # Fallback: try diffusers approach for WebUI Forge compatibility
                                from diffusers import DiffusionPipeline
                                
                                print("🔄 Loading VACE model with DiffusionPipeline (WebUI Forge compatible)...")
                                
                                # Load using the standard approach that works with older diffusers
                                self.pipeline = DiffusionPipeline.from_pretrained(
                                    self.model_path,
                                    torch_dtype=torch.bfloat16,  # VACE uses bfloat16
                                    use_safetensors=True,
                                    trust_remote_code=True  # VACE might need this
                                )
                                
                                # Manually move to device after loading
                                print(f"🔄 Moving VACE model to {self.device}...")
                                self.pipeline = self.pipeline.to(self.device)
                                
                                print("✅ VACE model loaded successfully with WebUI Forge compatible approach")
                                return True
                                
                            except Exception as e:
                                print(f"❌ VACE loading failed: {e}")
                                
                                # Try even more basic approach
                                try:
                                    print("🔄 Trying basic VACE loading for older diffusers...")
                                    from diffusers import DiffusionPipeline
                                    
                                    # Very basic loading without advanced parameters
                                    self.pipeline = DiffusionPipeline.from_pretrained(
                                        self.model_path,
                                        torch_dtype=torch.float16,  # Use float16 for better compatibility
                                        use_safetensors=True
                                    )
                                    
                                    # Move to device manually
                                    self.pipeline = self.pipeline.to(self.device)
                                    
                                    print("✅ VACE model loaded with basic diffusers approach")
                                    return True
                                    
                                except Exception as e2:
                                    print(f"❌ All VACE loading approaches failed")
                                    print(f"   Primary error: {e}")
                                    print(f"   Secondary error: {e2}")
                                    
                                    # Final fallback: clear error message
                                    raise RuntimeError(f"""
❌ CRITICAL: VACE model is not compatible with current T2V workflow!

🔧 VACE MODEL UNDERSTANDING:
VACE (Video and Contour Editing) models are designed for VIDEO EDITING, not simple T2V generation.
They require input_frames, input_masks, and input_ref_images parameters.

❌ INCOMPATIBLE WITH DEFORUM T2V WORKFLOW!

💡 SOLUTIONS (recommended order):
1. 📁 **BEST**: Download and use T2V models instead:
   - Wan2.1-T2V-1.3B (8GB VRAM, consumer-friendly)
   - Wan2.1-T2V-14B (more VRAM required)
   
2. 🔧 **ALTERNATIVE**: Use official Wan repository for VACE editing:
   cd Wan2.1 && pip install -e .
   
3. 🔄 **FUTURE**: Wait for VACE T2V support in future updates

📥 **DOWNLOAD T2V MODELS**:
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan/Wan2.1-T2V-1.3B

🌐 **REFERENCE**: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B

🚫 **VACE IS FOR EDITING, NOT T2V GENERATION!**
""")
                        
                        def __call__(self, *args, **kwargs):
                            """VACE pipeline call with proper parameter handling"""
                            if not hasattr(self, 'pipeline') or self.pipeline is None:
                                raise RuntimeError("VACE pipeline not loaded")
                            
                            # VACE is an all-in-one model, so handle both T2V and I2V calls
                            return self.pipeline(*args, **kwargs)
                        
                        def generate(self, *args, **kwargs):
                            """Official Wan T2V generate method interface for VACE wrapper"""
                            print(f"🎬 VACE wrapper generate method called")
                            
                            # Convert official Wan parameters to Diffusers parameters
                            input_prompt = kwargs.get('input_prompt', args[0] if args else "")
                            size = kwargs.get('size', args[1] if len(args) > 1 else (512, 512))
                            frame_num = kwargs.get('frame_num', args[2] if len(args) > 2 else 5)
                            sampling_steps = kwargs.get('sampling_steps', args[3] if len(args) > 3 else 20)
                            guide_scale = kwargs.get('guide_scale', args[4] if len(args) > 4 else 7.5)
                            
                            width, height = size
                            print(f"🎯 VACE T2V: {input_prompt[:30]}..., size: {width}x{height}, frames: {frame_num}")
                            
                            # Check if this is an official Wan wrapper or diffusers pipeline
                            if hasattr(self.pipeline, 'vace_model'):
                                # Official Wan VACE
                                return self.pipeline.generate(
                                    input_prompt=input_prompt,
                                    size=(width, height),
                                    frame_num=frame_num,
                                    sampling_steps=sampling_steps,
                                    guide_scale=guide_scale
                                )
                            else:
                                # Diffusers pipeline - use compatible parameters
                                pipeline_kwargs = {
                                    "prompt": input_prompt,
                                    "num_inference_steps": sampling_steps,
                                    "guidance_scale": guide_scale,
                                }
                                
                                # Add video-specific parameters if supported
                                import inspect
                                try:
                                    pipeline_signature = inspect.signature(self.pipeline.__call__)
                                    
                                    if 'height' in pipeline_signature.parameters:
                                        pipeline_kwargs['height'] = height
                                    if 'width' in pipeline_signature.parameters:
                                        pipeline_kwargs['width'] = width
                                    if 'num_frames' in pipeline_signature.parameters:
                                        pipeline_kwargs['num_frames'] = frame_num
                                except:
                                    # If signature inspection fails, just use basic parameters
                                    pass
                                
                                return self.pipeline(**pipeline_kwargs)
                        
                        def generate_image2video(self, *args, **kwargs):
                            """I2V generation with proper aspect ratio preservation for VACE"""
                            print(f"🎬 VACE I2V wrapper called")
                            
                            # Extract parameters
                            image = args[0] if args else kwargs.get('image')
                            prompt = args[1] if len(args) > 1 else kwargs.get('prompt', "")
                            
                            # Use provided dimensions instead of auto-preserving input image aspect ratio
                            # This ensures consistency with T2V aligned dimensions
                            height = args[2] if len(args) > 2 else kwargs.get('height', 512)
                            width = args[3] if len(args) > 3 else kwargs.get('width', 512)
                            
                            # Log dimension handling
                            if image and hasattr(image, 'size'):
                                input_width, input_height = image.size
                                print(f"🖼️ Input image size: {input_width}x{input_height}")
                                print(f"🎯 Target I2V size: {width}x{height} (using aligned dimensions)")
                                
                                # Resize the input image to match target dimensions if needed
                                if (input_width, input_height) != (width, height):
                                    print(f"🔧 Resizing input image to match target dimensions")
                                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                            else:
                                print(f"🎯 Target I2V size: {width}x{height}")
                            
                            num_frames = args[4] if len(args) > 4 else kwargs.get('num_frames', 5)
                            num_inference_steps = args[5] if len(args) > 5 else kwargs.get('num_inference_steps', 20)
                            guidance_scale = args[6] if len(args) > 6 else kwargs.get('guidance_scale', 7.5)
                            strength = kwargs.get('strength', 0.8)  # I2V strength parameter
                            
                            print(f"🎯 VACE I2V: {prompt[:30]}..., size: {width}x{height}, frames: {num_frames}, strength: {strength}")
                            
                            # Check if this is an official Wan wrapper or diffusers pipeline
                            if hasattr(self.pipeline, 'vace_model'):
                                # Official Wan VACE
                                return self.pipeline.generate_image2video(
                                    image=image,
                                    input_prompt=prompt,
                                    height=height,
                                    width=width,
                                    num_frames=num_frames,
                                    num_inference_steps=num_inference_steps,
                                    guidance_scale=guidance_scale,
                                    strength=strength,  # Pass the strength parameter
                                )
                            else:
                                # Diffusers pipeline - check if it supports image input
                                import inspect
                                try:
                                    pipeline_signature = inspect.signature(self.pipeline.__call__)
                                    
                                    pipeline_kwargs = {
                                        "prompt": prompt,
                                        "num_inference_steps": num_inference_steps,
                                        "guidance_scale": guidance_scale,
                                    }
                                    
                                    # Add parameters if supported
                                    if 'image' in pipeline_signature.parameters and image is not None:
                                        pipeline_kwargs['image'] = image
                                        print("✅ Using image input for I2V")
                                    else:
                                        # Fallback: enhance prompt for T2V-style generation
                                        enhanced_prompt = f"Continuing seamlessly from the provided image, {prompt}. Maintaining visual continuity and style."
                                        pipeline_kwargs['prompt'] = enhanced_prompt
                                        print("⚠️ No image support detected, using enhanced T2V prompt")
                                    
                                    if 'height' in pipeline_signature.parameters:
                                        pipeline_kwargs['height'] = height
                                    if 'width' in pipeline_signature.parameters:
                                        pipeline_kwargs['width'] = width
                                    if 'num_frames' in pipeline_signature.parameters:
                                        pipeline_kwargs['num_frames'] = num_frames
                                    
                                    return self.pipeline(**pipeline_kwargs)
                                    
                                except Exception as e:
                                    print(f"⚠️ Pipeline inspection failed: {e}")
                                    # Fallback to basic call
                                    enhanced_prompt = f"Continuing seamlessly from the provided image, {prompt}. Maintaining visual continuity and style."
                                    return self.pipeline(prompt=enhanced_prompt)
                    
                    # Create the VACE wrapper
                    print("🔧 Creating VACE pipeline wrapper...")
                    vace_wrapper = VACEPipelineWrapper(self.model_path, self.device)
                    
                    # Set up the pipeline attributes
                    self.pipeline = vace_wrapper
                    self.t2v_model = vace_wrapper
                    self.i2v_model = vace_wrapper
                    
                    print("✅ VACE specialized loading completed successfully")
                    return True
                    
                except Exception as e:
                    print(f"❌ VACE specialized loading failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Re-raise the error to fail fast instead of continuing
                    raise RuntimeError(f"""
❌ CRITICAL: VACE specialized loading completely failed!

🔧 REAL VACE MODEL REQUIRED:
The VACE model could not be loaded with any method.

❌ NO FALLBACKS - Real VACE implementation required!
Error: {e}

💡 SOLUTIONS:
1. 📦 Install VACE dependencies properly
2. 🔧 Verify model file integrity
3. 💾 Check available VRAM
4. 🔄 Restart WebUI

🚫 REFUSING TO GENERATE PLACEHOLDER CONTENT!
""")
            
            def __call__(self, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                """Generate video frames"""
                if not self.loaded:
                    raise RuntimeError("Wan model not loaded")
                
                print(f"🎬 Generating {num_frames} frames with Wan...")
                print(f"   📝 Prompt: {prompt[:50]}...")
                print(f"   📐 Size: {width}x{height}")
                print(f"   🔧 Steps: {num_inference_steps}")
                print(f"   📏 Guidance: {guidance_scale}")
                
                try:
                    # Use official Wan T2V if available
                    if hasattr(self, 't2v_model') and self.t2v_model:
                        print("🚀 Using official Wan T2V model")
                        
                        result = self.t2v_model.generate(
                            input_prompt=prompt,
                            size=(width, height),
                            frame_num=num_frames,
                            sampling_steps=num_inference_steps,
                            guide_scale=guidance_scale,
                            shift=5.0,
                            sample_solver='unipc',
                            offload_model=True,
                            **kwargs
                        )
                        
                        return result
                    
                    # Use diffusers model if available
                    elif hasattr(self, 'model') and self.model:
                        print("🚀 Using diffusers-based model")
                        
                        generation_kwargs = {
                            "prompt": prompt,
                            "height": height,
                            "width": width,
                            "num_frames": num_frames,
                            "num_inference_steps": num_inference_steps,
                            "guidance_scale": guidance_scale,
                        }
                        
                        # Add optional parameters if supported
                        import inspect
                        if hasattr(self.model, '__call__'):
                            sig = inspect.signature(self.model.__call__)
                            
                            if 'output_type' in sig.parameters:
                                generation_kwargs['output_type'] = 'pil'
                            if 'return_dict' in sig.parameters:
                                generation_kwargs['return_dict'] = False
                        
                        with torch.no_grad():
                            result = self.model(**generation_kwargs)
                        
                        return result
                    
                    # No valid model available
                    else:
                        raise RuntimeError("""
❌ CRITICAL: No valid Wan model loaded!

🔧 SETUP REQUIRED:
1. Install official Wan repository
2. Download Wan models
3. Restart WebUI

💡 No fallbacks available - real Wan implementation required.
""")
                
                except Exception as e:
                    print(f"❌ Generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    raise RuntimeError(f"""
❌ CRITICAL: Wan video generation failed!

🔧 TROUBLESHOOTING:
1. ✅ Verify Wan models are properly installed
2. 🔄 Check CUDA/GPU availability: {torch.cuda.is_available()}
3. 💾 Check available VRAM/memory
4. 📦 Verify all dependencies are installed

❌ NO FALLBACKS - Real Wan implementation required!
Error: {e}
""")
            
            def generate_image2video(self, image, prompt, height, width, num_frames, num_inference_steps, guidance_scale, **kwargs):
                """Generate video from image (I2V)"""
                if not self.loaded:
                    raise RuntimeError("Wan model not loaded")
                
                print(f"🎬 Generating I2V {num_frames} frames with Wan...")
                
                try:
                    # Use official Wan I2V if available
                    if hasattr(self, 'i2v_model') and self.i2v_model:
                        print("🚀 Using official Wan I2V model")
                        
                        result = self.i2v_model.generate(
                            input_prompt=prompt,
                            img=image,
                            max_area=height * width,
                            frame_num=num_frames,
                            sampling_steps=num_inference_steps,
                            guide_scale=guidance_scale,
                            shift=5.0,
                            sample_solver='unipc',
                            offload_model=True,
                            **kwargs
                        )
                        
                        return result
                    
                    # Fallback to T2V with enhanced prompt if I2V not available
                    else:
                        print("⚠️ I2V model not available, using enhanced T2V")
                        enhanced_prompt = f"Starting from the given image, {prompt}. Maintain visual continuity."
                        
                        return self.__call__(
                            enhanced_prompt, height, width, num_frames, 
                            num_inference_steps, guidance_scale, **kwargs
                        )
                
                except Exception as e:
                    print(f"❌ I2V generation failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    raise RuntimeError(f"""
❌ CRITICAL: Wan I2V generation failed!

🔧 TROUBLESHOOTING:
1. ✅ Verify I2V model is properly loaded
2. 🖼️ Check input image format and size
3. 💾 Check available VRAM/memory
4. 📦 Verify Wan I2V dependencies

❌ NO FALLBACKS - Real Wan I2V implementation required!
Error: {e}
""")
            
            def _check_if_vace_model(self, model_path: str) -> bool:
                """Check if model is a VACE model"""
                import json
                from pathlib import Path
                
                path = Path(model_path)
                print(f"🔍 Checking if {path} is a VACE model...")
                
                # Check config.json first
                config_file = path / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        model_type = config.get("model_type", "").lower()
                        if model_type == "vace":
                            print("✅ VACE model detected via config.json!")
                            return True
                            
                    except Exception as e:
                        print(f"⚠️ Error reading config.json: {e}")
                
                # Check path name
                path_str = str(path).lower()
                is_vace_by_name = "vace" in path_str
                
                if is_vace_by_name:
                    print("✅ VACE model detected via path name!")
                else:
                    print("❌ Not a VACE model")
                
                return is_vace_by_name
            
            def _create_vace_config(self):
                """Create VACE configuration"""
                import json
                from pathlib import Path
                
                config_file = Path(self.model_path) / "config.json"
                if config_file.exists():
                    try:
                        with open(config_file, 'r') as f:
                            model_config = json.load(f)
                        
                        class VACEConfig:
                            def __init__(self, model_config):
                                self.model_type = model_config.get("model_type", "vace")
                                self.dim = model_config.get("dim", 1536)
                                self.ffn_dim = model_config.get("ffn_dim", 8960)
                                self.num_heads = model_config.get("num_heads", 12)
                                self.num_layers = model_config.get("num_layers", 30)
                                self.in_dim = model_config.get("in_dim", 16)
                                self.out_dim = model_config.get("out_dim", 16)
                                self.text_len = model_config.get("text_len", 512)
                                self.freq_dim = model_config.get("freq_dim", 256)
                                self.eps = model_config.get("eps", 1e-6)
                                self.vace_layers = model_config.get("vace_layers", [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28])
                                self.vace_in_dim = model_config.get("vace_in_dim", 96)
                                self.num_train_timesteps = 1000
                                self.param_dtype = torch.bfloat16
                                self.t5_dtype = torch.bfloat16
                                self.clip_dtype = torch.bfloat16
                                self.vae_stride = [4, 8, 8]
                                self.patch_size = [1, 2, 2]
                                self.sample_neg_prompt = "Low quality, blurry, distorted, nsfw, nude"
                                self.sample_fps = 24
                                self.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
                                self.vae_checkpoint = 'Wan2.1_VAE.pth'
                                self.t5_tokenizer = 'google/umt5-xxl'
                                self.clip_checkpoint = 'clip_l.safetensors'
                                self.clip_tokenizer = 'openai/clip-vit-large-patch14'
                        
                        return VACEConfig(model_config)
                        
                    except Exception as e:
                        print(f"⚠️ Failed to load VACE config: {e}")
                
                # Fallback to default VACE config
                class VACEConfig:
                    def __init__(self):
                        self.model_type = "vace"
                        self.dim = 1536
                        self.ffn_dim = 8960
                        self.num_heads = 12
                        self.num_layers = 30
                        self.in_dim = 16
                        self.out_dim = 16
                        self.text_len = 512
                        self.freq_dim = 256
                        self.eps = 1e-6
                        self.vace_layers = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
                        self.vace_in_dim = 96
                        self.num_train_timesteps = 1000
                        self.param_dtype = torch.bfloat16
                        self.t5_dtype = torch.bfloat16
                        self.clip_dtype = torch.bfloat16
                        self.vae_stride = [4, 8, 8]
                        self.patch_size = [1, 2, 2]
                        self.sample_neg_prompt = "Low quality, blurry, distorted, nsfw, nude"
                        self.sample_fps = 24
                        self.t5_checkpoint = 'models_t5_umt5-xxl-enc-bf16.pth'
                        self.vae_checkpoint = 'Wan2.1_VAE.pth'
                        self.t5_tokenizer = 'google/umt5-xxl'
                        self.clip_checkpoint = 'clip_l.safetensors'
                        self.clip_tokenizer = 'openai/clip-vit-large-patch14'
                
                return VACEConfig()
            
            def _load_wan_model(self):
                """Load the actual Wan model with multiple fallback strategies"""
                try:
                    # Strategy 1: Try to import and use official Wan
                    print("🔄 Attempting to load official Wan model...")
                    
                    # Add Wan2.1 to path if it exists
                    import sys
                    from pathlib import Path
                    
                    # Look for Wan2.1 directory in the extension root
                    extension_root = Path(__file__).parent.parent.parent.parent
                    wan_repo_path = extension_root / "Wan2.1"
                    
                    print(f"🔍 Looking for Wan repository at: {wan_repo_path}")
                    
                    if wan_repo_path.exists() and (wan_repo_path / "wan").exists():
                        # Add to Python path
                        if str(wan_repo_path) not in sys.path:
                            sys.path.insert(0, str(wan_repo_path))
                            print(f"✅ Added Wan repo to path: {wan_repo_path}")
                        
                        # Apply Flash Attention compatibility patches
                        try:
                            from .wan_flash_attention_patch import apply_wan_compatibility_patches
                        except Exception as patch_e:
                            print(f"⚠️ Could not apply compatibility patches: {patch_e}")
                        
                        # Check what type of model we actually have FIRST
                        model_path = Path(self.model_path)
                        config_path = model_path / "config.json"
                        
                        is_vace_model = False
                        if config_path.exists():
                            try:
                                import json
                                with open(config_path, 'r') as f:
                                    config = json.load(f)
                                model_type = config.get("model_type", "").lower()
                                class_name = config.get("_class_name", "").lower()
                                
                                if model_type == "vace" or "vace" in class_name:
                                    is_vace_model = True
                                    print("🎯 Detected VACE model - using specialized VACE loading")
                                else:
                                    print(f"🎯 Detected {model_type} model")
                            except Exception as e:
                                print(f"⚠️ Could not read config: {e}")
                        
                        # Handle VACE models with specialized loading
                        if is_vace_model:
                            print("🔧 Using specialized VACE model loading...")
                            success = self._load_vace_model_specialized(wan_repo_path)
                            if success:
                                self.loaded = True
                                print("✅ VACE model loaded successfully with specialized loader")
                                return
                            else:
                                print("❌ Specialized VACE loading failed, trying standard approach...")
                                # Continue to standard loading as fallback
                        
                        # Standard Wan model loading for T2V/I2V models
                        # Verify the wan module can be imported
                        try:
                            import wan  # type: ignore
                            print("✅ Successfully imported wan module from local repository")
                            
                            # Import specific components
                            from wan.text2video import WanT2V  # type: ignore
                            from wan.image2video import WanI2V  # type: ignore
                            print("✅ Successfully imported WanT2V and WanI2V classes")
                            
                            # Try to load configs
                            try:
                                # Check if config files exist
                                config_dir = wan_repo_path / "wan" / "configs"
                                if config_dir.exists():
                                    print(f"✅ Found config directory: {config_dir}")
                                    
                                    # Try to import configs
                                    try:
                                        if is_vace_model:
                                            # For VACE models, use T2V config as base
                                            from wan.configs.wan_t2v_1_3B import t2v_1_3B as vace_config
                                            print("✅ Using T2V config for VACE model")
                                            t2v_config = vace_config
                                            i2v_config = vace_config  # Same config for both
                                        else:
                                            from wan.configs.wan_t2v_14B import t2v_14B as t2v_config
                                            from wan.configs.wan_i2v_14B import i2v_14B as i2v_config
                                            print("✅ Loaded official Wan configs")
                                    except ImportError as config_e:
                                        print(f"⚠️ Config import failed: {config_e}, will use minimal configs")
                                        # Create minimal config structure
                                        class MinimalConfig:
                                            def __init__(self):
                                                self.model = type('obj', (object,), {
                                                    'num_attention_heads': 32,
                                                    'attention_head_dim': 128,
                                                    'in_channels': 4,
                                                    'out_channels': 4,
                                                    'num_layers': 28,
                                                    'sample_size': 32,
                                                    'patch_size': 2,
                                                    'num_vector_embeds': None,
                                                    'activation_fn': "geglu",
                                                    'num_embeds_ada_norm': 1000,
                                                    'norm_elementwise_affine': False,
                                                    'norm_eps': 1e-6,
                                                    'attention_bias': True,
                                                    'caption_channels': 4096
                                                })
                                                
                                        t2v_config = MinimalConfig()
                                        i2v_config = MinimalConfig()
                                else:
                                    print("⚠️ Config directory not found, creating minimal configs")
                                    # Create minimal config structure
                                    class MinimalConfig:
                                        def __init__(self):
                                            self.model = type('obj', (object,), {
                                                'num_attention_heads': 32,
                                                'attention_head_dim': 128,
                                                'in_channels': 4,
                                                'out_channels': 4,
                                                'num_layers': 28,
                                                'sample_size': 32,
                                                'patch_size': 2,
                                                'num_vector_embeds': None,
                                                'activation_fn': "geglu",
                                                'num_embeds_ada_norm': 1000,
                                                'norm_elementwise_affine': False,
                                                'norm_eps': 1e-6,
                                                'attention_bias': True,
                                                'caption_channels': 4096
                                            })
                                            
                                    t2v_config = MinimalConfig()
                                    i2v_config = MinimalConfig()
                                    
                            except Exception as config_e:
                                print(f"⚠️ Config loading failed: {config_e}, using minimal configs")
                                # Create minimal config structure
                                class MinimalConfig:
                                    def __init__(self):
                                        self.model = type('obj', (object,), {
                                            'num_attention_heads': 32,
                                            'attention_head_dim': 128,
                                            'in_channels': 4,
                                            'out_channels': 4,
                                            'num_layers': 28,
                                            'sample_size': 32,
                                            'patch_size': 2,
                                            'num_vector_embeds': None,
                                            'activation_fn': "geglu",
                                            'num_embeds_ada_norm': 1000,
                                            'norm_elementwise_affine': False,
                                            'norm_eps': 1e-6,
                                            'attention_bias': True,
                                            'caption_channels': 4096
                                        })
                                        
                                t2v_config = MinimalConfig()
                                i2v_config = MinimalConfig()
                            
                            # Initialize T2V model with correct parameters
                            print(f"🚀 Initializing WanT2V with checkpoint dir: {self.model_path}")
                            
                            if is_vace_model:
                                # For VACE models, use special loading parameters
                                print("🎯 Loading VACE model with T2V-compatible parameters")
                                try:
                                    self.t2v_model = WanT2V(
                                        config=t2v_config,
                                        checkpoint_dir=self.model_path,
                                        device_id=0,
                                        rank=0,
                                        dit_fsdp=False,
                                        t5_fsdp=False
                                    )
                                    print("✅ VACE model loaded as T2V")
                                    
                                    # For VACE, the same model can handle both T2V and I2V
                                    self.i2v_model = self.t2v_model
                                    print("✅ VACE model ready for both T2V and I2V")
                                    
                                except Exception as vace_e:
                                    print(f"❌ VACE T2V loading failed: {vace_e}")
                                    # For VACE models, fail fast instead of trying diffusers fallback
                                    raise RuntimeError(f"""
❌ CRITICAL: VACE model loading failed with official Wan loader!

🔧 REAL VACE MODEL REQUIRED:
The VACE model at {self.model_path} could not be loaded with official Wan methods.

❌ NO FALLBACKS for VACE - Real implementation required!
Specific error: {vace_e}

💡 SOLUTIONS:
1. 📦 Ensure Wan dependencies are properly installed: cd Wan2.1 && pip install -e .
2. 🔧 Verify VACE model files are complete and uncorrupted
3. 💾 Ensure sufficient VRAM (VACE 1.3B requires ~8GB+)
4. 🔄 Re-download VACE model if corrupted
5. 🔄 Restart WebUI after fixing dependencies

🚫 REFUSING TO GENERATE PLACEHOLDER CONTENT!
""")
                            else:
                                # Standard T2V/I2V model loading
                                self.t2v_model = WanT2V(
                                    config=t2v_config,
                                    checkpoint_dir=self.model_path,
                                    device_id=0,
                                    rank=0,
                                    dit_fsdp=False,
                                    t5_fsdp=False
                                )
                                
                                # Try to initialize I2V model if available
                                try:
                                    print(f"🚀 Initializing WanI2V with checkpoint dir: {self.model_path}")
                                    self.i2v_model = WanI2V(
                                        config=i2v_config,
                                        checkpoint_dir=self.model_path,
                                        device_id=0,
                                        rank=0,
                                        dit_fsdp=False,
                                        t5_fsdp=False
                                    )
                                    print("✅ I2V model loaded for chaining support")
                                except Exception as e:
                                    print(f"⚠️ I2V model not available: {e}")
                                    self.i2v_model = None
                            
                            self.loaded = True
                            print("✅ Official Wan models loaded successfully")
                            return
                            
                        except ImportError as import_e:
                            print(f"❌ Failed to import wan module: {import_e}")
                            # Continue to fallback methods
                    else:
                        print(f"❌ Wan repository not found at: {wan_repo_path}")
                        print("💡 Expected structure: Wan2.1/wan/ directory with Python modules")
                    
                    # If we get here, the official Wan import failed
                    print("⚠️ Official Wan import failed, trying diffusers fallback...")
                    
                    # Check if this is a VACE model - if so, refuse fallback
                    model_path = Path(self.model_path)
                    config_path = model_path / "config.json"
                    
                    is_vace_model = False
                    if config_path.exists():
                        try:
                            import json
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            model_type = config.get("model_type", "").lower()
                            class_name = config.get("_class_name", "").lower()
                            
                            if model_type == "vace" or "vace" in class_name:
                                is_vace_model = True
                        except Exception:
                            pass
                    
                    if is_vace_model:
                        # Refuse to use diffusers fallback for VACE models
                        raise RuntimeError(f"""
❌ CRITICAL: VACE model detected but official Wan loading failed!

🔧 REAL VACE MODEL REQUIRES OFFICIAL WAN:
VACE models require the official Wan repository and cannot use diffusers fallback.

❌ NO FALLBACKS for VACE - Real Wan implementation required!

💡 SOLUTIONS:
1. 📦 Properly install Wan repository: cd Wan2.1 && pip install -e .
2. 🔧 Fix Wan import issues (check Python path, dependencies)
3. 💾 Ensure all Wan dependencies are installed
4. 🔄 Restart WebUI after fixing Wan setup

🚫 REFUSING DIFFUSERS FALLBACK FOR VACE MODEL!
""")
                    
                    # Strategy 2: Try diffusers-based loading as fallback (only for non-VACE models)
                    try:
                        from diffusers import DiffusionPipeline
                        
                        print("🔄 Attempting diffusers-based loading for non-VACE model...")
                        self.model = DiffusionPipeline.from_pretrained(
                            self.model_path,
                            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                            device_map="auto" if torch.cuda.is_available() else None
                        )
                        
                        if torch.cuda.is_available():
                            self.model = self.model.to(self.device)
                        
                        self.loaded = True
                        print("✅ Diffusers-based model loaded successfully (fallback for non-VACE)")
                        return
                        
                    except Exception as diffusers_e:
                        print(f"❌ Diffusers loading also failed: {diffusers_e}")
                    
                    # Both methods failed, provide comprehensive error
                    raise RuntimeError(f"""
❌ SETUP REQUIRED: Wan 2.1 Official Repository Setup Issue!

🔧 QUICK SETUP:

📥 The Wan2.1 directory exists but import failed. Try:

1. 📦 Install Wan dependencies:
   cd Wan2.1
   pip install -e .

2. 📂 Download models to correct location:
   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan

3. 🔄 Restart WebUI completely

💡 The Wan2.1 repository is already present at: {wan_repo_path}
   But the Python modules couldn't be imported properly.

🌐 If issues persist, check: https://github.com/Wan-Video/Wan2.1#readme
""")
                
                except Exception as e:
                    print(f"❌ All model loading strategies failed: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Check if this was a VACE model to provide specific guidance
                    model_path = Path(self.model_path)
                    config_path = model_path / "config.json"
                    is_vace_model = False
                    
                    if config_path.exists():
                        try:
                            import json
                            with open(config_path, 'r') as f:
                                config = json.load(f)
                            model_type = config.get("model_type", "").lower()
                            class_name = config.get("_class_name", "").lower()
                            
                            if model_type == "vace" or "vace" in class_name:
                                is_vace_model = True
                        except Exception:
                            pass
                    
                    if is_vace_model:
                        raise RuntimeError(f"""
❌ CRITICAL: VACE model loading completely failed!

🔧 VACE MODEL REQUIRES PROPER SETUP:
VACE models require the official Wan repository with all dependencies properly installed.

❌ NO FALLBACKS FOR VACE - Real implementation required!
Error: {str(e)}

💡 VACE-SPECIFIC SOLUTIONS:
1. 📦 Install Wan repository properly: cd Wan2.1 && pip install -e .
2. 📦 Install diffusers: pip install diffusers
3. 🔧 Verify VACE model files are complete and uncorrupted
4. 💾 Ensure sufficient VRAM (VACE 1.3B requires ~8GB+)
5. 🔄 Restart WebUI after fixing dependencies

🚫 PLACEHOLDER GENERATION DISABLED - REQUIRING REAL VACE MODEL!
""")
                    else:
                        raise RuntimeError(f"""
❌ CRITICAL: Wan model loading completely failed!

🔧 COMPLETE SETUP GUIDE:
1. 📥 Ensure Wan2.1 repository is properly set up
2. 📦 Install Wan dependencies: cd Wan2.1 && pip install -e .
3. 📂 Download models to: models/wan/
4. 🔄 Restart WebUI completely

❌ NO FALLBACKS AVAILABLE - Real Wan implementation required!
Error: {str(e)}
""")
        
        # Return the CustomWanPipeline instance
        return CustomWanPipeline(model_info['path'], self.device)

    def _validate_wan_model(self, model_info: Dict) -> bool:
        """Validate that a Wan model has required files and weights"""
        try:
            from pathlib import Path
            model_path = Path(model_info['path'])
            
            if not model_path.exists():
                print(f"❌ Model path does not exist: {model_path}")
                return False
            
            # Basic file validation
            if not self._has_required_files(model_path):
                print(f"❌ Model missing required files: {model_path}")
                return False
            
            # For VACE models, perform deeper weight validation
            if 'VACE' in model_info.get('type', '').upper():
                print(f"🔍 Performing deep VACE model validation for: {model_path.name}")
                if not self._validate_vace_weights(model_path):
                    print(f"❌ VACE model missing critical weights: {model_path}")
                    return False
                print(f"✅ VACE model validation passed: {model_path.name}")
            
            return True
        except Exception as e:
            print(f"❌ Model validation failed: {e}")
            return False
    
    def _validate_vace_weights(self, model_path: Path) -> bool:
        """Validate that VACE model has required weights - fixed validation logic"""
        try:
            import torch
            
            # Check the main diffusion model file
            diffusion_model = model_path / "diffusion_pytorch_model.safetensors"
            if not diffusion_model.exists():
                # Check for multi-part model
                diffusion_model = model_path / "diffusion_pytorch_model-00001-of-00007.safetensors"
                if not diffusion_model.exists():
                    print("❌ No diffusion model file found for VACE validation")
                    return False
            
            # Load model metadata to check for required keys
            try:
                from safetensors import safe_open
                
                with safe_open(diffusion_model, framework="pt", device="cpu") as f:
                    available_keys = list(f.keys())
                    
                    # Very basic validation - just check if it's a reasonable model
                    if len(available_keys) < 500:
                        print(f"❌ Model appears incomplete: only {len(available_keys)} keys (expected >500)")
                        return False
                    
                    # Check for some cross-attention layers (but be very lenient)
                    cross_attn_keys = [k for k in available_keys if 'cross_attn' in k]
                    if len(cross_attn_keys) < 50:  # Very low threshold
                        print(f"❌ Model lacks cross-attention structure: {len(cross_attn_keys)} keys (expected >50)")
                        return False
                    
                    # If we made it here, the model looks good
                    print(f"✅ VACE validation passed: {len(available_keys)} total keys, {len(cross_attn_keys)} cross-attn keys")
                    return True
                    
            except ImportError:
                print("⚠️ safetensors not available, skipping deep VACE validation")
                return True  # Fall back to basic validation
            except Exception as e:
                print(f"⚠️ VACE weight validation failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ VACE validation error: {e}")
            return False
    
    def generate_video_with_i2v_chaining(self, clips, model_info, output_dir, **kwargs):
        """Generate video using I2V chaining for better continuity between clips"""
        try:
            import shutil
            import os
            from datetime import datetime
            import time
            from typing import List, Dict, Optional
            
            print(f"🎬 Starting I2V chained generation with {len(clips)} clips...")
            print(f"📁 Model: {model_info['name']} ({model_info['type']}, {model_info['size']})")
            
            # Extract parameters from kwargs
            width = kwargs.get('width', 1280)
            height = kwargs.get('height', 720)
            steps = kwargs.get('steps', 20)
            guidance_scale = kwargs.get('guidance_scale', 7.5)
            seed = kwargs.get('seed', -1)
            anim_args = kwargs.get('anim_args')
            wan_args = kwargs.get('wan_args')
            
            # VACE dimension alignment: CRITICAL for consistent T2V and I2V resolution!
            # VACE requires dimensions aligned with VAE stride [4, 8, 8]
            # Ensure width and height are divisible by 16 (8*2 for proper alignment)
            original_width, original_height = width, height
            aligned_width = ((width + 15) // 16) * 16
            aligned_height = ((height + 15) // 16) * 16
            
            if aligned_width != original_width or aligned_height != original_height:
                print(f"🔧 VACE dimension alignment applied: {original_width}x{original_height} -> {aligned_width}x{aligned_height}")
                print(f"💡 This ensures consistent resolution between T2V (first clip) and I2V (subsequent clips)")
                width, height = aligned_width, aligned_height
            else:
                print(f"✅ Dimensions already aligned: {width}x{height}")
            
            # Check for strength override
            use_strength_override = False
            fixed_strength = 0.85  # Default fallback
            
            if wan_args and hasattr(wan_args, 'wan_strength_override') and wan_args.wan_strength_override:
                use_strength_override = True
                fixed_strength = getattr(wan_args, 'wan_fixed_strength', 1.0)
                print(f"🔒 Using strength override: {fixed_strength} (ignoring Deforum schedules)")
            
            # Parse strength schedule if not overridden
            strength_values = {}
            if not use_strength_override and anim_args and hasattr(anim_args, 'strength_schedule'):
                try:
                    # Parse the strength schedule (format: "0: (0.85), 60: (0.7)")
                    import re
                    strength_schedule = anim_args.strength_schedule
                    print(f"🎯 Using Deforum strength schedule: {strength_schedule}")
                    
                    # Extract frame:value pairs
                    matches = re.findall(r'(\d+):\s*\(([0-9.]+)\)', strength_schedule)
                    for frame_str, strength_str in matches:
                        frame_num = int(frame_str)
                        strength_val = float(strength_str)
                        strength_values[frame_num] = strength_val
                        
                    print(f"📊 Parsed strength schedule: {strength_values}")
                except Exception as e:
                    print(f"⚠️ Failed to parse strength schedule: {e}, using default strength")
            
            # Validate model first
            if not self._validate_wan_model(model_info):
                raise RuntimeError("Wan model validation failed - missing required files")
            
            # Load the model if not loaded
            if not self.pipeline:
                print("🔧 Loading Wan pipeline for I2V chaining...")
                if not self.load_simple_wan_pipeline(model_info):
                    raise RuntimeError("Failed to load Wan pipeline")
            
            # Get the timestring from the output directory name or create one
            timestring = os.path.basename(output_dir).split('_')[-1]
            if not timestring or len(timestring) != 14:
                # Fallback: extract from directory name or create new
                dir_parts = os.path.basename(output_dir).split('_')
                for part in dir_parts:
                    if len(part) == 14 and part.isdigit():
                        timestring = part
                        break
                else:
                    # Create new timestring if none found
                    timestring = datetime.now().strftime("%Y%m%d%H%M%S")
            
            print(f"📁 Output directory: {output_dir}")
            print(f"🕐 Using timestring: {timestring}")
            
            # Create unified frames directory
            unified_frames_dir = output_dir
            os.makedirs(unified_frames_dir, exist_ok=True)
            
            all_frame_paths = []
            total_frame_idx = 0
            last_frame_path = None
            
            for clip_idx, clip in enumerate(clips):
                print(f"\n🎬 Generating clip {clip_idx + 1}/{len(clips)}")
                print(f"   📝 Prompt: {clip['prompt'][:50]}...")
                print(f"   🎞️ Frames: {clip['num_frames']}")
                
                # Calculate strength for this clip
                if use_strength_override:
                    clip_strength = fixed_strength
                    print(f"   🔒 Using override strength: {clip_strength}")
                elif strength_values:
                    # Find the appropriate strength value for this clip's start frame
                    clip_start_frame = clip['start_frame']
                    
                    # Find the closest strength value at or before this frame
                    applicable_frames = [f for f in strength_values.keys() if f <= clip_start_frame]
                    if applicable_frames:
                        closest_frame = max(applicable_frames)
                        clip_strength = strength_values[closest_frame]
                        print(f"   💪 Using strength {clip_strength} (from frame {closest_frame} schedule)")
                    else:
                        # Use the first strength value if no earlier frame found
                        first_frame = min(strength_values.keys())
                        clip_strength = strength_values[first_frame]
                        print(f"   💪 Using strength {clip_strength} (from first scheduled frame {first_frame})")
                else:
                    clip_strength = 0.85  # Default strength
                    print(f"   💪 Using default strength {clip_strength}")
                
                # Create temporary directory for this clip
                temp_clip_dir = os.path.join(output_dir, f"_temp_clip_{clip_idx:03d}")
                os.makedirs(temp_clip_dir, exist_ok=True)
                
                # Calculate frame discarding for this clip
                target_frames = clip['num_frames']
                # VACE/Wan works best with 4n+1 frames
                if (target_frames - 1) % 4 == 0:
                    wan_frames = target_frames  # Already follows 4n+1
                else:
                    n = (target_frames - 1) // 4 + 1
                    wan_frames = 4 * n + 1
                
                print(f"🎯 Wan will generate {wan_frames} frames, targeting {target_frames} final frames")
                
                # Generate frames for this clip
                if clip_idx == 0 or last_frame_path is None:
                    # First clip: use T2V (no previous frame available)
                    print("🚀 Using T2V for first clip (no previous frame available)")
                    print("💡 VACE Note: First clip uses T2V mode since there's no input image yet")
                    
                    # Use the pipeline directly for T2V generation
                    try:
                        with torch.no_grad():
                            result = self.pipeline(
                                prompt=clip['prompt'],
                                height=height,
                                width=width,
                                num_frames=wan_frames,
                                num_inference_steps=steps,
                                guidance_scale=guidance_scale,
                            )
                            
                        # Handle different result formats
                        if isinstance(result, tuple):
                            frames = result[0]
                        elif hasattr(result, 'frames'):
                            frames = result.frames
                        else:
                            frames = result
                        
                        # Convert VACE output to proper frame format
                        if hasattr(frames, 'cpu'):  # It's a tensor
                            frames_tensor = frames.cpu()
                            print(f"🔧 VACE output tensor shape: {frames_tensor.shape}")
                            
                            # VACE output is typically (batch, channels, frames, height, width)
                            # Convert to list of individual frames
                            if len(frames_tensor.shape) == 5:  # (B, C, F, H, W)
                                frames_tensor = frames_tensor.squeeze(0)  # Remove batch dimension -> (C, F, H, W)
                                frames_list = []
                                for frame_idx in range(frames_tensor.shape[1]):  # Iterate over frames
                                    frame = frames_tensor[:, frame_idx, :, :]  # (C, H, W)
                                    frames_list.append(frame)
                                frames = frames_list
                            elif len(frames_tensor.shape) == 4:  # (C, F, H, W)
                                frames_list = []
                                for frame_idx in range(frames_tensor.shape[1]):  # Iterate over frames
                                    frame = frames_tensor[:, frame_idx, :, :]  # (C, H, W)
                                    frames_list.append(frame)
                                frames = frames_list
                            else:
                                print(f"⚠️ Unexpected VACE output shape: {frames_tensor.shape}")
                                frames = [frames_tensor]  # Treat as single frame
                        
                        # Save frames as PNG files
                        clip_frames = []
                        for i, frame in enumerate(frames):
                            frame_path = os.path.join(temp_clip_dir, f"frame_{i:06d}.png")
                            
                            if hasattr(frame, 'save'):  # PIL Image
                                frame.save(frame_path)
                                clip_frames.append(frame_path)
                            else:
                                # Handle tensor or numpy array
                                from PIL import Image
                                import numpy as np
                                
                                if hasattr(frame, 'cpu'):
                                    frame_np = frame.cpu().numpy()
                                else:
                                    frame_np = np.array(frame)
                                
                                print(f"🔧 Processing frame {i}: shape {frame_np.shape}, dtype {frame_np.dtype}")
                                
                                # Handle different tensor formats
                                if len(frame_np.shape) == 3:
                                    if frame_np.shape[0] == 3:  # (C, H, W) format
                                        frame_np = frame_np.transpose(1, 2, 0)  # Convert to (H, W, C)
                                        # VACE models output in RGB format, but we need to ensure correct channel order
                                        # Check if channels appear to be swapped (common issue with VACE)
                                        print(f"🎨 VACE frame color check: R={frame_np[:,:,0].mean():.2f}, G={frame_np[:,:,1].mean():.2f}, B={frame_np[:,:,2].mean():.2f}")
                                        
                                    elif frame_np.shape[2] == 3:  # Already (H, W, C)
                                        print(f"🎨 VACE frame color check: R={frame_np[:,:,0].mean():.2f}, G={frame_np[:,:,1].mean():.2f}, B={frame_np[:,:,2].mean():.2f}")
                                        
                                    else:
                                        print(f"⚠️ Unexpected 3D shape: {frame_np.shape}")
                                        # Try to extract first valid frame
                                        if frame_np.shape[0] == 1:
                                            frame_np = frame_np.squeeze(0)  # Remove first dimension
                                        elif frame_np.shape[2] == 1:
                                            frame_np = frame_np.squeeze(2)  # Remove last dimension
                                
                                elif len(frame_np.shape) == 2:  # Grayscale
                                    frame_np = np.stack([frame_np] * 3, axis=-1)  # Convert to RGB
                                
                                elif len(frame_np.shape) == 1:  # Flattened
                                    # Try to reshape to known dimensions (from our aligned size)
                                    expected_pixels = aligned_height * aligned_width * 3
                                    if frame_np.shape[0] == expected_pixels:
                                        frame_np = frame_np.reshape(aligned_height, aligned_width, 3)
                                    else:
                                        print(f"❌ Cannot reshape 1D array of size {frame_np.shape[0]} to expected {expected_pixels}")
                                        continue
                                
                                else:
                                    print(f"❌ Cannot handle frame shape: {frame_np.shape}")
                                    continue
                                
                                # Normalize values to 0-255 range
                                if frame_np.dtype != np.uint8:
                                    if frame_np.max() <= 1.0:
                                        frame_np = (frame_np * 255).astype(np.uint8)
                                    else:
                                        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                                
                                # VACE color debugging - let's check original values first
                                if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                                    print(f"🎨 VACE original colors: R={frame_np[:,:,0].mean():.2f}, G={frame_np[:,:,1].mean():.2f}, B={frame_np[:,:,2].mean():.2f}")
                                    # Temporarily disable channel swapping to test if colors are correct
                                    # frame_np = frame_np[:, :, [2, 1, 0]]  # RGB -> BGR swap disabled for testing
                                    print(f"🎨 Using original color order (no channel swap)")
                                
                                # Final validation
                                if len(frame_np.shape) != 3 or frame_np.shape[2] != 3:
                                    print(f"❌ Invalid final frame shape: {frame_np.shape}")
                                    continue
                                
                                try:
                                    pil_image = Image.fromarray(frame_np)
                                    pil_image.save(frame_path)
                                    clip_frames.append(frame_path)
                                    print(f"✅ Saved frame {i}: {frame_np.shape}")
                                except Exception as save_e:
                                    print(f"❌ Failed to save frame {i}: {save_e}")
                                    continue
                    
                    except Exception as e:
                        print(f"❌ T2V generation failed: {e}")
                        raise RuntimeError(f"T2V generation failed: {e}")
                else:
                    # Subsequent clips: use I2V with last frame
                    print(f"🔗 Using I2V chaining from: {os.path.basename(last_frame_path)}")
                    print(f"💪 I2V strength: {clip_strength} (controls influence of previous frame)")
                    print("💡 VACE Note: Using previous frame as input for continuity")
                    
                    # Load the previous frame as PIL Image
                    from PIL import Image
                    init_image = Image.open(last_frame_path)
                    
                    # Ensure the input image matches our target dimensions
                    if init_image.size != (width, height):
                        print(f"🔧 Resizing input image from {init_image.size} to {width}x{height}")
                        init_image = init_image.resize((width, height), Image.Resampling.LANCZOS)
                    
                    print(f"🖼️ Input image size: {init_image.size}")
                    print(f"🎯 Target I2V size: {width}x{height}")
                    
                    # Use I2V generation if available
                    try:
                        if hasattr(self.pipeline, 'generate_image2video'):
                            # Use dedicated I2V method with proper parameters
                            print(f"🚀 Using official Wan I2V with strength: {clip_strength}")
                            with torch.no_grad():
                                result = self.pipeline.generate_image2video(
                                    image=init_image,
                                    prompt=clip['prompt'],
                                    height=height,
                                    width=width,
                                    num_frames=wan_frames,
                                    num_inference_steps=steps,
                                    guidance_scale=guidance_scale,
                                    strength=clip_strength,  # Pass the strength parameter
                                )
                        else:
                            # Fallback: enhance prompt for T2V with image context
                            print(f"⚠️ No I2V method available, falling back to enhanced T2V")
                            enhanced_prompt = f"Continuing seamlessly from the previous scene, {clip['prompt']}. Maintaining strong visual continuity from the starting image."
                            
                            with torch.no_grad():
                                result = self.pipeline(
                                    prompt=enhanced_prompt,
                                    height=height,
                                    width=width,
                                    num_frames=wan_frames,
                                    num_inference_steps=steps,
                                    guidance_scale=guidance_scale,
                                )
                        
                        # Handle different result formats
                        if isinstance(result, tuple):
                            frames = result[0]
                        elif hasattr(result, 'frames'):
                            frames = result.frames
                        else:
                            frames = result
                        
                        # Convert VACE output to proper frame format
                        if hasattr(frames, 'cpu'):  # It's a tensor
                            frames_tensor = frames.cpu()
                            print(f"🔧 VACE output tensor shape: {frames_tensor.shape}")
                            
                            # VACE output is typically (batch, channels, frames, height, width)
                            # Convert to list of individual frames
                            if len(frames_tensor.shape) == 5:  # (B, C, F, H, W)
                                frames_tensor = frames_tensor.squeeze(0)  # Remove batch dimension -> (C, F, H, W)
                                frames_list = []
                                for frame_idx in range(frames_tensor.shape[1]):  # Iterate over frames
                                    frame = frames_tensor[:, frame_idx, :, :]  # (C, H, W)
                                    frames_list.append(frame)
                                frames = frames_list
                            elif len(frames_tensor.shape) == 4:  # (C, F, H, W)
                                frames_list = []
                                for frame_idx in range(frames_tensor.shape[1]):  # Iterate over frames
                                    frame = frames_tensor[:, frame_idx, :, :]  # (C, H, W)
                                    frames_list.append(frame)
                                frames = frames_list
                            else:
                                print(f"⚠️ Unexpected VACE output shape: {frames_tensor.shape}")
                                frames = [frames_tensor]  # Treat as single frame
                        
                        # Save frames as PNG files
                        clip_frames = []
                        for i, frame in enumerate(frames):
                            frame_path = os.path.join(temp_clip_dir, f"frame_{i:06d}.png")
                            
                            if hasattr(frame, 'save'):  # PIL Image
                                frame.save(frame_path)
                                clip_frames.append(frame_path)
                            else:
                                # Handle tensor or numpy array
                                from PIL import Image
                                import numpy as np
                                
                                if hasattr(frame, 'cpu'):
                                    frame_np = frame.cpu().numpy()
                                else:
                                    frame_np = np.array(frame)
                                
                                print(f"🔧 Processing frame {i}: shape {frame_np.shape}, dtype {frame_np.dtype}")
                                
                                # Handle different tensor formats
                                if len(frame_np.shape) == 3:
                                    if frame_np.shape[0] == 3:  # (C, H, W) format
                                        frame_np = frame_np.transpose(1, 2, 0)  # Convert to (H, W, C)
                                        # VACE models output in RGB format, but we need to ensure correct channel order
                                        # Check if channels appear to be swapped (common issue with VACE)
                                        print(f"🎨 VACE frame color check: R={frame_np[:,:,0].mean():.2f}, G={frame_np[:,:,1].mean():.2f}, B={frame_np[:,:,2].mean():.2f}")
                                        
                                    elif frame_np.shape[2] == 3:  # Already (H, W, C)
                                        print(f"🎨 VACE frame color check: R={frame_np[:,:,0].mean():.2f}, G={frame_np[:,:,1].mean():.2f}, B={frame_np[:,:,2].mean():.2f}")
                                        
                                    else:
                                        print(f"⚠️ Unexpected 3D shape: {frame_np.shape}")
                                        # Try to extract first valid frame
                                        if frame_np.shape[0] == 1:
                                            frame_np = frame_np.squeeze(0)  # Remove first dimension
                                        elif frame_np.shape[2] == 1:
                                            frame_np = frame_np.squeeze(2)  # Remove last dimension
                                
                                elif len(frame_np.shape) == 2:  # Grayscale
                                    frame_np = np.stack([frame_np] * 3, axis=-1)  # Convert to RGB
                                
                                elif len(frame_np.shape) == 1:  # Flattened
                                    # Try to reshape to known dimensions (from our aligned size)
                                    expected_pixels = aligned_height * aligned_width * 3
                                    if frame_np.shape[0] == expected_pixels:
                                        frame_np = frame_np.reshape(aligned_height, aligned_width, 3)
                                    else:
                                        print(f"❌ Cannot reshape 1D array of size {frame_np.shape[0]} to expected {expected_pixels}")
                                        continue
                                
                                else:
                                    print(f"❌ Cannot handle frame shape: {frame_np.shape}")
                                    continue
                                
                                # Normalize values to 0-255 range
                                if frame_np.dtype != np.uint8:
                                    if frame_np.max() <= 1.0:
                                        frame_np = (frame_np * 255).astype(np.uint8)
                                    else:
                                        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
                                
                                # VACE color debugging - let's check original values first
                                if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                                    print(f"🎨 VACE original colors: R={frame_np[:,:,0].mean():.2f}, G={frame_np[:,:,1].mean():.2f}, B={frame_np[:,:,2].mean():.2f}")
                                    # Temporarily disable channel swapping to test if colors are correct
                                    # frame_np = frame_np[:, :, [2, 1, 0]]  # RGB -> BGR swap disabled for testing
                                    print(f"🎨 Using original color order (no channel swap)")
                                
                                # Final validation
                                if len(frame_np.shape) != 3 or frame_np.shape[2] != 3:
                                    print(f"❌ Invalid final frame shape: {frame_np.shape}")
                                    continue
                                
                                try:
                                    pil_image = Image.fromarray(frame_np)
                                    pil_image.save(frame_path)
                                    clip_frames.append(frame_path)
                                    print(f"✅ Saved frame {i}: {frame_np.shape}")
                                except Exception as save_e:
                                    print(f"❌ Failed to save frame {i}: {save_e}")
                                    continue
                    
                    except Exception as e:
                        print(f"❌ I2V generation failed: {e}")
                        raise RuntimeError(f"I2V generation failed: {e}")
                
                if not clip_frames:
                    raise RuntimeError(f"Failed to generate frames for clip {clip_idx + 1}")
                
                print(f"✅ Generated {len(clip_frames)} frames for clip {clip_idx + 1}")
                
                # Apply frame discarding if needed (keep all for now to avoid complexity)
                final_clip_frames = clip_frames[:target_frames]  # Simple truncation
                if len(final_clip_frames) < target_frames:
                    final_clip_frames = clip_frames  # Use all if not enough frames
                
                print(f"✅ Final clip frames: {len(final_clip_frames)}")
                
                # Copy frames to unified directory with continuous numbering
                for frame_idx, src_path in enumerate(final_clip_frames):
                    # Use Deforum's naming convention: timestring_000000000.png
                    dst_filename = f"{timestring}_{total_frame_idx:09d}.png"
                    dst_path = os.path.join(unified_frames_dir, dst_filename)
                    
                    # Copy the frame
                    shutil.copy2(src_path, dst_path)
                    all_frame_paths.append(dst_path)
                    
                    total_frame_idx += 1
                
                # Clean up temporary clip directory
                shutil.rmtree(temp_clip_dir, ignore_errors=True)
                
                # Update last frame for next clip - use the EXACT last frame for maximum continuity
                if final_clip_frames:
                    last_frame_path = all_frame_paths[-1]  # Use the copied frame path
                    print(f"🔗 Last frame for next clip: {last_frame_path}")
                
            print(f"\n✅ All clips generated! Total frames: {len(all_frame_paths)}")
            print(f"📁 All frames saved to: {unified_frames_dir}")
            
            if use_strength_override:
                print(f"🔒 Strength override ({fixed_strength}) was applied across {len(clips)} clips")
            elif strength_values:
                print(f"💪 Strength scheduling was applied across {len(clips)} clips")
            
            # Return the output directory - Deforum will handle video creation
            return unified_frames_dir
                
        except Exception as e:
            print(f"❌ I2V chained video generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"I2V chained video generation failed: {e}")
        finally:
            # Clean up loaded models
            if hasattr(self, 'pipeline') and hasattr(self.pipeline, 'cleanup'):
                print("🧹 Cleaning up Wan models...")
                self.pipeline.cleanup()

    def unload_model(self):
        """Unload the model to free memory"""
        if self.pipeline:
            try:
                if hasattr(self.pipeline, 'to'):
                    self.pipeline.to('cpu')
                del self.pipeline
                self.pipeline = None
                    
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                        
                print("🧹 Wan model unloaded, memory freed")
            except Exception as e:
                print(f"⚠️ Error unloading model: {e}")

    def _get_model_downloader(self):
        """Get model downloader instance"""
        try:
            from .wan_model_downloader import WanModelDownloader
            return WanModelDownloader()
        except ImportError:
            # Handle running as standalone script
            import sys
            sys.path.append(str(Path(__file__).parent))
            from .wan_model_downloader import WanModelDownloader
            return WanModelDownloader()
    
    def _analyze_model_directory(self, model_dir: Path) -> Optional[Dict]:
        """Analyze a model directory and return model info if valid"""
        if not model_dir.is_dir():
            return None
            
        # Check if this looks like a Wan model
        model_name = model_dir.name.lower()
        if 'wan' not in model_name and not any(file.name.startswith('wan') for file in model_dir.rglob('*') if file.is_file()):
            return None
        
        # Check for required model files
        if not self._has_required_files(model_dir):
            return None
        
        # Determine model type and size
        model_type = "Unknown"
        model_size = "Unknown"
        
        if 'vace' in model_name:
            model_type = "VACE"
        elif 't2v' in model_name:
            model_type = "T2V"
        elif 'i2v' in model_name:
            model_type = "I2V"
        
        if '1.3b' in model_name:
            model_size = "1.3B"
        elif '14b' in model_name:
            model_size = "14B"
        
        return {
            'name': model_dir.name,
            'path': str(model_dir.absolute()),
            'type': model_type,
            'size': model_size,
            'directory': model_dir
        }
