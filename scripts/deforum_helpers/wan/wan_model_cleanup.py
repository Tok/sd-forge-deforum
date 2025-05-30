#!/usr/bin/env python3
"""
Wan Model Cleanup Utility

This script helps users identify and clean up corrupted or incomplete Wan models.
It can be run standalone to check model integrity and optionally delete/redownload corrupted models.
"""

import os
import sys
from pathlib import Path
import shutil

def find_wan_models():
    """Find all Wan model directories"""
    # Get script directory and work relative to webui root
    script_dir = Path(__file__).parent
    webui_root = script_dir.parent.parent.parent.parent  # Go up to webui root
    
    search_paths = [
        webui_root / "models/wan",
        webui_root / "models/Wan",
        Path("models/wan"),  # Current directory
        Path("models/Wan"),
    ]
    
    print(f"🔍 Searching for models in:")
    for path in search_paths:
        print(f"   - {path.absolute()}")
    
    models = []
    for base_path in search_paths:
        if base_path.exists():
            print(f"✅ Found directory: {base_path}")
            
            # Look for direct model files in this directory
            if _looks_like_model_dir(base_path):
                models.append(base_path)
                print(f"   📁 Found model directory: {base_path.name}")
            
            # Also look in subdirectories
            for item in base_path.iterdir():
                if item.is_dir() and 'wan' in item.name.lower():
                    if _looks_like_model_dir(item):
                        models.append(item)
                        print(f"   📁 Found model directory: {item.name}")
                elif item.is_dir() and _looks_like_model_dir(item):
                    # Check subdirectories that might contain models even if they don't have 'wan' in name
                    models.append(item)  
                    print(f"   📁 Found model directory: {item.name}")
        else:
            print(f"❌ Directory not found: {base_path}")
    
    return models

def _looks_like_model_dir(path: Path) -> bool:
    """Check if a directory looks like it contains a model"""
    if not path.is_dir():
        return False
    
    # Check for common model files
    has_model_files = (
        (path / "model_index.json").exists() or
        (path / "config.json").exists() or
        (path / "transformer").exists() or
        any(f.name.endswith(".safetensors") for f in path.rglob("*.safetensors")) or
        any(f.name.endswith(".pth") for f in path.rglob("*.pth")) or
        any(f.name.startswith("diffusion_pytorch_model") for f in path.iterdir()) or
        any(f.name.startswith("models_") for f in path.iterdir())
    )
    
    # Don't treat directories with only repository files as models
    repo_files = {'.gitattributes', '.git', 'README.md', 'LICENSE.txt', 'assets', 'examples'}
    only_repo_files = all(f.name in repo_files or f.name.startswith('.') for f in path.iterdir() if f.is_file())
    
    return has_model_files and not only_repo_files

def validate_vace_model(model_path: Path) -> tuple[bool, list]:
    """Validate a VACE model for required I2V components"""
    required_files = [
        "model_index.json",
        "scheduler/scheduler_config.json", 
        "text_encoder/config.json",
        "tokenizer/tokenizer.json",
        "transformer/config.json",
        "vae/config.json"
    ]
    
    # VACE-specific I2V cross-attention keys (check in transformer weights)
    transformer_path = model_path / "transformer" / "diffusion_pytorch_model.safetensors"
    missing_files = []
    
    # Check basic structure
    for file_path in required_files:
        if not (model_path / file_path).exists():
            missing_files.append(str(file_path))
    
    # Check for I2V cross-attention weights if transformer exists
    if transformer_path.exists():
        try:
            from safetensors import safe_open
            required_i2v_keys = [
                'blocks.0.cross_attn.k_img.weight',
                'blocks.0.cross_attn.v_img.weight', 
                'blocks.0.cross_attn.norm_k_img.weight',
                'blocks.0.cross_attn.norm_v_img.weight',
                'blocks.0.cross_attn.norm_q_img.weight'
            ]
            
            with safe_open(transformer_path, framework="pt", device="cpu") as f:
                existing_keys = list(f.keys())
                for key in required_i2v_keys:
                    if key not in existing_keys:
                        missing_files.append(f"transformer I2V key: {key}")
                        
        except Exception as e:
            missing_files.append(f"Error reading transformer weights: {e}")
    
    is_valid = len(missing_files) == 0
    return is_valid, missing_files

def main():
    print("🔍 Wan Model Cleanup Utility")
    print("=" * 50)
    
    models = find_wan_models()
    if not models:
        print("❌ No Wan models found in common locations")
        return
    
    print(f"📁 Found {len(models)} Wan model(s):")
    
    corrupted_models = []
    valid_models = []
    
    for i, model in enumerate(models, 1):
        print(f"\n{i}. Checking: {model.name}")
        print(f"   Path: {model}")
        
        if 'vace' in model.name.lower():
            is_valid, missing = validate_vace_model(model)
            if is_valid:
                print(f"   ✅ Valid VACE model")
                valid_models.append(model)
            else:
                print(f"   ❌ Corrupted VACE model")
                print(f"   Missing: {len(missing)} components")
                for missing_item in missing[:3]:  # Show first 3
                    print(f"     - {missing_item}")
                if len(missing) > 3:
                    print(f"     - ... and {len(missing) - 3} more")
                corrupted_models.append(model)
        else:
            # Basic check for legacy models and unknown types
            if (model / "model_index.json").exists():
                print(f"   ✅ Valid legacy model")
                valid_models.append(model)
            else:
                # Check if it has any recognizable Wan model structure
                has_valid_structure = (
                    (model / "model_index.json").exists() or
                    (model / "transformer").exists() or
                    any(f.name.startswith("wan") for f in model.rglob("*.pth")) or
                    any(f.name.startswith("wan") for f in model.rglob("*.safetensors")) or
                    len(list(model.rglob("*.safetensors"))) > 0  # Has any safetensors files
                )
                
                if has_valid_structure:
                    print(f"   ✅ Valid legacy model (has recognizable structure)")
                    valid_models.append(model)
                else:
                    print(f"   ❌ Invalid/leftover files (not a proper Wan model)")
                    print(f"   Contains: {[f.name for f in model.iterdir()][:5]}...")  # Show first 5 files
                    corrupted_models.append(model)
    
    print(f"\n📊 Summary:")
    print(f"   ✅ Valid models: {len(valid_models)}")
    print(f"   ❌ Corrupted models: {len(corrupted_models)}")
    
    if corrupted_models:
        print(f"\n🔧 Corrupted models found:")
        for model in corrupted_models:
            print(f"   - {model.name} ({model})")
        
        response = input(f"\n❓ Delete {len(corrupted_models)} corrupted model(s)? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            for model in corrupted_models:
                print(f"🗑️ Deleting: {model}")
                try:
                    shutil.rmtree(model)
                    print(f"   ✅ Deleted successfully")
                except Exception as e:
                    print(f"   ❌ Failed to delete: {e}")
            
            print(f"\n💡 To redownload models:")
            print(f"   huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir models/wan/Wan2.1-VACE-1.3B")
            print(f"   huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/wan/Wan2.1-VACE-14B")
        else:
            print(f"   Models left unchanged. You can delete them manually if needed.")

if __name__ == "__main__":
    main() 