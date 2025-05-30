#!/usr/bin/env python3
"""
Wan Model Validator - Advanced validation with hash checking and integrity verification
"""

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import shutil
import time

class WanModelValidator:
    """Advanced Wan model validator with proper hash verification"""
    
    def __init__(self):
        # Only look in Wan-specific directories
        self.base_paths = [
            Path("models/wan"),
            Path("models/Wan"), 
            Path("../../models/wan"),
            Path("../../models/Wan")
        ]
        
    def discover_models(self) -> List[Dict[str, Any]]:
        """Discover only actual Wan models with detailed metadata"""
        models = []
        processed_paths = set()
        
        for base_path in self.base_paths:
            if not base_path.exists():
                continue
                
            print(f"🔍 Scanning Wan directory: {base_path}")
            
            # Look for Wan model directories (not individual files)
            for item in base_path.iterdir():
                if not item.is_dir():
                    continue
                    
                # Skip if we've already processed this path (avoid duplicates)
                abs_path = item.resolve()
                if abs_path in processed_paths:
                    continue
                processed_paths.add(abs_path)
                
                # Check if this looks like a Wan model directory
                if self._is_wan_model_directory(item):
                    model_info = self._get_model_info(item)
                    if model_info:
                        models.append(model_info)
                        print(f"✅ Found Wan model: {model_info['name']} ({model_info['type']})")
                else:
                    # Skip non-Wan directories
                    print(f"⏭️ Skipping non-Wan directory: {item.name}")
        
        return models
    
    def _is_wan_model_directory(self, path: Path) -> bool:
        """Check if directory contains a Wan model"""
        if not path.is_dir():
            return False
            
        # Must have Wan in the name or be a known Wan model structure
        name_lower = path.name.lower()
        has_wan_name = any(keyword in name_lower for keyword in [
            'wan', 'vace', 't2v', 'i2v'
        ])
        
        # Check for Wan model structure
        has_wan_structure = any([
            (path / "config.json").exists(),
            (path / "diffusion_pytorch_model.safetensors").exists(),
            (path / "Wan2.1_VAE.pth").exists(),
            (path / "models_t5_umt5-xxl-enc-bf16.pth").exists(),
            any(f.name.startswith("wan") for f in path.rglob("*.pth")),
            any(f.name.startswith("wan") for f in path.rglob("*.safetensors")),
            (path / "google" / "umt5-xxl").exists()
        ])
        
        return has_wan_name and has_wan_structure
    
    def _get_model_info(self, model_path: Path) -> Optional[Dict[str, Any]]:
        """Get detailed information about a Wan model"""
        try:
            # Calculate directory size
            total_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            
            # Determine model type and name
            name = model_path.name
            model_type = self._determine_model_type(model_path, name)
            
            return {
                'name': name,
                'type': model_type,
                'path': str(model_path),
                'size': total_size,
                'size_formatted': self._format_size(total_size)
            }
            
        except Exception as e:
            print(f"❌ Error getting info for {model_path}: {e}")
            return None
    
    def _determine_model_type(self, model_path: Path, name: str) -> str:
        """Determine the type of Wan model"""
        name_lower = name.lower()
        
        # Check for VACE models (all-in-one)
        if 'vace' in name_lower:
            return 'VACE'
        elif 't2v' in name_lower:
            return 'T2V'
        elif 'i2v' in name_lower:
            return 'I2V'
        else:
            # Check structure to determine type
            if (model_path / "diffusion_pytorch_model.safetensors").exists():
                # Could be any type, check config
                config_path = model_path / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                        # Try to determine from config
                        if any('cross_attn' in str(config).lower() for key in config):
                            return 'VACE'  # Likely has cross attention for I2V
                    except:
                        pass
                return 'Unknown'
            else:
                return 'Legacy'
        
    def validate_model_integrity(self, model_path: Path) -> Dict[str, Any]:
        """Comprehensive model integrity validation"""
        results = {
            'path': str(model_path),
            'valid': True,
            'errors': [],
            'warnings': [],
            'checks': {}
        }
        
        print(f"\n🔍 Validating model: {model_path.name}")
        print("-" * 50)
        
        # 1. Basic structure check
        structure_ok = self._check_basic_structure(model_path)
        results['checks']['structure'] = structure_ok
        if structure_ok:
            print("✅ Basic structure: VALID")
        else:
            print("❌ Basic structure: INVALID")
            results['errors'].append("Missing required files (config.json or model files)")
            results['valid'] = False
            
        # 2. File size validation
        size_ok = self._validate_file_sizes(model_path)
        results['checks']['file_sizes'] = size_ok
        if size_ok:
            print("✅ File sizes: VALID")
        else:
            print("⚠️ File sizes: Some files are suspiciously small")
            results['warnings'].append("Some files may be incomplete (very small sizes)")
            
        # 3. JSON config validation
        config_ok = self._validate_json_configs(model_path)
        results['checks']['json_configs'] = config_ok
        if config_ok:
            print("✅ JSON configs: VALID")
        else:
            print("❌ JSON configs: INVALID")
            results['errors'].append("Invalid or corrupted JSON configuration files")
            results['valid'] = False
            
        # 4. Safetensors validation (if available)
        safetensors_ok = self._validate_safetensors(model_path)
        results['checks']['safetensors'] = safetensors_ok
        if safetensors_ok is True:
            print("✅ Safetensors: VALID")
        elif safetensors_ok is False:
            print("❌ Safetensors: INVALID")
            results['errors'].append("Corrupted or invalid safetensors files")
            results['valid'] = False
        else:
            print("⚠️ Safetensors: SKIPPED (not available or not applicable)")
            
        # 5. Git LFS pointer check
        lfs_ok = self._check_git_lfs_pointers(model_path)
        results['checks']['git_lfs'] = lfs_ok
        if lfs_ok is True:
            print("✅ Git LFS: No incomplete downloads detected")
        elif lfs_ok is False:
            print("❌ Git LFS: Found incomplete LFS downloads")
            results['errors'].append("Found Git LFS pointer files instead of actual model files")
            results['valid'] = False
        else:
            print("ℹ️ Git LFS: Not applicable")
            
        return results
        
    def _check_basic_structure(self, model_path: Path) -> bool:
        """Check if model has basic required structure for Wan models"""
        try:
            # Check for essential Wan model files
            essential_files = [
                "config.json"  # All Wan models should have this
            ]
            
            # At least one of these model files should exist
            model_files = [
                "diffusion_pytorch_model.safetensors",
                "diffusion_pytorch_model-00001-of-00007.safetensors",  # Multi-part models
                "model_index.json"  # Legacy format
            ]
            
            # Check for essential files
            for essential_file in essential_files:
                if not (model_path / essential_file).exists():
                    print(f"   ❌ Missing essential file: {essential_file}")
                    return False
            
            # Check for at least one model file
            has_model_file = any((model_path / mf).exists() for mf in model_files)
            if not has_model_file:
                print(f"   ❌ No model files found. Expected one of: {model_files}")
                return False
            
            # Additional Wan-specific checks
            wan_specific_files = [
                "Wan2.1_VAE.pth",
                "models_t5_umt5-xxl-enc-bf16.pth"
            ]
            
            has_wan_files = any((model_path / wf).exists() for wf in wan_specific_files)
            if not has_wan_files:
                print(f"   ⚠️ No Wan-specific files found, might be a non-Wan model")
                # Don't fail validation for this, just warn
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error checking basic structure: {e}")
            return False
        
    def _validate_file_sizes(self, model_path: Path) -> bool:
        """Check if file sizes are reasonable"""
        suspicious_files = []
        
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                size = file_path.stat().st_size
                
                # Check for suspiciously small files that should be large
                if file_path.suffix in ['.safetensors', '.pth', '.ckpt']:
                    if size < 1024:  # Less than 1KB is suspicious for model files
                        suspicious_files.append(f"{file_path.name} ({size} bytes)")
                        
                # Check for empty files
                if size == 0:
                    suspicious_files.append(f"{file_path.name} (empty)")
                    
        if suspicious_files:
            print(f"⚠️ Found {len(suspicious_files)} suspicious files:")
            for sf in suspicious_files[:5]:  # Show first 5
                print(f"   • {sf}")
            if len(suspicious_files) > 5:
                print(f"   ... and {len(suspicious_files) - 5} more")
            return False
            
        return True
        
    def _validate_json_configs(self, model_path: Path) -> bool:
        """Validate JSON configuration files"""
        json_files = [
            "config.json",
            "model_index.json"
        ]
        
        all_valid = True
        
        for json_file in json_files:
            file_path = model_path / json_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                    print(f"   ✅ {json_file}: Valid JSON")
                except json.JSONDecodeError as e:
                    print(f"   ❌ {json_file}: Invalid JSON - {e}")
                    all_valid = False
                except Exception as e:
                    print(f"   ❌ {json_file}: Error reading - {e}")
                    all_valid = False
                    
        return all_valid
        
    def _validate_safetensors(self, model_path: Path) -> Optional[bool]:
        """Validate safetensors files"""
        try:
            from safetensors import safe_open
        except ImportError:
            return None  # Skip if safetensors not available
            
        safetensors_files = list(model_path.glob("*.safetensors"))
        safetensors_files.extend(model_path.glob("**/*.safetensors"))
        
        if not safetensors_files:
            return None  # No safetensors files to check
            
        all_valid = True
        
        for st_file in safetensors_files:
            try:
                with safe_open(st_file, framework="pt", device="cpu") as f:
                    # Try to access metadata and verify it's readable
                    keys = list(f.keys())
                    if len(keys) == 0:
                        print(f"   ⚠️ {st_file.name}: No tensors found")
                        all_valid = False
                    else:
                        print(f"   ✅ {st_file.name}: {len(keys)} tensors")
                        
            except Exception as e:
                print(f"   ❌ {st_file.name}: Error - {e}")
                all_valid = False
                
        return all_valid
        
    def _check_git_lfs_pointers(self, model_path: Path) -> Optional[bool]:
        """Check for Git LFS pointer files that weren't properly downloaded"""
        lfs_pointers = []
        
        for file_path in model_path.rglob("*"):
            if file_path.is_file() and file_path.stat().st_size < 200:  # LFS pointers are small
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.startswith('version https://git-lfs.github.com'):
                            lfs_pointers.append(file_path.name)
                except:
                    pass  # Ignore binary files or encoding errors
                    
        if lfs_pointers:
            print(f"   ❌ Found {len(lfs_pointers)} Git LFS pointer files:")
            for lfs_file in lfs_pointers[:3]:
                print(f"      • {lfs_file}")
            if len(lfs_pointers) > 3:
                print(f"      ... and {len(lfs_pointers) - 3} more")
            return False
            
        return True if any(f.suffix in ['.safetensors', '.pth'] for f in model_path.rglob("*")) else None
        
    def compute_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> Optional[str]:
        """Compute file hash"""
        try:
            hash_obj = hashlib.new(algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            print(f"❌ Error computing hash for {file_path}: {e}")
            return None
            
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory"""
        total = 0
        try:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    total += file_path.stat().st_size
        except Exception:
            pass
        return total
        
    def _format_size(self, size_bytes: int) -> str:
        """Format size in human readable format"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}PB"
        
    def cleanup_invalid_models(self, models: List[Dict], auto_confirm: bool = False) -> List[str]:
        """Provide instructions for cleaning up invalid models instead of automatically deleting them"""
        invalid_models = []
        
        for model in models:
            validation_result = self.validate_model_integrity(Path(model['path']))
            if not validation_result['valid']:
                invalid_models.append({
                    'name': model['name'],
                    'path': model['path'],
                    'size': model['size_formatted'],
                    'errors': validation_result['errors']
                })
        
        if not invalid_models:
            print("✅ All models are valid - no cleanup needed")
            return []
        
        print(f"\n⚠️ Found {len(invalid_models)} invalid model(s):")
        print("-" * 50)
        
        for i, invalid_model in enumerate(invalid_models, 1):
            print(f"{i}. {invalid_model['name']} ({invalid_model['size']})")
            print(f"   Path: {invalid_model['path']}")
            print(f"   Issues: {', '.join(invalid_model['errors'])}")
            print()
        
        print("🛠️ MANUAL CLEANUP INSTRUCTIONS:")
        print("=" * 50)
        print("For safety, models are NOT automatically deleted.")
        print("If you want to remove invalid models, please:")
        print()
        print("1. 📋 Copy the paths above")
        print("2. 🔍 Verify the issues are real (not temporary)")
        print("3. 💾 Backup any important data if needed")
        print("4. 🗑️ Manually delete the directories:")
        print()
        
        for invalid_model in invalid_models:
            print(f"   rm -rf \"{invalid_model['path']}\"")
        
        print()
        print("5. 📥 Re-download models using:")
        print()
        
        for invalid_model in invalid_models:
            model_name = invalid_model['name'].lower()
            if 'vace' in model_name and '1.3b' in model_name:
                print(f"   huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir models/wan/Wan2.1-VACE-1.3B")
            elif 'vace' in model_name and '14b' in model_name:
                print(f"   huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir models/wan/Wan2.1-VACE-14B")
            elif 't2v' in model_name and '1.3b' in model_name:
                print(f"   huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/wan/Wan2.1-T2V-1.3B")
            elif 't2v' in model_name and '14b' in model_name:
                print(f"   huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir models/wan/Wan2.1-T2V-14B")
            elif 'i2v' in model_name and '1.3b' in model_name:
                print(f"   huggingface-cli download Wan-AI/Wan2.1-I2V-1.3B --local-dir models/wan/Wan2.1-I2V-1.3B")
            elif 'i2v' in model_name and '14b' in model_name:
                print(f"   huggingface-cli download Wan-AI/Wan2.1-I2V-14B --local-dir models/wan/Wan2.1-I2V-14B")
        
        print()
        print("💡 TIP: Enable 'Auto-Download Models' in the Wan tab for automatic re-downloading")
        print("⚠️ SAFETY: Always verify issues before deleting - some errors may be temporary")
        
        return [model['name'] for model in invalid_models]

    def validate_against_huggingface_checksums(self, model_path: Path, repo_id: str = None) -> Dict[str, Any]:
        """Validate model files against official HuggingFace checksums"""
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'checked_files': {},
            'skipped_files': []
        }
        
        # Try to determine repo_id from model name if not provided
        if not repo_id:
            model_name = model_path.name
            if 'vace' in model_name.lower() and '1.3b' in model_name.lower():
                repo_id = "Wan-AI/Wan2.1-VACE-1.3B"
            elif 'vace' in model_name.lower() and '14b' in model_name.lower():
                repo_id = "Wan-AI/Wan2.1-VACE-14B"
            elif 't2v' in model_name.lower() and '1.3b' in model_name.lower():
                repo_id = "Wan-AI/Wan2.1-T2V-1.3B"
            elif 't2v' in model_name.lower() and '14b' in model_name.lower():
                repo_id = "Wan-AI/Wan2.1-T2V-14B"
            elif 'i2v' in model_name.lower() and '1.3b' in model_name.lower():
                repo_id = "Wan-AI/Wan2.1-I2V-1.3B"
            elif 'i2v' in model_name.lower() and '14b' in model_name.lower():
                repo_id = "Wan-AI/Wan2.1-I2V-14B"
            else:
                results['errors'].append("Could not determine HuggingFace repo ID for checksum validation")
                results['valid'] = False
                return results
        
        print(f"🔐 Validating against official checksums for {repo_id}")
        
        try:
            from huggingface_hub import HfApi
            
            api = HfApi()
            
            # Get repository file list
            try:
                repo_files = api.list_repo_files(repo_id=repo_id)
            except Exception as e:
                results['errors'].append(f"Failed to get repo files: {e}")
                results['valid'] = False
                return results
            
            # Key files to check with their expected checksums
            # These are known checksums from HuggingFace official pages
            known_checksums = {
                "Wan-AI/Wan2.1-VACE-1.3B": {
                    "Wan2.1_VAE.pth": "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981",
                    "models_t5_umt5-xxl-enc-bf16.pth": None,  # Unknown checksum
                    "diffusion_pytorch_model.safetensors": None,  # Unknown checksum  
                    "config.json": None  # Small file, checksum varies
                }
            }
            
            # Get checksums for the specific repo
            repo_checksums = known_checksums.get(repo_id, {})
            
            # Check files that exist locally and have known checksums
            files_to_check = []
            for file_name, expected_sha256 in repo_checksums.items():
                if file_name in repo_files:
                    local_file = model_path / file_name
                    if local_file.exists() and expected_sha256:
                        files_to_check.append((file_name, expected_sha256, local_file))
                    elif local_file.exists() and not expected_sha256:
                        results['skipped_files'].append(f"{file_name} (no known checksum)")
            
            if not files_to_check:
                results['warnings'].append("No files with known checksums found to validate")
                print(f"   ⚠️ No files with known checksums found to validate")
                return results
            
            print(f"   Checking {len(files_to_check)} files with known checksums...")
            
            # Check each file
            valid_files = 0
            for file_name, expected_sha256, local_file in files_to_check:
                try:
                    print(f"   🔄 {file_name}: Computing checksum...")
                    local_sha256 = self.compute_file_hash(local_file, 'sha256')
                    
                    if not local_sha256:
                        print(f"   ❌ {file_name}: Failed to compute local checksum")
                        results['errors'].append(f"Failed to compute checksum for {file_name}")
                        results['valid'] = False
                        continue
                    
                    # Compare checksums
                    if local_sha256.lower() == expected_sha256.lower():
                        print(f"   ✅ {file_name}: Official checksum verified")
                        valid_files += 1
                        results['checked_files'][file_name] = {
                            'status': 'valid',
                            'expected': expected_sha256,
                            'actual': local_sha256
                        }
                    else:
                        print(f"   ❌ {file_name}: Checksum mismatch!")
                        print(f"      Expected: {expected_sha256}")
                        print(f"      Actual:   {local_sha256}")
                        results['errors'].append(f"Checksum mismatch for {file_name}")
                        results['valid'] = False
                        results['checked_files'][file_name] = {
                            'status': 'invalid',
                            'expected': expected_sha256,
                            'actual': local_sha256
                        }
                        
                except Exception as e:
                    print(f"   ⚠️ {file_name}: Validation failed - {e}")
                    results['warnings'].append(f"Could not validate {file_name}: {e}")
                    results['skipped_files'].append(file_name)
            
            if valid_files > 0:
                print(f"✅ Checksum validation completed: {valid_files}/{len(files_to_check)} files verified")
            else:
                print(f"❌ No files could be verified against known checksums")
                if not results['errors']:  # Only set invalid if no other errors
                    results['valid'] = False
                    results['errors'].append("No files could be verified against known checksums")
            
        except ImportError:
            results['warnings'].append("huggingface_hub not available for checksum validation")
            print("⚠️ huggingface_hub not available - install with: pip install huggingface_hub")
        except Exception as e:
            results['errors'].append(f"HuggingFace checksum validation failed: {e}")
            results['valid'] = False
            print(f"❌ HuggingFace checksum validation error: {e}")
        
        return results


def main():
    """Main function for command line usage"""
    validator = WanModelValidator()
    
    # Discover all models
    models = validator.discover_models()
    
    if not models:
        print("\n❌ No Wan models found!")
        print("\n💡 SUGGESTIONS:")
        print("1. Download models using: huggingface-cli download Wan-AI/Wan2.1-VACE-1.3B --local-dir models/wan/Wan2.1-VACE-1.3B")
        print("2. Check if models are in the correct directories")
        print("3. Ensure models were downloaded completely")
        return
        
    print(f"\n📊 Summary: Found {len(models)} model(s)")
    print("=" * 60)
    
    for model in models:
        print(f"• {model['name']} ({model['type']}, {model['size_formatted']})")
        
    # Offer validation and cleanup
    print("\n🔧 OPTIONS:")
    print("1. Validate all models")
    print("2. Show cleanup instructions for invalid models") 
    print("3. Exit")
    
    try:
        choice = input("\nEnter choice [1-3]: ").strip()
        
        if choice == '1':
            print("\n🔍 Running comprehensive validation...")
            for model in models:
                validator.validate_model_integrity(Path(model['path']))
                
        elif choice == '2':
            invalid_models = validator.cleanup_invalid_models(models)
            if invalid_models:
                print(f"\n📋 Found {len(invalid_models)} invalid models that need attention:")
                for name in invalid_models:
                    print(f"   • {name}")
                print("\n💡 Follow the instructions above to safely clean up invalid models")
            else:
                print("\n✅ All models are valid - no cleanup needed")
                
        elif choice == '3':
            print("👋 Goodbye!")
            return
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        

if __name__ == "__main__":
    main() 