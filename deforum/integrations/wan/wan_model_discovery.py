#!/usr/bin/env python3
"""
Wan Model Discovery System
Auto-discovers Wan models from common locations without requiring manual paths
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys

class WanModelDiscovery:
    """Smart Wan model discovery that finds models automatically"""
    
    def __init__(self):
        self.common_model_locations = self._get_common_model_locations()
        self.discovered_models = []
        
    def _get_common_model_locations(self) -> List[Path]:
        """Get common locations where Wan models might be stored"""
        locations = []
        
        # Current extension directory
        current_dir = Path(__file__).parent.parent.parent
        
        # Common model locations
        potential_locations = [
            # Webui model directories
            current_dir.parent.parent / "models" / "wan",
            current_dir.parent.parent / "models" / "wan",
            current_dir.parent.parent / "models" / "Wan",
            current_dir.parent.parent / "models" / "wan_models",
            
            # Extension model directories  
            current_dir / "models",
            current_dir / "wan_models",
            
            # HuggingFace cache (common location)
            Path.home() / ".cache" / "huggingface" / "hub",
            
            # Common download locations
            Path.home() / "Downloads",
            Path("C:/") / "AI_Models" / "wan" if os.name == 'nt' else Path.home() / "AI_Models" / "wan",
            
            # Official Wan repo model locations
            current_dir / "Wan2.1" / "models",
            
            # User's documents (common on Windows)
            Path.home() / "Documents" / "AI_Models" / "wan" if os.name == 'nt' else None,
        ]
        
        # Filter out None values and add existing paths
        for loc in potential_locations:
            if loc and loc.exists():
                locations.append(loc)
                
        return locations
    
    def discover_models(self) -> List[Dict]:
        """Discover all available Wan models automatically"""
        print("🔍 Auto-discovering Wan models...")
        
        discovered = []
        
        for location in self.common_model_locations:
            print(f"   📂 Searching: {location}")
            models = self._scan_directory_for_models(location)
            discovered.extend(models)
            
        # Also scan HuggingFace cache for downloaded models
        hf_models = self._scan_huggingface_cache()
        discovered.extend(hf_models)
        
        # Remove duplicates and sort by preference
        unique_models = self._deduplicate_models(discovered)
        self.discovered_models = self._sort_models_by_preference(unique_models)
        
        if self.discovered_models:
            print(f"✅ Found {len(self.discovered_models)} Wan model(s):")
            for i, model in enumerate(self.discovered_models):
                print(f"   {i+1}. {model['name']} ({model['type']}, {model['size']}) - {model['path']}")
        else:
            print("❌ No Wan models found in common locations")
            
        return self.discovered_models
    
    def _scan_directory_for_models(self, directory: Path) -> List[Dict]:
        """Scan a directory for Wan model files"""
        models = []
        
        try:
            # Look for direct model files in this directory
            if self._is_wan_model_directory(directory):
                model_info = self._analyze_model_directory(directory)
                if model_info:
                    models.append(model_info)
            
            # Look for subdirectories that might contain models
            for subdir in directory.iterdir():
                if subdir.is_dir() and not subdir.name.startswith('.'):
                    if self._is_wan_model_directory(subdir):
                        model_info = self._analyze_model_directory(subdir)
                        if model_info:
                            models.append(model_info)
                            
        except (PermissionError, OSError):
            pass  # Skip directories we can't access
            
        return models
    
    def _scan_huggingface_cache(self) -> List[Dict]:
        """Scan HuggingFace cache for Wan models"""
        models = []
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        
        if not hf_cache.exists():
            return models
            
        try:
            for model_dir in hf_cache.iterdir():
                if model_dir.is_dir() and "wan" in model_dir.name.lower():
                    # Look for snapshots directory
                    snapshots_dir = model_dir / "snapshots"
                    if snapshots_dir.exists():
                        for snapshot in snapshots_dir.iterdir():
                            if snapshot.is_dir():
                                if self._is_wan_model_directory(snapshot):
                                    model_info = self._analyze_model_directory(snapshot)
                                    if model_info:
                                        model_info['source'] = 'HuggingFace Cache'
                                        models.append(model_info)
                                break  # Use first snapshot found
        except (PermissionError, OSError):
            pass
            
        return models
    
    def _is_wan_model_directory(self, directory: Path) -> bool:
        """Check if a directory contains Wan model files"""
        required_files = [
            'diffusion_pytorch_model.safetensors',
            'Wan2.1_VAE.pth',
            'models_t5_umt5-xxl-enc-bf16.pth'
        ]
        
        for required_file in required_files:
            if not (directory / required_file).exists():
                # Check for multi-part models (14B)
                if required_file == 'diffusion_pytorch_model.safetensors':
                    multi_part_exists = any(
                        (directory / f"diffusion_pytorch_model-{i:05d}-of-00007.safetensors").exists()
                        for i in range(1, 8)
                    )
                    if not multi_part_exists:
                        return False
                else:
                    return False
                    
        return True
    
    def _analyze_model_directory(self, directory: Path) -> Optional[Dict]:
        """Analyze a model directory to extract metadata"""
        try:
            # Determine model type and size
            model_type = self._detect_model_type(directory)
            model_size = self._detect_model_size(directory)
            
            # Generate friendly name
            name = self._generate_model_name(directory, model_type, model_size)
            
            return {
                'name': name,
                'path': str(directory),
                'type': model_type,
                'size': model_size,
                'source': 'Local Discovery',
                'config_path': str(directory / 'config.json') if (directory / 'config.json').exists() else None,
                'directory': directory
            }
            
        except Exception as e:
            print(f"   ⚠️ Error analyzing {directory}: {e}")
            return None
    
    def _detect_model_type(self, directory: Path) -> str:
        """Detect if model is VACE, T2V, I2V, etc."""
        dir_name = directory.name.lower()
        
        # Check directory name for type indicators
        if 'vace' in dir_name:
            return 'VACE (All-in-One)'
        elif 'i2v' in dir_name:
            return 'I2V (Image-to-Video)'
        elif 't2v' in dir_name:
            return 'T2V (Text-to-Video)'
        elif 'flf2v' in dir_name:
            return 'FLF2V (First-Last-Frame-to-Video)'
        
        # Check config.json for more details
        config_path = directory / 'config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                # Look for type indicators in config
                config_str = json.dumps(config).lower()
                if 'vace' in config_str:
                    return 'VACE (All-in-One)'
                elif 'i2v' in config_str:
                    return 'I2V (Image-to-Video)'
                    
            except (json.JSONDecodeError, IOError):
                pass
        
        # Default assumption based on file structure
        return 'T2V (Text-to-Video)'
    
    def _detect_model_size(self, directory: Path) -> str:
        """Detect model size (1.3B, 14B, etc.)"""
        dir_name = directory.name.lower()
        
        # Check directory name for size indicators
        if '1.3b' in dir_name or '1_3b' in dir_name:
            return '1.3B'
        elif '14b' in dir_name:
            return '14B'
        
        # Check for multi-part model files (indicates 14B)
        multi_part_exists = any(
            (directory / f"diffusion_pytorch_model-{i:05d}-of-00007.safetensors").exists()
            for i in range(1, 8)
        )
        if multi_part_exists:
            return '14B'
        
        # Check file sizes as a hint (rough estimate)
        main_model = directory / 'diffusion_pytorch_model.safetensors'
        if main_model.exists():
            try:
                size_gb = main_model.stat().st_size / (1024**3)
                if size_gb < 10:
                    return '1.3B'
                else:
                    return '14B'
            except OSError:
                pass
        
        return '1.3B'  # Default assumption
    
    def _generate_model_name(self, directory: Path, model_type: str, model_size: str) -> str:
        """Generate a friendly name for the model"""
        dir_name = directory.name
        
        # Extract meaningful parts from directory name
        if 'Wan' in dir_name:
            base_name = dir_name
        else:
            base_name = f"Wan-{model_type.split('(')[0]}-{model_size}"
            
        return base_name
    
    def _deduplicate_models(self, models: List[Dict]) -> List[Dict]:
        """Remove duplicate models (same model in multiple locations)"""
        seen = set()
        unique = []
        
        for model in models:
            # Create a signature based on type, size, and some path elements
            signature = f"{model['type']}_{model['size']}_{model['directory'].name}"
            
            if signature not in seen:
                seen.add(signature)
                unique.append(model)
                
        return unique
    
    def _sort_models_by_preference(self, models: List[Dict]) -> List[Dict]:
        """Sort models by preference (VACE > T2V, local > cache, etc.)"""
        def preference_score(model):
            score = 0
            
            # Prefer VACE models (most capable)
            if 'VACE' in model['type']:
                score += 100
            elif 'T2V' in model['type']:
                score += 50
            elif 'I2V' in model['type']:
                score += 30
                
            # Prefer local over cache
            if 'Local' in model['source']:
                score += 10
                
            # Prefer 1.3B for stability
            if '1.3B' in model['size']:
                score += 5
                
            return score
            
        return sorted(models, key=preference_score, reverse=True)
    
    def get_best_model(self) -> Optional[Dict]:
        """Get the best available model"""
        if not self.discovered_models:
            self.discover_models()
            
        return self.discovered_models[0] if self.discovered_models else None
    
    def get_model_by_preference(self, prefer_vace: bool = True, prefer_size: str = "1.3B") -> Optional[Dict]:
        """Get model based on specific preferences"""
        if not self.discovered_models:
            self.discover_models()
            
        for model in self.discovered_models:
            type_match = True
            if prefer_vace and 'VACE' not in model['type']:
                continue
                
            if prefer_size and prefer_size not in model['size']:
                continue
                
            return model
            
        # Fallback to any available model
        return self.discovered_models[0] if self.discovered_models else None

def discover_wan_models() -> List[Dict]:
    """Convenience function to discover Wan models"""
    discovery = WanModelDiscovery()
    return discovery.discover_models()

def get_best_wan_model() -> Optional[Dict]:
    """Convenience function to get the best available Wan model"""
    discovery = WanModelDiscovery()
    return discovery.get_best_model()

if __name__ == "__main__":
    # Test the discovery system
    print("🧪 Testing Wan Model Discovery System...")
    
    discovery = WanModelDiscovery()
    models = discovery.discover_models()
    
    if models:
        print(f"\n🎉 Discovery successful! Found {len(models)} model(s)")
        best = discovery.get_best_model()
        print(f"🏆 Best model: {best['name']} ({best['type']}, {best['size']})")
        print(f"📁 Path: {best['path']}")
    else:
        print("\n❌ No models found - you may need to download Wan models first")
        print("💡 Suggested locations to place models:")
        for loc in discovery.common_model_locations:
            print(f"   📂 {loc}")
