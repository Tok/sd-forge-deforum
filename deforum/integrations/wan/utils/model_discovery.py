#!/usr/bin/env python3
"""
WAN Model Discovery System
Auto-discovers WAN models from common locations without requiring manual paths
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
from deforum.utils.system.logging import get_logger, emoji_if_enabled

# Initialize logger
logger = get_logger()


class WanModelDiscovery:
    """Smart WAN model discovery that finds models automatically"""
    
    def __init__(self):
        self.common_model_locations = self._get_common_model_locations()
        self.discovered_models = []
        
    def _get_common_model_locations(self) -> List[Path]:
        """Get common locations where WAN models might be stored"""
        locations = []
        
        # Current extension directory
        current_dir = Path(__file__).parent.parent.parent
        
        # Common model locations
        potential_locations = [
            # Webui model directories
            current_dir.parent.parent / "models" / "wan",
            current_dir.parent.parent / "models" / "WAN",
            current_dir.parent.parent / "models" / "Wan",
            current_dir.parent.parent / "models" / "wan_models",
            
            # Extension model directories  
            current_dir / "models",
            current_dir / "wan_models",
            
            # HuggingFace cache (common location)
            Path.home() / ".cache" / "huggingface" / "hub",
            
            # Common download locations
            Path.home() / "Downloads",
            Path("C:/") / "AI_Models" / "WAN" if os.name == 'nt' else Path.home() / "AI_Models" / "WAN",
            
            # Official WAN repo model locations
            current_dir / "Wan2.1" / "models",
            
            # User's documents (common on Windows)
            Path.home() / "Documents" / "AI_Models" / "WAN" if os.name == 'nt' else None,
        ]
        
        # Filter out None values and add existing paths
        for loc in potential_locations:
            if loc and loc.exists():
                locations.append(loc)
                
        return locations
    
    def discover_models(self) -> List[Dict]:
        """Discover all available WAN models automatically"""
        logger.info(f"{emoji_if_enabled("🔍")} Auto-discovering WAN models...")
        
        discovered = []
        
        for location in self.common_model_locations:
            logger.info(f"   📂 Searching: {location}")
            models = self._scan_directory_for_models(location)
            discovered.extend(models)
            
        # Also scan HuggingFace cache for downloaded models
        hf_models = self._scan_huggingface_cache()
        discovered.extend(hf_models)
        
        # Remove duplicates and sort by preference
        unique_models = self._deduplicate_models(discovered)
        self.discovered_models = self._sort_models_by_preference(unique_models)
        
        if self.discovered_models:
            logger.info(f"{emoji_if_enabled("✅")} Found {len(self.discovered_models)} WAN model(s):")
            for i, model in enumerate(self.discovered_models):
                logger.info(f"   {i+1}. {model['name']} ({model['type']}, {model['size']}) - {model['path']}")
        else:
            logger.info("No WAN models found in common locations", emoji='off')
            
        return self.discovered_models
    
    def _scan_directory_for_models(self, directory: Path) -> List[Dict]:
        """Scan a directory for WAN model files"""
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
        """Scan HuggingFace cache for WAN models"""
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
        """Check if a directory contains WAN model files (2.1 or 2.2 format)"""

        # Check for Wan 2.2 Diffusers format (standard diffusers structure)
        is_wan22_diffusers = (
            (directory / 'vae').exists() and
            (directory / 'text_encoder').exists() and
            (directory / 'model_index.json').exists()
        )
        if is_wan22_diffusers:
            return True

        # Check for Wan 2.1 format (flat directory with .pth files)
        required_files_21 = [
            'diffusion_pytorch_model.safetensors',
            'Wan2.1_VAE.pth',
            'models_t5_umt5-xxl-enc-bf16.pth'
        ]

        for required_file in required_files_21:
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
            logger.warning(f"   ⚠️ Error analyzing {directory}: {e}")
            return None
    
    def _detect_model_type(self, directory: Path) -> str:
        """Detect if model is TI2V, T2V, I2V, S2V, etc."""
        dir_name = directory.name.lower()

        # Check directory name for type indicators (Wan 2.2 first, then 2.1)
        if 'ti2v' in dir_name:
            return 'TI2V'  # Wan 2.2 unified text/image-to-video
        elif 's2v' in dir_name:
            return 'S2V'   # Wan 2.2 speech-to-video
        elif 'animate' in dir_name:
            return 'Animate'  # Wan 2.2 character animation
        elif 'i2v' in dir_name:
            return 'I2V'   # Image-to-video
        elif 't2v' in dir_name:
            return 'T2V'   # Text-to-video
        elif 'flf2v' in dir_name:
            return 'FLF2V' # First-last-frame-to-video

        # Check model_index.json for Wan 2.2 Diffusers models
        model_index_path = directory / 'model_index.json'
        if model_index_path.exists():
            try:
                with open(model_index_path, 'r') as f:
                    config = json.load(f)

                # Check _class_name field
                class_name = config.get('_class_name', '').lower()
                if 'ti2v' in class_name or 'imagetovideo' in class_name:
                    return 'TI2V'
                elif 'i2v' in class_name:
                    return 'I2V'

            except (json.JSONDecodeError, IOError):
                pass

        # Check config.json for more details (Wan 2.1 format)
        config_path = directory / 'config.json'
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)

                # Look for type indicators in config
                config_str = json.dumps(config).lower()
                if 'i2v' in config_str:
                    return 'I2V'

            except (json.JSONDecodeError, IOError):
                pass

        # Default assumption based on file structure
        return 'T2V'
    
    def _detect_model_size(self, directory: Path) -> str:
        """Detect model size (5B, 1.3B, 14B, A14B, etc.)"""
        dir_name = directory.name.lower()

        # Check directory name for size indicators (Wan 2.2 first)
        if '5b' in dir_name or '5_b' in dir_name:
            return '5B'
        elif 'a14b' in dir_name or 'a_14b' in dir_name:
            return 'A14B'  # MoE architecture: 27B total, 14B active
        elif '1.3b' in dir_name or '1_3b' in dir_name:
            return '1.3B'
        elif '14b' in dir_name:
            return '14B'

        # Check model_index.json for Wan 2.2 Diffusers models
        model_index_path = directory / 'model_index.json'
        if model_index_path.exists():
            try:
                with open(model_index_path, 'r') as f:
                    config = json.load(f)

                # Extract from _name_or_path or similar fields
                name_or_path = config.get('_name_or_path', '').lower()
                if '5b' in name_or_path:
                    return '5B'
                elif 'a14b' in name_or_path or '14b' in name_or_path:
                    return 'A14B'

            except (json.JSONDecodeError, IOError):
                pass

        # Check for multi-part model files (indicates 14B)
        multi_part_exists = any(
            (directory / f"diffusion_pytorch_model-{i:05d}-of-00007.safetensors").exists()
            for i in range(1, 8)
        )
        if multi_part_exists:
            return '14B'

        # Check transformer model file sizes for Wan 2.2
        transformer_model = directory / 'transformer' / 'diffusion_pytorch_model.safetensors'
        if transformer_model.exists():
            try:
                size_gb = transformer_model.stat().st_size / (1024**3)
                if size_gb < 15:
                    return '5B'
                else:
                    return 'A14B'
            except OSError:
                pass

        # Check legacy file sizes for Wan 2.1
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

        return '5B'  # Default to 5B for Wan 2.2 models
    
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
        """Sort models by preference (TI2V > T2V > I2V, prefer smaller/faster)"""
        def preference_score(model):
            score = 0

            # Model type priority (Wan 2.2 unified models preferred)
            type_priority = {
                'TI2V': 100,     # Wan 2.2 unified text/image-to-video (best)
                'T2V': 80,       # Standard text-to-video
                'I2V': 60,       # Image-to-video only
                'S2V': 20,       # Speech-to-video (specialized)
                'Animate': 10,   # Character animation (specialized)
            }
            score += type_priority.get(model['type'], 0)

            # Size priority (prefer smaller/faster for consumer GPUs)
            size_priority = {
                '5B': 50,        # Wan 2.2 TI2V-5B (fastest, RTX 4090 compatible)
                '1.3B': 40,      # Wan 2.1 small models
                '14B': 20,       # Wan 2.1 large models
                'A14B': 10,      # Wan 2.2 MoE (27B total, requires more VRAM)
            }
            score += size_priority.get(model['size'], 0)

            # Prefer local over cache
            if 'Local' in model['source']:
                score += 5

            return score

        return sorted(models, key=preference_score, reverse=True)
    
    def get_best_model(self) -> Optional[Dict]:
        """Get the best available model"""
        if not self.discovered_models:
            self.discover_models()
            
        return self.discovered_models[0] if self.discovered_models else None
    
    def get_model_by_preference(self, prefer_size: str = "5B") -> Optional[Dict]:
        """Get model based on size preference"""
        if not self.discovered_models:
            self.discover_models()

        for model in self.discovered_models:
            if prefer_size and prefer_size in model['size']:
                return model

        # Fallback to any available model
        return self.discovered_models[0] if self.discovered_models else None

def discover_wan_models() -> List[Dict]:
    """Convenience function to discover WAN models"""
    discovery = WanModelDiscovery()
    return discovery.discover_models()

def get_best_wan_model() -> Optional[Dict]:
    """Convenience function to get the best available WAN model"""
    discovery = WanModelDiscovery()
    return discovery.get_best_model()

if __name__ == "__main__":
    # Test the discovery system
    logger.info("🧪 Testing WAN Model Discovery System...")
    
    discovery = WanModelDiscovery()
    models = discovery.discover_models()
    
    if models:
        logger.info(f"\n🎉 Discovery successful! Found {len(models)} model(s)")
        best = discovery.get_best_model()
        logger.info(f"🏆 Best model: {best['name']} ({best['type']}, {best['size']})")
        logger.info(f"📁 Path: {best['path']}")
    else:
        logger.info("\n❌ No models found - you may need to download WAN models first", emoji='off')
        logger.info("Suggested locations to place models:", emoji='bulb')
        for loc in discovery.common_model_locations:
            print(f"   📂 {loc}") 