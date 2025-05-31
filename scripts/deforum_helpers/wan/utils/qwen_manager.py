"""
Qwen Model Manager for QwenPromptExpander Integration
Handles auto-downloading, model selection, and management of Qwen models
"""

import os
import sys
import torch
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from .prompt_extend import QwenPromptExpander


class QwenModelManager:
    """Manages Qwen models for prompt enhancement"""
    
    # Model specifications with VRAM requirements and capabilities
    MODEL_SPECS = {
        "QwenVL2.5_7B": {
            "hf_name": "Qwen/Qwen2.5-VL-7B-Instruct",
            "vram_gb": 16,
            "is_vl": True,
            "description": "7B Vision-Language model, supports image+text input"
        },
        "QwenVL2.5_3B": {
            "hf_name": "Qwen/Qwen2.5-VL-3B-Instruct", 
            "vram_gb": 8,
            "is_vl": True,
            "description": "3B Vision-Language model, supports image+text input"
        },
        "Qwen2.5_14B": {
            "hf_name": "Qwen/Qwen2.5-14B-Instruct",
            "vram_gb": 28,
            "is_vl": False,
            "description": "14B Text-only model, high quality prompt enhancement"
        },
        "Qwen2.5_7B": {
            "hf_name": "Qwen/Qwen2.5-7B-Instruct",
            "vram_gb": 14,
            "is_vl": False,
            "description": "7B Text-only model, good balance of quality and speed"
        },
        "Qwen2.5_3B": {
            "hf_name": "Qwen/Qwen2.5-3B-Instruct",
            "vram_gb": 6,
            "is_vl": False,
            "description": "3B Text-only model, fast and memory-efficient"
        }
    }
    
    def __init__(self, models_dir: str = "models/qwen"):
        """
        Initialize Qwen model manager
        
        Args:
            models_dir: Directory to store Qwen models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._cached_expandable = None
        
    def get_available_vram(self) -> float:
        """Get available VRAM in GB"""
        try:
            if torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024**3)
            else:
                # Use system RAM as fallback
                return psutil.virtual_memory().total / (1024**3)
        except:
            return 8.0  # Conservative fallback
            
    def auto_select_model(self, prefer_vl: bool = False) -> str:
        """
        Automatically select the best Qwen model based on available VRAM
        
        Args:
            prefer_vl: Whether to prefer vision-language models
            
        Returns:
            str: Selected model name
        """
        available_vram = self.get_available_vram()
        
        print(f"🧠 Available VRAM: {available_vram:.1f}GB")
        
        # Sort models by VRAM requirement (ascending)
        sorted_models = sorted(
            self.MODEL_SPECS.items(),
            key=lambda x: x[1]['vram_gb']
        )
        
        # Find the best model that fits in VRAM
        best_model = None
        for model_name, specs in sorted_models:
            if specs['vram_gb'] <= available_vram:
                if prefer_vl and specs['is_vl']:
                    best_model = model_name
                elif not prefer_vl and not specs['is_vl']:
                    best_model = model_name
                elif best_model is None:  # Fallback to any compatible model
                    best_model = model_name
                    
        # If no model fits, use the smallest one
        if best_model is None:
            best_model = sorted_models[0][0]
            print(f"⚠️ No model fits in {available_vram:.1f}GB VRAM, using smallest: {best_model}")
        else:
            specs = self.MODEL_SPECS[best_model]
            print(f"✅ Auto-selected: {best_model} ({specs['description']})")
            
        return best_model
        
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a Qwen model is already downloaded"""
        if model_name == "Auto-Select":
            return True  # Auto-select doesn't need to be downloaded
            
        model_spec = self.MODEL_SPECS.get(model_name)
        if not model_spec:
            return False
            
        # Check in local models directory
        local_path = self.models_dir / model_name
        if local_path.exists() and (local_path / "config.json").exists():
            return True
            
        # Check if available via transformers (cached)
        try:
            from transformers import AutoConfig
            AutoConfig.from_pretrained(model_spec['hf_name'])
            return True
        except:
            return False
            
    def download_model(self, model_name: str) -> bool:
        """
        Download a Qwen model if not already available
        
        Args:
            model_name: Name of the model to download
            
        Returns:
            bool: Success status
        """
        if model_name == "Auto-Select":
            # Auto-select and download the best model
            model_name = self.auto_select_model()
            
        if self.is_model_downloaded(model_name):
            print(f"✅ Model {model_name} already available")
            return True
            
        model_spec = self.MODEL_SPECS.get(model_name)
        if not model_spec:
            print(f"❌ Unknown model: {model_name}")
            return False
            
        print(f"📥 Downloading Qwen model: {model_name}")
        print(f"   HuggingFace: {model_spec['hf_name']}")
        print(f"   VRAM requirement: {model_spec['vram_gb']}GB")
        print(f"   Description: {model_spec['description']}")
        
        try:
            # Try downloading using huggingface_hub
            from huggingface_hub import snapshot_download
            
            local_path = self.models_dir / model_name
            
            snapshot_download(
                repo_id=model_spec['hf_name'],
                local_dir=str(local_path),
                local_dir_use_symlinks=False
            )
            
            print(f"✅ Successfully downloaded {model_name} to {local_path}")
            return True
            
        except ImportError:
            print("⚠️ huggingface_hub not available, trying alternative download...")
            return self._download_with_transformers(model_name, model_spec)
            
        except Exception as e:
            print(f"❌ Failed to download {model_name}: {e}")
            print("💡 Trying to use cached/online version...")
            return self._download_with_transformers(model_name, model_spec)
            
    def _download_with_transformers(self, model_name: str, model_spec: Dict) -> bool:
        """Fallback download using transformers library"""
        try:
            from transformers import AutoConfig, AutoTokenizer
            
            print(f"📥 Downloading {model_name} using transformers...")
            
            # This will cache the model
            AutoConfig.from_pretrained(model_spec['hf_name'])
            AutoTokenizer.from_pretrained(model_spec['hf_name'])
            
            print(f"✅ {model_name} cached successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download {model_name} with transformers: {e}")
            return False
            
    def create_prompt_expander(self, model_name: str, auto_download: bool = True) -> Optional[QwenPromptExpander]:
        """
        Create a QwenPromptExpander instance with the specified model
        
        Args:
            model_name: Name of the Qwen model to use
            auto_download: Whether to auto-download missing models
            
        Returns:
            QwenPromptExpander instance or None if failed
        """
        if model_name == "Auto-Select":
            model_name = self.auto_select_model()
            
        model_spec = self.MODEL_SPECS.get(model_name)
        if not model_spec:
            print(f"❌ Unknown model: {model_name}")
            return None
            
        # Download model if needed
        if auto_download and not self.is_model_downloaded(model_name):
            if not self.download_model(model_name):
                print(f"❌ Failed to download {model_name}")
                return None
                
        try:
            # Determine model path
            local_path = self.models_dir / model_name
            if local_path.exists():
                model_path = str(local_path)
            else:
                model_path = model_spec['hf_name']  # Use HF name for cached models
                
            # Create QwenPromptExpander
            expander = QwenPromptExpander(
                model_name=model_path,
                is_vl=model_spec['is_vl'],
                device=0 if torch.cuda.is_available() else "cpu"
            )
            
            print(f"✅ Created QwenPromptExpander with {model_name}")
            return expander
            
        except Exception as e:
            print(f"❌ Failed to create QwenPromptExpander: {e}")
            return None
            
    def enhance_prompts(self, 
                       prompts: Dict[str, str], 
                       model_name: str = "Auto-Select",
                       language: str = "English",
                       auto_download: bool = True) -> Dict[str, str]:
        """
        Enhance a dictionary of prompts using QwenPromptExpander
        
        Args:
            prompts: Dictionary of frame_number -> prompt
            model_name: Qwen model to use
            language: Target language for enhanced prompts
            auto_download: Whether to auto-download missing models
            
        Returns:
            Dictionary of frame_number -> enhanced_prompt
        """
        if not prompts:
            return {}
            
        # Create or reuse prompt expander
        if self._cached_expandable is None or self._cached_expandable[0] != model_name:
            expander = self.create_prompt_expander(model_name, auto_download)
            if expander is None:
                print("❌ Failed to create prompt expander, returning original prompts")
                return prompts
            self._cached_expandable = (model_name, expander)
        else:
            expander = self._cached_expandable[1]
            
        enhanced_prompts = {}
        
        print(f"🎨 Enhancing {len(prompts)} prompts with {model_name}...")
        
        for frame_num, original_prompt in prompts.items():
            try:
                print(f"   📝 Enhancing frame {frame_num}: {original_prompt[:50]}...")
                
                result = expander(
                    prompt=original_prompt,
                    tar_lang="en" if language == "English" else "zh"
                )
                
                if result.status:
                    enhanced_prompts[frame_num] = result.prompt
                    print(f"   ✅ Enhanced: {result.prompt[:50]}...")
                else:
                    enhanced_prompts[frame_num] = original_prompt
                    print(f"   ⚠️ Enhancement failed, using original prompt")
                    
            except Exception as e:
                print(f"   ❌ Error enhancing frame {frame_num}: {e}")
                enhanced_prompts[frame_num] = original_prompt
                
        print(f"✅ Enhanced {len(enhanced_prompts)} prompts successfully")
        return enhanced_prompts
        
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a Qwen model"""
        if model_name == "Auto-Select":
            selected = self.auto_select_model()
            info = self.MODEL_SPECS.get(selected, {}).copy()
            info['selected_model'] = selected
            return info
        else:
            return self.MODEL_SPECS.get(model_name, {})
            
    def cleanup_cache(self):
        """Clean up cached prompt expander"""
        if self._cached_expandable is not None:
            try:
                expander = self._cached_expandable[1]
                if hasattr(expander, 'model'):
                    # Move model to CPU to free VRAM
                    expander.model = expander.model.to("cpu")
                torch.cuda.empty_cache()
            except:
                pass
            self._cached_expandable = None
            print("🧹 Cleaned up Qwen model cache")


# Global instance for easy access
qwen_manager = QwenModelManager() 