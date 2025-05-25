"""
WAN Integration Module for Deforum - CORRECTED REPOSITORY VERSION
Fixed to use actual Wan-AI repository with correct download URLs

This module provides:
1. Strict WAN model validation with correct filenames and URLs
2. Manual download instructions from working Wan-AI repository  
3. Real WAN model inference only - no placeholders or fallbacks
4. Fail fast approach - if WAN doesn't work, nothing is generated
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import json
from pathlib import Path
import cv2
import random
import subprocess
import sys
import shutil
from .generate import generate
from .animation_key_frames import DeformAnimKeys


class WanVideoGenerator:
    """
    WAN video generator - STRICT MODE, NO FALLBACKS
    Updated with correct Wan-AI repository URLs
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        self.loaded = False
        self.wan_t2v = None
        self.wan_i2v = None
        self.wan_config = None
        self.extension_root = Path(__file__).parent.parent.parent
        
    def validate_model_structure(self) -> Dict[str, Any]:
        """
        Validate WAN model structure - STRICT VALIDATION with corrected filenames
        """
        validation_result = {
            'has_required_files': False,
            'found_files': {},
            'missing_files': {},
            'can_use_wan_pipeline': False,
            'recommendations': []
        }
        
        if not self.model_path.exists():
            raise RuntimeError(f"❌ WAN model path does not exist: {self.model_path}")
        
        print(f"🔍 Validating WAN model directory: {self.model_path}")
        
        # Define EXACT required files from Wan-AI/Wan2.1-VACE-14B repository
        required_wan_files = {
            'dit_model': [
                'diffusion_pytorch_model-00001-of-00007.safetensors',  # Multi-part DiT model
                'diffusion_pytorch_model.safetensors.index.json',     # DiT index file
                'diffusion_pytorch_model.safetensors'                 # Alternative single file
            ],
            'vae': [
                'Wan2.1_VAE.pth'  # Exact filename from Wan-AI repo
            ],
            't5_encoder': [
                'models_t5_umt5-xxl-enc-bf16.pth'  # Exact filename from Wan-AI repo
            ],
            'config': [
                'config.json'  # Model configuration
            ]
        }
        
        print("📋 Checking for REQUIRED WAN model files from Wan-AI repository:")
        
        for component, possible_files in required_wan_files.items():
            found_files = []
            for filename in possible_files:
                file_path = self.model_path / filename
                if file_path.exists():
                    found_files.append(filename)
                    print(f"  ✅ {component}: Found {filename}")
            
            if found_files:
                validation_result['found_files'][component] = found_files
            else:
                validation_result['missing_files'][component] = possible_files
                print(f"  ❌ {component}: MISSING - Looking for {possible_files}")
        
        # Check for DiT model parts (multi-part model)
        dit_parts = []
        for i in range(1, 8):  # Parts 1-7 based on repo structure
            part_file = self.model_path / f"diffusion_pytorch_model-{i:05d}-of-00007.safetensors"
            if part_file.exists():
                dit_parts.append(part_file.name)
        
        if dit_parts:
            if 'dit_model' not in validation_result['found_files']:
                validation_result['found_files']['dit_model'] = []
            validation_result['found_files']['dit_model'].extend(dit_parts)
            print(f"  ✅ DiT model parts: Found {len(dit_parts)}/7 parts")
        
        # Check critical components (DiT, VAE, T5 are REQUIRED)
        has_dit = 'dit_model' in validation_result['found_files'] and len(validation_result['found_files']['dit_model']) > 0
        has_vae = 'vae' in validation_result['found_files']
        has_t5 = 't5_encoder' in validation_result['found_files']
        has_config = 'config' in validation_result['found_files']
        
        print(f"\n🔍 WAN Component Status:")
        print(f"  DiT Model: {'✅ FOUND' if has_dit else '❌ MISSING'}")
        print(f"  VAE: {'✅ FOUND' if has_vae else '❌ MISSING'}")
        print(f"  T5 Encoder: {'✅ FOUND' if has_t5 else '❌ MISSING'}")
        print(f"  Config: {'✅ FOUND' if has_config else '❌ MISSING'}")
        
        # STRICT REQUIREMENT - DiT, VAE, and T5 must be present
        if has_dit and has_vae and has_t5:
            validation_result['can_use_wan_pipeline'] = True
            validation_result['has_required_files'] = True
            print(f"\n✅ All required WAN components found - can proceed with generation")
            if not has_config:
                print(f"⚠️ Config missing but proceeding (will use default config)")
        else:
            # FAIL FAST - provide exact instructions with corrected URLs
            missing_components = []
            if not has_dit:
                missing_components.append("DiT Model")
            if not has_vae:
                missing_components.append("VAE")
            if not has_t5:
                missing_components.append("T5 Encoder")
            
            error_msg = self._generate_download_instructions(validation_result['missing_files'])
            raise RuntimeError(f"❌ CRITICAL WAN COMPONENTS MISSING: {', '.join(missing_components)}\n\n{error_msg}")
        
        return validation_result
    
    def _generate_download_instructions(self, missing_files: Dict[str, List[str]]) -> str:
        """Generate corrected manual download instructions with working URLs"""
        instructions = """
═══════════════════════════════════════════════════════════════════════════════
🚨 WAN MODEL COMPONENTS MISSING - MANUAL DOWNLOAD REQUIRED 🚨
═══════════════════════════════════════════════════════════════════════════════

To use WAN video generation, download these files from the corrected Wan-AI repository:
📁 Model Directory: {model_path}

CORRECTED REPOSITORY: Wan-AI/Wan2.1-VACE-14B
""".format(model_path=self.model_path)
        
        download_links = {
            'dit_model': {
                'files': [
                    'diffusion_pytorch_model-00001-of-00007.safetensors',
                    'diffusion_pytorch_model-00002-of-00007.safetensors', 
                    'diffusion_pytorch_model-00003-of-00007.safetensors',
                    'diffusion_pytorch_model-00004-of-00007.safetensors',
                    'diffusion_pytorch_model-00005-of-00007.safetensors',
                    'diffusion_pytorch_model-00006-of-00007.safetensors',
                    'diffusion_pytorch_model-00007-of-00007.safetensors',
                    'diffusion_pytorch_model.safetensors.index.json'
                ],
                'instructions': '''
🔗 DOWNLOAD DiT MODEL (Multi-part, ~63GB total):
Base URL: https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/

📥 DOWNLOAD COMMANDS:
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00001-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00002-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00003-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00004-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00005-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00006-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model-00007-of-00007.safetensors
wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/diffusion_pytorch_model.safetensors.index.json

📥 OR USE HUGGINGFACE CLI (recommended for large files):
pip install huggingface_hub
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir {model_path} --include="diffusion_pytorch_model*"'''.format(model_path="{model_path}")
            },
            'vae': {
                'files': ['Wan2.1_VAE.pth'],
                'instructions': '''
🔗 DOWNLOAD VAE (508 MB):
   https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/Wan2.1_VAE.pth
   
📥 DOWNLOAD COMMAND:
   wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/Wan2.1_VAE.pth
   
   OR save as: Wan2.1_VAE.pth'''
            },
            't5_encoder': {
                'files': ['models_t5_umt5-xxl-enc-bf16.pth'],
                'instructions': '''
🔗 DOWNLOAD T5 ENCODER (11.4 GB):
   https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
   
📥 DOWNLOAD COMMAND:
   wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
   
   OR save as: models_t5_umt5-xxl-enc-bf16.pth'''
            },
            'config': {
                'files': ['config.json'],
                'instructions': '''
🔗 DOWNLOAD CONFIG (Optional, 325 B):
   https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/config.json
   
📥 DOWNLOAD COMMAND:
   wget https://huggingface.co/Wan-AI/Wan2.1-VACE-14B/resolve/main/config.json'''
            }
        }
        
        for component, file_list in missing_files.items():
            if component in download_links:
                info = download_links[component]
                instructions += f"""
┌─────────────────────────────────────────────────────────────────────────────┐
│ MISSING: {component.upper().replace('_', ' ')}                                
└─────────────────────────────────────────────────────────────────────────────┘
Expected files: {', '.join(info['files'])}
{info['instructions'].format(model_path=self.model_path)}
"""
        
        instructions += """

═══════════════════════════════════════════════════════════════════════════════
🔧 EASIEST METHOD - Download entire repository:
═══════════════════════════════════════════════════════════════════════════════

pip install huggingface_hub
huggingface-cli download Wan-AI/Wan2.1-VACE-14B --local-dir {model_path}

This will download ALL required files automatically (~75GB total).

═══════════════════════════════════════════════════════════════════════════════
🔧 AFTER DOWNLOADING:
═══════════════════════════════════════════════════════════════════════════════

1. Place ALL downloaded files directly in: {model_path}
2. Verify file names match EXACTLY (case-sensitive)
3. Restart the WAN generation process

⚠️  IMPORTANT: 
   • Files must have EXACT names shown above
   • Files must be in the root model directory, not subfolders
   • All required files must be present for WAN to work

🚫 NO FALLBACKS: If files are missing, generation will fail completely.
   This is intentional to ensure proper WAN model usage.

🎯 Repository: https://huggingface.co/Wan-AI/Wan2.1-VACE-14B
═══════════════════════════════════════════════════════════════════════════════
""".format(model_path=self.model_path)
        
        return instructions
    
    def setup_wan_environment(self) -> bool:
        """
        Set up WAN environment - STRICT MODE
        """
        print("🚀 Setting up WAN environment...")
        
        wan_repo_dir = self.extension_root / "wan_official_repo"
        
        if not wan_repo_dir.exists() or not (wan_repo_dir / "wan").exists():
            raise RuntimeError(f"❌ WAN repository not found at {wan_repo_dir}")
        
        # Add to Python path
        if str(wan_repo_dir) not in sys.path:
            sys.path.insert(0, str(wan_repo_dir))
        
        # Install required dependencies
        self._install_wan_dependencies()
        
        print("✅ WAN environment ready")
        return True
    
    def _install_wan_dependencies(self):
        """Install WAN dependencies - STRICT MODE"""
        print("📦 Installing WAN dependencies...")
        
        required_deps = [
            "easydict",
            "einops>=0.8.0"
        ]
        
        for dep in required_deps:
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", dep, "--quiet"
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    print(f"  ✅ {dep}")
                else:
                    raise RuntimeError(f"Failed to install {dep}: {result.stderr}")
            except Exception as e:
                raise RuntimeError(f"Critical dependency installation failed for {dep}: {e}")
    
    def load_model(self):
        """
        Load WAN models - STRICT MODE, NO FALLBACKS
        """
        if self.loaded:
            return
            
        print("🔄 Loading WAN models (STRICT MODE - NO FALLBACKS)...")
        
        # STRICT validation - will raise exception if components missing
        validation = self.validate_model_structure()
        
        # Setup environment
        env_ready = self.setup_wan_environment()
        if not env_ready:
            raise RuntimeError("❌ WAN environment setup failed")
        
        # Load WAN pipeline
        try:
            self._load_wan_pipeline()
            print("✅ WAN pipeline loaded successfully")
        except Exception as e:
            raise RuntimeError(f"❌ WAN pipeline loading failed: {e}")
        
        self.loaded = True
        print("🎉 WAN models ready for generation")
    
    def _load_wan_pipeline(self):
        """Load actual WAN pipeline - STRICT MODE"""
        try:
            print("📦 Loading WAN components...")
            
            # Import WAN modules
            from wan.text2video import WanT2V
            from wan.configs.wan_t2v_14B import t2v_14B
            
            # Load T2V configuration
            self.wan_config = t2v_14B
            print(f"  Using WAN T2V 14B config")
            
            # Initialize WAN T2V - Fix device handling
            # Handle both string and torch.device objects
            if isinstance(self.device, str):
                device_str = self.device
            elif hasattr(self.device, 'type'):
                # torch.device object - convert to string  
                device_str = str(self.device)
            else:
                # Fallback - try to convert to string
                device_str = str(self.device)
            
            # Extract device ID from device string
            device_id = int(device_str.split(':')[-1]) if ':' in device_str else 0
            
            self.wan_t2v = WanT2V(
                config=self.wan_config,
                checkpoint_dir=str(self.model_path),
                device_id=device_id,
                rank=0,
                t5_cpu=True,  # Use CPU for T5 to save VRAM
                t5_fsdp=False,
                dit_fsdp=False
            )
            print("  ✅ WAN T2V pipeline loaded")
            
            # Note: I2V requires separate CLIP model not available in VACE-14B
            # Users would need to download from Wan-AI/Wan2.1-I2V-14B-720P for I2V support
            print("  ⚠️ I2V not available in VACE-14B model - T2V only mode")
            self.wan_i2v = None
                
        except Exception as e:
            raise RuntimeError(f"Failed to load WAN pipeline: {e}")
    
    def generate_txt2video(self, 
                          prompt: str, 
                          duration: float = 4.0,
                          fps: int = 60,
                          resolution: str = "1280x720",
                          steps: int = 50,
                          guidance_scale: float = 7.5,
                          seed: int = -1,
                          motion_strength: float = 1.0,
                          **kwargs) -> List[np.ndarray]:
        """Generate video from text - WAN ONLY, NO FALLBACKS"""
        
        if not self.loaded:
            self.load_model()
        
        if self.wan_t2v is None:
            raise RuntimeError("❌ WAN T2V pipeline not loaded - cannot generate video")
        
        width, height = map(int, resolution.split('x'))
        num_frames = self.calculate_frame_count(duration, fps)
        
        # Ensure frame count is 4n+1 for WAN
        if (num_frames - 1) % 4 != 0:
            num_frames = ((num_frames - 1) // 4) * 4 + 1
        
        print(f"🎬 Generating video with WAN T2V (from Wan-AI repository):")
        print(f"  Prompt: {prompt}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Frames: {num_frames} (adjusted for WAN 4n+1 requirement)")
        
        try:
            # Use actual WAN T2V generation
            video_tensor = self.wan_t2v.generate(
                input_prompt=prompt,
                size=(width, height),
                frame_num=num_frames,
                shift=5.0,
                sample_solver='unipc',
                sampling_steps=steps,
                guide_scale=guidance_scale,
                n_prompt="",  # Use default negative prompt from config
                seed=seed,
                offload_model=True
            )
            
            if video_tensor is None:
                raise RuntimeError("❌ WAN T2V returned None - generation failed")
            
            # Convert tensor to frame list
            frames = self._tensor_to_frames(video_tensor)
            
            if not frames:
                raise RuntimeError("❌ No frames extracted from WAN T2V output")
            
            print(f"✅ Generated {len(frames)} frames with WAN T2V")
            return frames
            
        except Exception as e:
            raise RuntimeError(f"❌ WAN T2V generation failed: {e}")
    
    def generate_img2video(self, 
                          init_image: np.ndarray,
                          prompt: str, 
                          duration: float = 4.0,
                          fps: int = 60,
                          resolution: str = "1280x720",
                          steps: int = 50,
                          guidance_scale: float = 7.5,
                          seed: int = -1,
                          motion_strength: float = 1.0,
                          **kwargs) -> List[np.ndarray]:
        """Generate video from image - WAN ONLY, NO FALLBACKS"""
        
        if not self.loaded:
            self.load_model()
        
        # VACE-14B model doesn't include I2V components
        raise RuntimeError("""❌ WAN I2V not available with VACE-14B model.

For Image-to-Video generation, you need to download the I2V model from:
https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P

The VACE-14B model only supports Text-to-Video generation.
Please use Text-to-Video mode or download the I2V model separately.""")
    
    def _tensor_to_frames(self, video_tensor: torch.Tensor) -> List[np.ndarray]:
        """Convert WAN video tensor to frame list"""
        try:
            # WAN returns tensor in format (C, T, H, W)
            if video_tensor.dim() == 4:
                # Transpose to (T, H, W, C)
                video_tensor = video_tensor.permute(1, 2, 3, 0)
            
            # Convert to numpy and scale to 0-255
            video_np = video_tensor.cpu().float().numpy()
            
            # Denormalize from [-1, 1] to [0, 1]
            video_np = (video_np + 1.0) / 2.0
            
            # Scale to [0, 255] and convert to uint8
            video_np = np.clip(video_np * 255.0, 0, 255).astype(np.uint8)
            
            # Convert to frame list (BGR format for consistency)
            frames = []
            for t in range(video_np.shape[0]):
                frame_rgb = video_np[t]
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                frames.append(frame_bgr)
            
            return frames
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert WAN tensor to frames: {e}")
    
    def calculate_frame_count(self, duration: float, fps: float) -> int:
        """Calculate number of frames"""
        if duration <= 0 or fps <= 0:
            raise ValueError("Duration and FPS must be positive")
        return max(1, int(duration * fps))
    
    def extract_last_frame(self, video_frames: List) -> np.ndarray:
        """Extract last frame from video"""
        if not video_frames:
            raise ValueError("Empty video frames list")
        
        last_frame = video_frames[-1]
        if isinstance(last_frame, np.ndarray):
            return last_frame.copy()
        elif hasattr(last_frame, 'mode'):  # PIL Image
            return cv2.cvtColor(np.array(last_frame), cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(f"Unsupported frame type: {type(last_frame)}")
    
    def unload_model(self):
        """Unload model and free memory"""
        if self.wan_t2v:
            del self.wan_t2v
            self.wan_t2v = None
        
        if self.wan_i2v:
            del self.wan_i2v
            self.wan_i2v = None
            
        if self.wan_config:
            del self.wan_config
            self.wan_config = None
        
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🧹 WAN model cleanup completed")
        self.loaded = False


class WanPromptScheduler:
    """Handle prompt scheduling for video generation"""
    
    def __init__(self, animation_prompts: Dict[str, str], wan_args, video_args):
        if not animation_prompts:
            raise ValueError("Animation prompts cannot be empty")
        self.animation_prompts = animation_prompts
        self.wan_args = wan_args
        self.video_args = video_args
        
    def parse_prompts_and_timing(self) -> List[Tuple[str, float, float]]:
        """Parse animation prompts and calculate timing"""
        frame_prompts = []
        
        for frame_str, prompt in self.animation_prompts.items():
            try:
                if isinstance(frame_str, str) and frame_str.isdigit():
                    frame_num = int(frame_str)
                elif isinstance(frame_str, (int, float)):
                    frame_num = int(frame_str)
                else:
                    continue
                frame_prompts.append((frame_num, prompt))
            except ValueError:
                continue
                
        frame_prompts.sort(key=lambda x: x[0])
        
        if not frame_prompts:
            raise ValueError("No valid frame prompts found")
            
        fps = self.wan_args.wan_fps
        default_duration = min(self.wan_args.wan_clip_duration, 8.0)  # Cap at 8 seconds for stability
        
        clips = []
        for i, (frame_num, prompt) in enumerate(frame_prompts):
            start_time = frame_num / fps
            
            if i < len(frame_prompts) - 1:
                next_frame = frame_prompts[i + 1][0]
                frame_count = next_frame - frame_num
                duration = frame_count / fps
            else:
                duration = default_duration
            
            # Ensure reasonable duration limits
            duration = max(0.5, min(duration, 8.0))
                
            clips.append((prompt, start_time, duration))
            
        return clips


def validate_wan_settings(wan_args) -> List[str]:
    """Validate WAN settings - STRICT MODE"""
    errors = []
    
    if wan_args.wan_enabled:
        try:
            width, height = map(int, wan_args.wan_resolution.split('x'))
            if width <= 0 or height <= 0:
                errors.append("Invalid resolution: dimensions must be positive")
        except (ValueError, AttributeError):
            errors.append(f"Invalid resolution format: {wan_args.wan_resolution}")
            
        if wan_args.wan_clip_duration <= 0 or wan_args.wan_clip_duration > 30:
            errors.append("Clip duration must be between 0 and 30 seconds")
            
        if wan_args.wan_fps <= 0 or wan_args.wan_fps > 60:
            errors.append("FPS must be between 1 and 60")
            
        if wan_args.wan_inference_steps < 1 or wan_args.wan_inference_steps > 100:
            errors.append("Inference steps must be between 1 and 100")
            
        if wan_args.wan_guidance_scale < 1.0 or wan_args.wan_guidance_scale > 20.0:
            errors.append("Guidance scale must be between 1.0 and 20.0")
    
    if errors:
        raise ValueError("WAN validation failed: " + "; ".join(errors))
    
    return []
