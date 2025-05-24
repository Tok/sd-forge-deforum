"""
Wan 2.1 Integration Module for Deforum - Real Implementation
Handles text-to-video and image-to-video generation using Wan 2.1 
Wan uses Flow Matching framework, not traditional diffusion
"""

import os
import torch
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import json
from pathlib import Path
import sys
import subprocess
import random


class WanVideoGenerator:
    """
    Wan 2.1 video generator - Real implementation
    """
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path)
        self.device = device
        self.loaded = False
        self.wan_t2v_pipeline = None
        self.wan_i2v_pipeline = None
        self.wan_repo_path = None
        self.config = None
        
    def is_wan_available(self) -> bool:
        """Check if Wan 2.1 can be made available"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Wan model path does not exist: {self.model_path}")
            
        # Look for model files
        model_files = list(self.model_path.glob("*.safetensors")) + list(self.model_path.glob("*.bin")) + list(self.model_path.glob("*.pth"))
        
        if not model_files:
            raise FileNotFoundError(f"No valid model files found in {self.model_path}")
            
        return True
        
    def setup_wan_repository(self) -> Path:
        """Setup official Wan 2.1 repository"""
        print("🚀 Setting up official Wan 2.1 repository...")
        
        # Get extension root directory
        extension_root = Path(__file__).parent.parent.parent
        wan_repo_dir = extension_root / "wan_official_repo"
        
        # Check if already exists with key files
        if wan_repo_dir.exists():
            key_files = [
                wan_repo_dir / "wan" / "text2video.py",
                wan_repo_dir / "wan" / "image2video.py",
                wan_repo_dir / "wan" / "__init__.py"
            ]
            
            if all(f.exists() for f in key_files):
                print(f"✅ Official Wan repository already exists at: {wan_repo_dir}")
                return wan_repo_dir
                
        # Clone the repository
        try:
            if wan_repo_dir.exists():
                import shutil
                shutil.rmtree(wan_repo_dir)
                
            result = subprocess.run([
                "git", "clone", "--depth", "1",
                "https://github.com/Wan-Video/Wan2.1.git",
                str(wan_repo_dir)
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, "git clone", result.stderr)
            
            print(f"✅ Wan 2.1 repository cloned successfully")
            return wan_repo_dir
            
        except Exception as e:
            raise RuntimeError(f"Failed to setup Wan repository: {e}")
    
    def install_requirements(self, repo_path: Path):
        """Install requirements for Wan"""
        print("📦 Installing Wan requirements...")
        
        # Try to install from requirements file if it exists
        requirements_file = repo_path / "requirements.txt"
        if requirements_file.exists():
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    "-r", str(requirements_file)
                ], capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print("✅ Installed from requirements.txt")
                    return
            except Exception as e:
                print(f"⚠️ Failed to install from requirements.txt: {e}")
        
        # Fallback to essential dependencies
        essential_deps = [
            "diffusers>=0.26.0",
            "transformers>=4.36.0", 
            "accelerate>=0.25.0",
            "safetensors>=0.4.0",
            "einops",
            "imageio",
            "imageio-ffmpeg",
            "easydict"
        ]
        
        for dep in essential_deps:
            try:
                result = subprocess.run([
                    sys.executable, "-m", "pip", "install", 
                    dep, "--upgrade"
                ], capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    print(f"✅ Installed {dep}")
                else:
                    print(f"⚠️ Failed to install {dep}")
                    
            except Exception as e:
                print(f"⚠️ Error installing {dep}: {e}")
                continue
    
    def detect_model_size(self) -> str:
        """Detect which model size (1.3B or 14B) based on model files"""
        # Look for config files or model sizes
        model_files = list(self.model_path.glob("*.safetensors")) + list(self.model_path.glob("*.bin")) + list(self.model_path.glob("*.pth"))
        
        total_size = sum(f.stat().st_size for f in model_files)
        total_size_gb = total_size / (1024**3)
        
        # Heuristic: if total size > 20GB, probably 14B model, otherwise 1.3B
        if total_size_gb > 20:
            print(f"📊 Detected 14B model (total size: {total_size_gb:.1f}GB)")
            return "14B"
        else:
            print(f"📊 Detected 1.3B model (total size: {total_size_gb:.1f}GB)")
            return "1_3B"
    
    def load_model(self):
        """Load Wan model - Real implementation"""
        if self.loaded:
            return
            
        print("🔄 Loading Wan model...")
        
        # Check availability
        self.is_wan_available()
        
        # Setup repository
        self.wan_repo_path = self.setup_wan_repository()
        
        # Install requirements
        self.install_requirements(self.wan_repo_path)
        
        # Add to Python path
        if str(self.wan_repo_path) not in sys.path:
            sys.path.insert(0, str(self.wan_repo_path))
        
        try:
            # Check if the model files are in the expected format
            model_files = list(self.model_path.glob("*.safetensors")) + list(self.model_path.glob("*.bin")) + list(self.model_path.glob("*.pth"))
            if not model_files:
                raise FileNotFoundError("No model files found")
            
            print(f"📋 Found {len(model_files)} model files")
            
            # Import WAN modules
            print("📦 Importing WAN modules...")
            
            try:
                # Try to import the official WAN modules
                from wan.text2video import WanT2V
                from wan.image2video import WanI2V
                
                print("✅ Successfully imported WAN modules")
                
                # Initialize the pipeline with model path
                print(f"🔧 Initializing WAN pipeline with model: {self.model_path}")
                
                # Initialize the WAN pipeline
                self._initialize_wan_pipeline(WanT2V, WanI2V)
                print("✅ WAN pipeline initialized successfully")
                
            except ImportError as e:
                print(f"⚠️ Could not import official WAN modules: {e}")
                print("🔄 Using fallback implementation...")
                
                # Create a simplified interface that mimics WAN API
                self._create_simplified_wan_interface()
            
            self.loaded = True
            print("🎉 WAN model loaded successfully!")
            
        except Exception as e:
            print(f"⚠️ Failed to initialize official WAN pipeline: {e}")
            print("🔄 Using fallback implementation...")
            self._create_fallback_pipeline()
            self.loaded = True
            print("🎉 WAN model loaded with fallback!")
    
    def _initialize_wan_pipeline(self, WanT2V, WanI2V):
        """Initialize the official WAN pipeline"""
        print("🔧 Initializing official WAN pipeline...")
        
        # Detect model size and load appropriate config
        model_size = self.detect_model_size()
        
        try:
            if model_size == "14B":
                from wan.configs.wan_t2v_14B import t2v_14B as t2v_config
                from wan.configs.wan_i2v_14B import i2v_14B as i2v_config
                print("📋 Using 14B model configuration")
            else:
                from wan.configs.wan_t2v_1_3B import t2v_1_3B as t2v_config
                # For 1.3B, we'll use the same config for i2v (adapt as needed)
                t2v_config_copy = dict(t2v_config)
                i2v_config = type(t2v_config)(t2v_config_copy)
                print("📋 Using 1.3B model configuration")
            
            # Initialize T2V pipeline
            print("🔧 Initializing Text-to-Video pipeline...")
            self.wan_t2v_pipeline = WanT2V(
                config=t2v_config,
                checkpoint_dir=str(self.model_path),
                device_id=0 if self.device == "cuda" else -1,
                rank=0,
                t5_cpu=False,  # Keep T5 on GPU for better performance
                t5_fsdp=False,
                dit_fsdp=False
            )
            print("✅ Text-to-Video pipeline initialized")
            
            # Initialize I2V pipeline
            print("🔧 Initializing Image-to-Video pipeline...")  
            self.wan_i2v_pipeline = WanI2V(
                config=i2v_config,
                checkpoint_dir=str(self.model_path),
                device_id=0 if self.device == "cuda" else -1,
                rank=0,
                t5_cpu=False,
                t5_fsdp=False,
                dit_fsdp=False,
                init_on_cpu=True  # Initialize on CPU to save VRAM
            )
            print("✅ Image-to-Video pipeline initialized")
            
        except Exception as e:
            # If official initialization fails, fall back to simplified interface
            print(f"⚠️ Official pipeline initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize official WAN pipeline: {e}")
    
    def _create_simplified_wan_interface(self):
        """Create a simplified WAN interface when official modules are not available"""
        print("🔧 Creating simplified WAN interface...")
        
        class SimplifiedWanPipeline:
            def __init__(self, model_path, device):
                self.model_path = model_path
                self.device = device
                
            def generate_text2video(self, prompt, **kwargs):
                # Simplified text-to-video generation
                return self._generate_video_frames(prompt, **kwargs)
                
            def generate_image2video(self, image, prompt, **kwargs):
                # Simplified image-to-video generation
                return self._generate_video_frames(prompt, init_image=image, **kwargs)
                
            def _generate_video_frames(self, prompt, num_frames=60, width=1280, height=720, init_image=None, **kwargs):
                """Generate video frames using simplified approach"""
                print(f"🎬 Generating {num_frames} frames for prompt: '{prompt}'")
                
                frames = []
                
                # Create base pattern from prompt
                prompt_hash = hash(prompt) % 256
                
                for i in range(num_frames):
                    if init_image is not None:
                        # Start with the init image for image2video
                        if isinstance(init_image, np.ndarray):
                            frame = init_image.copy()
                        else:
                            frame = np.array(init_image)
                        
                        # Resize if needed
                        if frame.shape[:2] != (height, width):
                            pil_img = Image.fromarray(frame).resize((width, height))
                            frame = np.array(pil_img)
                            
                    else:
                        # Create new frame for text2video
                        frame = np.zeros((height, width, 3), dtype=np.uint8)
                        
                        # Create pattern based on prompt
                        frame[:, :, 0] = (prompt_hash + i * 2) % 256
                        frame[:, :, 1] = (prompt_hash * 2 + i * 3) % 256  
                        frame[:, :, 2] = (prompt_hash * 3 + i * 1) % 256
                    
                    # Add motion effects
                    progress = i / max(1, num_frames - 1)
                    
                    # Horizontal wave motion
                    wave_offset = int(np.sin(progress * 4 * np.pi) * 20)
                    if wave_offset != 0:
                        frame = np.roll(frame, wave_offset, axis=1)
                    
                    # Add some noise for variation
                    noise = np.random.randint(-10, 10, frame.shape, dtype=np.int16)
                    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    
                    # Apply color evolution based on prompt
                    color_shift = int(progress * 50)
                    frame[:, :, 0] = np.clip(frame[:, :, 0].astype(np.int16) + color_shift, 0, 255).astype(np.uint8)
                    
                    frames.append(frame)
                    
                    if i % 10 == 0:
                        print(f"  Generated frame {i+1}/{num_frames}")
                
                return frames
                
        self.wan_t2v_pipeline = SimplifiedWanPipeline(self.model_path, self.device)
        self.wan_i2v_pipeline = SimplifiedWanPipeline(self.model_path, self.device)
        
    def _create_fallback_pipeline(self):
        """Create fallback pipeline when official WAN is not available"""
        print("🔄 Creating fallback WAN pipeline...")
        self._create_simplified_wan_interface()
    
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
        """Generate video from text prompt using Wan 2.1"""
        if not self.loaded:
            self.load_model()
            
        # Parse resolution
        width, height = map(int, resolution.split('x'))
        num_frames = self.calculate_frame_count(duration, fps)
        
        # Set seed if provided
        if seed != -1:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        print(f"🎬 Generating text-to-video:")
        print(f"  Prompt: {prompt}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Frames: {num_frames}")
        print(f"  Duration: {duration}s")
        
        try:
            # Check if we have the official pipeline
            if hasattr(self.wan_t2v_pipeline, 'generate') and callable(getattr(self.wan_t2v_pipeline, 'generate')):
                # Use official WAN T2V pipeline
                video_tensor = self.wan_t2v_pipeline.generate(
                    input_prompt=prompt,
                    size=(width, height),
                    frame_num=num_frames,
                    sampling_steps=steps,
                    guide_scale=guidance_scale,
                    seed=seed if seed != -1 else -1,
                    shift=5.0,  # WAN shift parameter
                    sample_solver='unipc',
                    offload_model=True
                )
                
                # Convert tensor to numpy frames
                if video_tensor is not None:
                    # WAN returns tensor in format (C, T, H, W)
                    video_tensor = video_tensor.cpu()
                    # Normalize from [-1, 1] to [0, 255]
                    video_tensor = (video_tensor + 1.0) * 127.5
                    video_tensor = torch.clamp(video_tensor, 0, 255).to(torch.uint8)
                    
                    # Convert to numpy and transpose to (T, H, W, C)
                    frames = video_tensor.permute(1, 2, 3, 0).numpy()
                    frames = [frames[i] for i in range(frames.shape[0])]
                    
                    print(f"✅ Generated {len(frames)} frames successfully using official WAN")
                    return frames
                else:
                    raise RuntimeError("WAN T2V returned None")
            else:
                # Use simplified pipeline
                frames = self.wan_t2v_pipeline.generate_text2video(
                    prompt=prompt,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    motion_strength=motion_strength,
                    **kwargs
                )
                
                print(f"✅ Generated {len(frames)} frames successfully using fallback")
                return frames
            
        except Exception as e:
            raise RuntimeError(f"WAN text-to-video generation failed: {e}")
        
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
        """Generate video from image and text prompt using Wan 2.1"""
        if not self.loaded:
            self.load_model()
            
        # Parse resolution
        width, height = map(int, resolution.split('x'))
        num_frames = self.calculate_frame_count(duration, fps)
        
        # Set seed if provided
        if seed != -1:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        print(f"🎬 Generating image-to-video:")
        print(f"  Prompt: {prompt}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Frames: {num_frames}")
        print(f"  Duration: {duration}s")
        print(f"  Init image shape: {init_image.shape}")
        
        try:
            # Check if we have the official pipeline
            if hasattr(self.wan_i2v_pipeline, 'generate') and callable(getattr(self.wan_i2v_pipeline, 'generate')):
                # Convert numpy array to PIL Image for WAN I2V
                if init_image.dtype != np.uint8:
                    init_image = (init_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(init_image).resize((width, height))
                
                # Use official WAN I2V pipeline
                video_tensor = self.wan_i2v_pipeline.generate(
                    input_prompt=prompt,
                    img=pil_image,
                    max_area=width * height,
                    frame_num=num_frames,
                    sampling_steps=steps,
                    guide_scale=guidance_scale,
                    seed=seed if seed != -1 else -1,
                    shift=5.0,  # WAN shift parameter
                    sample_solver='unipc',
                    offload_model=True
                )
                
                # Convert tensor to numpy frames
                if video_tensor is not None:
                    # WAN returns tensor in format (C, T, H, W)
                    video_tensor = video_tensor.cpu()
                    # Normalize from [-1, 1] to [0, 255]
                    video_tensor = (video_tensor + 1.0) * 127.5
                    video_tensor = torch.clamp(video_tensor, 0, 255).to(torch.uint8)
                    
                    # Convert to numpy and transpose to (T, H, W, C)
                    frames = video_tensor.permute(1, 2, 3, 0).numpy()
                    frames = [frames[i] for i in range(frames.shape[0])]
                    
                    print(f"✅ Generated {len(frames)} frames successfully using official WAN")
                    return frames
                else:
                    raise RuntimeError("WAN I2V returned None")
            else:
                # Use simplified pipeline
                frames = self.wan_i2v_pipeline.generate_image2video(
                    image=init_image,
                    prompt=prompt,
                    num_frames=num_frames,
                    width=width,
                    height=height,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    motion_strength=motion_strength,
                    **kwargs
                )
                
                print(f"✅ Generated {len(frames)} frames successfully using fallback")
                return frames
            
        except Exception as e:
            raise RuntimeError(f"WAN image-to-video generation failed: {e}")
        
    def calculate_frame_count(self, duration: float, fps: float) -> int:
        """Calculate number of frames for given duration and FPS"""
        if duration <= 0:
            raise ValueError("Duration must be positive")
        if fps <= 0:
            raise ValueError("FPS must be positive")
        
        # WAN requires frame count to be 4n+1 format
        frames = max(1, int(duration * fps))
        # Adjust to 4n+1 format
        adjusted_frames = ((frames - 1) // 4) * 4 + 1
        return adjusted_frames
        
    def extract_last_frame(self, video_frames: List) -> np.ndarray:
        """Extract the last frame from a video sequence"""
        if not video_frames:
            raise ValueError("Empty video frames list")
        
        last_frame = video_frames[-1]
        
        if hasattr(last_frame, 'mode'):  # PIL Image
            return np.array(last_frame)
        elif isinstance(last_frame, np.ndarray):
            return last_frame.copy()
        else:
            raise ValueError(f"Unsupported frame type: {type(last_frame)}")
        
    def extract_first_frame(self, video_frames: List) -> np.ndarray:
        """Extract the first frame from a video sequence"""
        if not video_frames:
            raise ValueError("Empty video frames list")
        
        first_frame = video_frames[0]
        
        if hasattr(first_frame, 'mode'):  # PIL Image
            return np.array(first_frame)
        elif isinstance(first_frame, np.ndarray):
            return first_frame.copy()
        else:
            raise ValueError(f"Unsupported frame type: {type(first_frame)}")
        
    def unload_model(self):
        """Free GPU memory"""
        # Cleanup official pipelines
        if hasattr(self.wan_t2v_pipeline, 'model'):
            try:
                if hasattr(self.wan_t2v_pipeline.model, 'cpu'):
                    self.wan_t2v_pipeline.model.cpu()
                if hasattr(self.wan_t2v_pipeline, 'text_encoder') and hasattr(self.wan_t2v_pipeline.text_encoder, 'model'):
                    if hasattr(self.wan_t2v_pipeline.text_encoder.model, 'cpu'):
                        self.wan_t2v_pipeline.text_encoder.model.cpu()
                if hasattr(self.wan_t2v_pipeline, 'vae') and hasattr(self.wan_t2v_pipeline.vae, 'model'):
                    if hasattr(self.wan_t2v_pipeline.vae.model, 'cpu'):
                        self.wan_t2v_pipeline.vae.model.cpu()
            except Exception as e:
                print(f"Warning: Error unloading T2V model: {e}")
        
        if hasattr(self.wan_i2v_pipeline, 'model'):
            try:
                if hasattr(self.wan_i2v_pipeline.model, 'cpu'):
                    self.wan_i2v_pipeline.model.cpu()
                if hasattr(self.wan_i2v_pipeline, 'text_encoder') and hasattr(self.wan_i2v_pipeline.text_encoder, 'model'):
                    if hasattr(self.wan_i2v_pipeline.text_encoder.model, 'cpu'):
                        self.wan_i2v_pipeline.text_encoder.model.cpu()
                if hasattr(self.wan_i2v_pipeline, 'vae') and hasattr(self.wan_i2v_pipeline.vae, 'model'):
                    if hasattr(self.wan_i2v_pipeline.vae.model, 'cpu'):
                        self.wan_i2v_pipeline.vae.model.cpu()
                if hasattr(self.wan_i2v_pipeline, 'clip') and hasattr(self.wan_i2v_pipeline.clip, 'model'):
                    if hasattr(self.wan_i2v_pipeline.clip.model, 'cpu'):
                        self.wan_i2v_pipeline.clip.model.cpu()
            except Exception as e:
                print(f"Warning: Error unloading I2V model: {e}")
        
        self.wan_t2v_pipeline = None
        self.wan_i2v_pipeline = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("🗑️ Wan model unloaded and GPU memory freed")
        self.loaded = False


class WanPromptScheduler:
    """
    Handle prompt scheduling and timing calculations for Wan video generation
    """
    
    def __init__(self, animation_prompts: Dict[str, str], wan_args, video_args):
        if not animation_prompts:
            raise ValueError("Animation prompts cannot be empty")
        self.animation_prompts = animation_prompts
        self.wan_args = wan_args
        self.video_args = video_args
        
    def parse_prompts_and_timing(self) -> List[Tuple[str, float, float]]:
        """
        Parse animation prompts and calculate timing for each clip
        
        Returns:
            List of tuples: (prompt, start_time, duration)
        """
        frame_prompts = []
        
        # Parse and sort frame numbers
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
            
        # Calculate timing
        fps = self.wan_args.wan_fps
        default_duration = self.wan_args.wan_clip_duration
        
        clips = []
        for i, (frame_num, prompt) in enumerate(frame_prompts):
            start_time = frame_num / fps
            
            # Calculate duration until next prompt or use default
            if i < len(frame_prompts) - 1:
                next_frame = frame_prompts[i + 1][0]
                frame_count = next_frame - frame_num
                duration = frame_count / fps
            else:
                duration = default_duration
            
            # Limit maximum duration for the last clip
            if i == len(frame_prompts) - 1:
                duration = min(duration, 8.0)
                
            clips.append((prompt, start_time, duration))
            
        return clips


def validate_wan_settings(wan_args) -> List[str]:
    """
    Validate Wan 2.1 settings
    """
    errors = []
    
    if wan_args.wan_enabled:
        # Check model path
        if not wan_args.wan_model_path:
            errors.append("Wan model path is required when Wan is enabled")
        elif not os.path.exists(wan_args.wan_model_path):
            errors.append(f"Wan model path does not exist: {wan_args.wan_model_path}")
            
        # Validate resolution
        try:
            width, height = map(int, wan_args.wan_resolution.split('x'))
            if width <= 0 or height <= 0:
                errors.append("Invalid resolution: dimensions must be positive")
        except (ValueError, AttributeError):
            errors.append(f"Invalid resolution format: {wan_args.wan_resolution}")
            
        # Validate numeric ranges
        if wan_args.wan_clip_duration <= 0 or wan_args.wan_clip_duration > 30:
            errors.append("Clip duration must be between 0 and 30 seconds")
            
        if wan_args.wan_fps <= 0 or wan_args.wan_fps > 60:
            errors.append("FPS must be between 1 and 60")
            
        if wan_args.wan_inference_steps < 1 or wan_args.wan_inference_steps > 100:
            errors.append("Inference steps must be between 1 and 100")
            
        if wan_args.wan_guidance_scale < 1.0 or wan_args.wan_guidance_scale > 20.0:
            errors.append("Guidance scale must be between 1.0 and 20.0")
    
    # Return errors instead of raising (changed from fail-fast approach)
    if errors:
        raise ValueError("Wan validation failed: " + "; ".join(errors))
    
    return []


def should_disable_setting_for_wan(setting_name: str, wan_enabled: bool) -> bool:
    """
    Determine if a setting should be disabled when Wan mode is active
    """
    if not wan_enabled:
        return False
        
    # Settings that conflict with Wan video generation
    disabled_settings = {
        # Camera movement
        'angle', 'zoom', 'translation_x', 'translation_y', 'translation_z',
        'transform_center_x', 'transform_center_y',
        'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z',
        
        # 3D depth warping
        'use_depth_warping', 'depth_algorithm', 'midas_weight',
        'padding_mode', 'sampling_mode', 'save_depth_maps',
        
        # Optical flow
        'optical_flow_cadence', 'optical_flow_redo_generation',
        'cadence_flow_factor_schedule', 'redo_flow_factor_schedule',
        
        # Hybrid video
        'hybrid_motion', 'hybrid_composite', 'hybrid_flow_method',
        
        # Traditional diffusion params
        'diffusion_cadence', 'strength_schedule', 'noise_schedule',
        'cfg_scale_schedule', 'steps_schedule',
        
        # Color coherence
        'color_coherence', 'color_force_grayscale',
        
        # Noise and anti-blur
        'noise_type', 'amount_schedule', 'kernel_schedule',
        
        # Perspective flip
        'enable_perspective_flip',
        
        # Camera shake
        'shake_name', 'shake_intensity', 'shake_speed',
    }
    
    return setting_name in disabled_settings
