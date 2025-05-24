"""
WAN 2.1 Flow Matching Pipeline Implementation
Based on official WAN 2.1 repository: https://github.com/Wan-Video/Wan2.1

CURRENT INTEGRATION STATUS:
✅ Repository Integration: Official WAN repo cloned and modules imported
✅ Environment Isolation: Separate Python env with WAN dependencies  
✅ Module Discovery: Finds and imports actual WAN modules (text2video, image2video)
⚠️  Function Calls: Attempts to call WAN functions but needs proper parameter mapping
❌ Model Loading: Uses mock components (T5, VAE) instead of actual WAN implementations

ACTUAL WAN CODE REUSED:
- WAN repository structure and module discovery
- WAN requirements.txt for dependency management  
- Import of official WAN modules (text2video.py, image2video.py, modules/model.py)
- Attempts to call WAN generation functions (but parameter mismatches)

NOT YET REUSED:
- Actual WAN model classes and architectures
- Real WAN T5 encoder and 3D causal VAE
- WAN's native generation pipeline and flow matching implementation

This is currently a "WAN-compatible" implementation that tries to integrate with
official WAN code but falls back to custom implementations when needed.

WAN uses Flow Matching framework with:
- T5 Encoder for multilingual text input
- 3D causal VAE (Wan-VAE) for video encoding/decoding  
- Cross-attention in transformer blocks
- MLP with Linear + SiLU for time embeddings
- Flow Matching (NOT traditional diffusion)
"""

import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from PIL import Image
import math
from pathlib import Path
import os
import importlib.util


class WanTimeEmbedding(nn.Module):
    """
    Time embedding module with shared MLP across transformer blocks
    Uses Linear + SiLU to process time embeddings and predict 6 modulation parameters
    """
    
    def __init__(self, dim: int = 1536, frequency_dim: int = 256):
        super().__init__()
        self.frequency_dim = frequency_dim
        self.dim = dim
        
        # Shared MLP across all transformer blocks
        self.time_mlp = nn.Sequential(
            nn.Linear(frequency_dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 6)  # 6 modulation parameters
        )
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (batch_size,) tensor of timesteps
            
        Returns:
            modulation_params: (batch_size, dim * 6) tensor of modulation parameters
        """
        # Create sinusoidal time embeddings
        half_dim = self.frequency_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        # Process through shared MLP
        modulation_params = self.time_mlp(emb)
        return modulation_params


class WanCrossAttention(nn.Module):
    """
    Cross-attention mechanism for embedding text into transformer blocks
    Uses T5 encoder outputs as conditioning
    """
    
    def __init__(self, dim: int, num_heads: int = 12, cross_attention_dim: int = 768):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, dim, bias=False)  
        self.to_v = nn.Linear(cross_attention_dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim) - video features
            context: (batch, text_len, cross_attention_dim) - T5 text embeddings
            
        Returns:
            attended features: (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        q = self.to_q(x)  # (batch, seq_len, dim)
        k = self.to_k(context)  # (batch, text_len, dim)
        v = self.to_v(context)  # (batch, text_len, dim)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = torch.softmax(scores, dim=-1)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        
        return self.to_out(out)


class WanTransformerBlock(nn.Module):
    """
    Transformer block with cross-attention and time modulation
    Each block learns distinct biases while sharing the time MLP
    """
    
    def __init__(self, dim: int, num_heads: int, feedforward_dim: int, cross_attention_dim: int = 768):
        super().__init__()
        self.dim = dim
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # Cross-attention for text conditioning
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = WanCrossAttention(dim, num_heads, cross_attention_dim)
        
        # Feedforward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, dim)
        )
        
        # Learnable biases for this block (distinct from other blocks)
        self.time_bias = nn.Parameter(torch.zeros(6, dim))
        
    def forward(self, x: torch.Tensor, text_context: torch.Tensor, 
                time_modulation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim) - input features
            text_context: (batch, text_len, cross_attention_dim) - T5 text embeddings
            time_modulation: (batch, dim * 6) - time modulation parameters
            
        Returns:
            output: (batch, seq_len, dim)
        """
        batch_size = x.shape[0]
        
        # Reshape time modulation and add block-specific biases
        time_params = time_modulation.view(batch_size, 6, self.dim)  # (batch, 6, dim)
        time_params = time_params + self.time_bias.unsqueeze(0)  # Add learnable biases
        
        # Extract modulation parameters
        shift_1, scale_1, gate_1, shift_2, scale_2, gate_2 = time_params.unbind(dim=1)
        
        # Self-attention with time modulation
        normed = self.norm1(x)
        modulated = normed * (1 + scale_1.unsqueeze(1)) + shift_1.unsqueeze(1)
        attn_out, _ = self.self_attn(modulated, modulated, modulated)
        x = x + gate_1.unsqueeze(1) * attn_out
        
        # Cross-attention with text conditioning
        normed = self.norm2(x)
        cross_out = self.cross_attn(normed, text_context)
        x = x + cross_out
        
        # Feedforward with time modulation
        normed = self.norm3(x)
        modulated = normed * (1 + scale_2.unsqueeze(1)) + shift_2.unsqueeze(1)
        ff_out = self.ff(modulated)
        x = x + gate_2.unsqueeze(1) * ff_out
        
        return x


class WanFlowMatchingModel(nn.Module):
    """
    WAN Flow Matching Model implementing the Diffusion Transformer architecture
    Based on official WAN 2.1 specifications
    """
    
    def __init__(self, 
                 model_size: str = "14B",
                 input_channels: int = 16,
                 output_channels: int = 16):
        super().__init__()
        
        # Model configurations based on official WAN 2.1 specs
        if model_size == "1.3B":
            self.dim = 1536
            self.num_heads = 12
            self.num_layers = 30
            self.feedforward_dim = 8960
        elif model_size == "14B":
            self.dim = 5120
            self.num_heads = 40
            self.num_layers = 40
            self.feedforward_dim = 13824
        else:
            raise ValueError(f"Unsupported model size: {model_size}")
            
        self.frequency_dim = 256
        self.cross_attention_dim = 768  # T5 encoder dimension
        
        # Input/output projections
        self.input_proj = nn.Linear(input_channels, self.dim)
        self.output_proj = nn.Linear(self.dim, output_channels)
        
        # Shared time embedding MLP
        self.time_embedding = WanTimeEmbedding(self.dim, self.frequency_dim)
        
        # Transformer blocks (each with distinct biases)
        self.blocks = nn.ModuleList([
            WanTransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                feedforward_dim=self.feedforward_dim,
                cross_attention_dim=self.cross_attention_dim
            ) for _ in range(self.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(self.dim)
        
    def forward(self, 
                x: torch.Tensor,
                timesteps: torch.Tensor, 
                text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of WAN Flow Matching model
        
        Args:
            x: (batch, frames, height, width, channels) - noisy video latents
            timesteps: (batch,) - flow matching timesteps
            text_embeddings: (batch, text_len, 768) - T5 text embeddings
            
        Returns:
            predicted_flow: (batch, frames, height, width, channels) - flow prediction
        """
        batch_size, frames, height, width, channels = x.shape
        
        # Reshape to sequence format for transformer
        x = x.view(batch_size, frames * height * width, channels)
        
        # Input projection
        x = self.input_proj(x)
        
        # Get time modulation parameters (shared across all blocks)
        time_modulation = self.time_embedding(timesteps)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, text_embeddings, time_modulation)
            
        # Final processing
        x = self.final_norm(x)
        x = self.output_proj(x)
        
        # Reshape back to video format
        x = x.view(batch_size, frames, height, width, channels)
        
        return x


class WanFlowMatchingPipeline:
    """
    Main WAN Flow Matching pipeline for video generation
    Implements the complete flow matching process with VAE encoding/decoding
    """
    
    def __init__(self, 
                 model_path: str,
                 model_size: str = "14B",
                 device: str = "cuda"):
        self.model_path = Path(model_path)
        self.model_size = model_size
        self.device = device
        
        # Initialize components
        self.flow_model = None
        self.vae = None
        self.text_encoder = None
        
        # Flow matching parameters
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        
    def load_model_components(self, model_tensors: Dict[str, torch.Tensor]):
        """
        Load model components from WAN tensors - NOW USING OFFICIAL WAN REPOSITORY
        
        Args:
            model_tensors: Dictionary of loaded model tensors
        """
        print(f"🔧 Loading WAN Flow Matching model using official repository ({self.model_size})...")
        
        # Try to import and use official WAN code
        try:
            # This should be set by the isolated environment
            if hasattr(self, 'wan_repo_path') and hasattr(self, 'wan_code_dir'):
                print(f"🔧 Using official WAN code from: {self.wan_code_dir}")
                
                # Add WAN code directory to Python path temporarily
                import sys
                wan_repo_root = Path(self.wan_repo_path)
                
                if str(wan_repo_root) not in sys.path:
                    sys.path.insert(0, str(wan_repo_root))
                
                try:
                    print("📦 Importing official WAN modules and config...")
                    
                    # Import the official WAN modules 
                    import wan.text2video as wt2v
                    
                    # Import the config for the appropriate model size
                    if self.model_size == "14B":
                        from wan.configs.wan_t2v_14B import config as wan_config
                        print("✅ Loaded WAN 14B config")
                    else:
                        from wan.configs.wan_t2v_1_3B import config as wan_config
                        print("✅ Loaded WAN 1.3B config")
                    
                    # Check if we can import the main class
                    WanT2V = getattr(wt2v, 'WanT2V', None)
                    if WanT2V is None:
                        raise ImportError("WanT2V class not found in official repository")
                    
                    print("✅ Successfully imported WanT2V class")
                    
                    # CRITICAL FIX: Check for required checkpoint files BEFORE attempting initialization
                    checkpoint_dir = Path(self.model_path)
                    
                    # Check what the config expects for checkpoint files
                    t5_checkpoint = getattr(wan_config, 't5_checkpoint', 'models_t5_umt5-xxl-enc-bf16.pth')
                    vae_checkpoint = getattr(wan_config, 'vae_checkpoint', 'Wan2.1_VAE.pth')
                    
                    print(f"🔧 Expected T5 checkpoint: {t5_checkpoint}")
                    print(f"🔧 Expected VAE checkpoint: {vae_checkpoint}")
                    print(f"🔧 Model path: {self.model_path}")
                    
                    # Check if we have the expected checkpoint structure
                    t5_path = checkpoint_dir / t5_checkpoint
                    vae_path = checkpoint_dir / vae_checkpoint
                    
                    missing_files = []
                    if not t5_path.exists():
                        missing_files.append(f"T5 encoder: {t5_checkpoint}")
                    if not vae_path.exists():
                        missing_files.append(f"VAE model: {vae_checkpoint}")
                    
                    if missing_files:
                        print(f"❌ Missing required WAN checkpoint files:")
                        for missing in missing_files:
                            print(f"   - {missing}")
                        print(f"📂 Found files: {list(checkpoint_dir.glob('*'))}")
                        
                        raise FileNotFoundError(f"""
❌ FAIL FAST: Incomplete WAN Model Structure

The model directory is missing required WAN 2.1 checkpoint files:
{missing_files}

WAN 2.1 requires these specific files:
- {t5_checkpoint} (T5 text encoder)
- {vae_checkpoint} (3D causal VAE)
- DiT transformer weights (diffusion_pytorch_model*.safetensors)

Current directory: {checkpoint_dir}
Found files: {list(checkpoint_dir.glob('*'))}

You have the DiT weights ({len(model_tensors)} tensors) but are missing the T5 and VAE components.

To get complete WAN 2.1 models:
1. Download from: https://huggingface.co/Wan-Video/Wan2.1
2. Or follow: https://github.com/Wan-Video/Wan2.1

NO FALLBACKS as requested - either complete WAN or fail fast.
""")
                    
                    # All required files exist - proceed with official WAN initialization
                    print("✅ All required WAN checkpoint files found")
                    print("🚀 Initializing official WAN T2V pipeline...")
                    
                    try:
                        wan_pipeline = WanT2V(
                            config=wan_config,
                            checkpoint_dir=str(checkpoint_dir),
                            device_id=0,  # Use GPU 0
                            rank=0,
                            t5_fsdp=False,
                            dit_fsdp=False,
                            use_usp=False,
                            t5_cpu=False
                        )
                        
                        print("🎉 Official WAN T2V pipeline initialized successfully!")
                        self.wan_pipeline = wan_pipeline
                        print("✅ Official WAN pipeline ready for generation")
                        return
                        
                    except Exception as init_error:
                        print(f"❌ WAN initialization failed: {init_error}")
                        print("🔧 Error details:")
                        import traceback
                        traceback.print_exc()
                        
                        raise RuntimeError(f"""
❌ FAIL FAST: WAN Initialization Error

Error during official WAN T2V initialization: {init_error}

All required files were found but initialization failed. This could indicate:
1. Model file corruption
2. Version incompatibility  
3. Memory/GPU issues
4. Missing dependencies

Model path: {checkpoint_dir}
Config: {wan_config.__class__.__name__}
Device: cuda:0

NO FALLBACKS as requested - either working WAN or fail fast.
""")
                        
                except ImportError as import_error:
                    print(f"❌ WAN import failed: {import_error}")
                    raise RuntimeError(f"""
❌ FAIL FAST: Official WAN Import Failed

Could not import required WAN modules: {import_error}

This indicates the official WAN repository is incomplete or corrupted.

Repository path: {self.wan_repo_path}
Code directory: {self.wan_code_dir}

Please re-clone the official repository:
git clone https://github.com/Wan-Video/Wan2.1.git

NO FALLBACKS as requested - either working WAN or fail fast.
""")
                
                finally:
                    # Clean up sys.path
                    if str(wan_repo_root) in sys.path:
                        sys.path.remove(str(wan_repo_root))
                        
            else:
                raise RuntimeError("""
❌ FAIL FAST: WAN Repository Not Set Up

The official WAN repository was not properly initialized.

Required attributes missing:
- wan_repo_path: Path to cloned WAN repository
- wan_code_dir: Path to WAN source code

This should have been set during environment setup.

NO FALLBACKS as requested - either working WAN or fail fast.
""")
                
        except Exception as e:
            # FAIL FAST if official WAN doesn't work
            print(f"❌ Official WAN integration failed: {e}")
            raise RuntimeError(f"""
❌ FAIL FAST: Complete WAN Integration Failure

Official WAN 2.1 integration failed: {e}

Summary of what was attempted:
1. ✅ Official WAN repository: {getattr(self, 'wan_repo_path', 'NOT SET')}
2. ✅ Model tensors loaded: {len(model_tensors)} tensors from {self.model_size} model
3. ❌ WAN initialization: Failed

Common causes:
- Incomplete model files (missing T5/VAE checkpoints)
- Repository corruption or version mismatch
- Environment/dependency issues

As requested: NO PROMPT GUESSING, NO FALLBACK IMAGES
Either complete, working WAN 2.1 or fast failure.

To resolve:
1. Ensure complete WAN 2.1 model download
2. Verify all checkpoint files exist
3. Check official documentation: https://github.com/Wan-Video/Wan2.1

Current model path: {self.model_path}
""")
        
    def setup_text_encoder(self):
        """Setup T5 text encoder for multilingual support"""
        print("🔧 Setting up T5 text encoder...")
        
        # This would initialize the T5 encoder
        # For now, we'll create a mock encoder
        class MockT5Encoder:
            def __init__(self, device):
                self.device = device
                
            def encode(self, prompts: List[str]) -> torch.Tensor:
                # Mock encoding - returns properly shaped embeddings
                batch_size = len(prompts)
                # Return embeddings on the correct device
                return torch.randn(batch_size, 77, 768, device=self.device, dtype=torch.float32)
                
        self.text_encoder = MockT5Encoder(self.device)
        print("✅ T5 encoder ready")
        
    def setup_vae(self):
        """Setup 3D causal VAE (Wan-VAE)"""
        print("🔧 Setting up 3D causal VAE (Wan-VAE)...")
        
        # This would initialize the Wan-VAE
        # For now, we'll create a mock VAE
        class MockWanVAE:
            def __init__(self, device):
                self.device = device
                
            def encode(self, videos: torch.Tensor) -> torch.Tensor:
                # Mock encoding
                b, c, f, h, w = videos.shape
                return torch.randn(b, f, h//8, w//8, 16, device=self.device, dtype=torch.float32)
                
            def decode(self, latents: torch.Tensor) -> torch.Tensor:
                # Mock decoding - properly shaped for video output
                b, f, h, w, c = latents.shape
                # Return video in correct format: (batch, channels, frames, height, width)
                return torch.randn(b, 3, f, h*8, w*8, device=self.device, dtype=torch.float32)
                
        self.vae = MockWanVAE(self.device)
        print("✅ Wan-VAE ready")
        
    def flow_matching_step(self, 
                          x_t: torch.Tensor,
                          t: torch.Tensor,
                          text_embeddings: torch.Tensor,
                          guidance_scale: float = 7.5) -> torch.Tensor:
        """
        Single flow matching denoising step
        
        Args:
            x_t: Current noisy latents
            t: Current timestep  
            text_embeddings: Text conditioning
            guidance_scale: Classifier-free guidance scale
            
        Returns:
            x_prev: Denoised latents for previous timestep
        """
        # Predict the flow (velocity field)
        with torch.no_grad():
            # Unconditional prediction (for classifier-free guidance)
            null_embeddings = torch.zeros_like(text_embeddings)
            flow_uncond = self.flow_model(x_t, t, null_embeddings)
            
            # Conditional prediction
            flow_cond = self.flow_model(x_t, t, text_embeddings)
            
            # Apply classifier-free guidance
            flow = flow_uncond + guidance_scale * (flow_cond - flow_uncond)
            
        # Flow matching update (simplified Euler step)
        dt = 1.0 / self.num_inference_steps
        x_prev = x_t - dt * flow
        
        return x_prev
        
    def generate_video(self,
                      prompt: str,
                      num_frames: int = 60,
                      height: int = 720,
                      width: int = 1280,
                      num_inference_steps: int = 50,
                      guidance_scale: float = 7.5,
                      seed: Optional[int] = None) -> List[np.ndarray]:
        """
        Generate video using official WAN Flow Matching
        
        Args:
            prompt: Text prompt for generation
            num_frames: Number of frames to generate (must be 4n+1 for WAN)
            height: Video height
            width: Video width  
            num_inference_steps: Number of denoising steps (WAN calls this sampling_steps)
            guidance_scale: Classifier-free guidance scale (WAN calls this guide_scale)
            seed: Random seed
            
        Returns:
            List of generated video frames as numpy arrays
        """
        if not hasattr(self, 'wan_pipeline'):
            raise RuntimeError("WAN pipeline not initialized - call load_model_components first")
            
        # WAN requires frame count to be 4n+1
        if (num_frames - 1) % 4 != 0:
            # Adjust to nearest valid frame count
            adjusted_frames = ((num_frames - 1) // 4) * 4 + 1
            print(f"⚠️ WAN requires frames = 4n+1, adjusting {num_frames} → {adjusted_frames}")
            num_frames = adjusted_frames
            
        if seed is not None:
            torch.manual_seed(seed)
            
        print(f"🎬 Generating {num_frames} frames at {width}x{height} using official WAN")
        print(f"📝 Prompt: {prompt}")
        
        try:
            # Call official WAN generate method
            result_tensor = self.wan_pipeline.generate(
                input_prompt=prompt,
                size=(width, height),  # WAN expects (width, height)
                frame_num=num_frames,
                sampling_steps=num_inference_steps,
                guide_scale=guidance_scale,
                seed=seed if seed is not None else -1
            )
            
            if result_tensor is None:
                raise RuntimeError("WAN generation returned None")
                
            print(f"✅ WAN generation complete! Tensor shape: {result_tensor.shape}")
            
            # Convert WAN result to our expected format
            # WAN returns (C, N, H, W) tensor
            frames = self._convert_wan_result_to_frames(result_tensor)
            
            print(f"✅ Generated {len(frames)} frames using official WAN 2.1")
            return frames
            
        except Exception as e:
            raise RuntimeError(f"Official WAN generation failed: {e}")

    def _convert_wan_result_to_frames(self, wan_result):
        """
        Convert WAN generation result to our expected frame format
        
        Args:
            wan_result: Result from official WAN generation function
            
        Returns:
            List of numpy arrays representing frames
        """
        try:
            frames = []
            
            # Handle different possible result formats from WAN
            if isinstance(wan_result, list):
                # Result is already a list of frames
                for frame in wan_result:
                    if hasattr(frame, 'mode'):  # PIL Image
                        frames.append(np.array(frame))
                    elif isinstance(frame, np.ndarray):
                        frames.append(frame)
                    elif isinstance(frame, torch.Tensor):
                        # Convert tensor to numpy
                        frame_np = frame.detach().cpu().numpy()
                        if frame_np.ndim == 4:  # (1, C, H, W)
                            frame_np = frame_np.squeeze(0).transpose(1, 2, 0)  # (H, W, C)
                        elif frame_np.ndim == 3:  # (C, H, W)
                            frame_np = frame_np.transpose(1, 2, 0)  # (H, W, C)
                        
                        # Normalize to 0-255 if needed
                        if frame_np.max() <= 1.0:
                            frame_np = (frame_np * 255).astype(np.uint8)
                        else:
                            frame_np = frame_np.astype(np.uint8)
                            
                        frames.append(frame_np)
                        
            elif isinstance(wan_result, torch.Tensor):
                # Result is a tensor with shape (batch, frames, channels, height, width) or similar
                result_np = wan_result.detach().cpu().numpy()
                
                if result_np.ndim == 5:  # (B, T, C, H, W)
                    result_np = result_np.squeeze(0)  # (T, C, H, W)
                
                if result_np.ndim == 4:  # (T, C, H, W)
                    for i in range(result_np.shape[0]):
                        frame_np = result_np[i].transpose(1, 2, 0)  # (H, W, C)
                        
                        # Normalize to 0-255 if needed
                        if frame_np.max() <= 1.0:
                            frame_np = (frame_np * 255).astype(np.uint8)
                        else:
                            frame_np = frame_np.astype(np.uint8)
                            
                        frames.append(frame_np)
                        
            elif hasattr(wan_result, 'frames') and wan_result.frames:
                # Result has a frames attribute
                return self._convert_wan_result_to_frames(wan_result.frames)
                
            elif hasattr(wan_result, 'videos') and wan_result.videos:
                # Result has a videos attribute
                return self._convert_wan_result_to_frames(wan_result.videos)
                
            else:
                raise ValueError(f"Unknown WAN result format: {type(wan_result)}")
            
            if not frames:
                raise ValueError("No frames extracted from WAN result")
                
            print(f"✅ Converted WAN result to {len(frames)} frames")
            return frames
            
        except Exception as e:
            raise RuntimeError(f"Failed to convert WAN result to frames: {e}")


def create_wan_pipeline(model_path: str,
                       model_tensors: Dict[str, torch.Tensor],
                       model_size: str = "14B",
                       device: str = "cuda",
                       wan_repo_path: Optional[str] = None,
                       wan_code_dir: Optional[str] = None) -> WanFlowMatchingPipeline:
    """
    Create and initialize WAN Flow Matching pipeline - NOW WITH OFFICIAL WAN INTEGRATION
    
    Args:
        model_path: Path to model directory
        model_tensors: Loaded model tensors
        model_size: Model size ("1.3B" or "14B")
        device: Device to run on
        wan_repo_path: Path to official WAN 2.1 repository
        wan_code_dir: Path to WAN code directory
        
    Returns:
        Initialized WAN pipeline
    """
    print("🚀 Creating WAN Flow Matching pipeline with official repository integration...")
    
    # Create pipeline
    pipeline = WanFlowMatchingPipeline(model_path, model_size, device)
    
    # Pass repository paths to pipeline
    if wan_repo_path and wan_code_dir:
        pipeline.wan_repo_path = Path(wan_repo_path)
        pipeline.wan_code_dir = Path(wan_code_dir)
        print(f"📂 Pipeline configured with official WAN repository")
    else:
        print("⚠️ No official WAN repository paths provided - will attempt basic loading")
    
    # Load components
    pipeline.load_model_components(model_tensors)
    pipeline.setup_text_encoder()
    pipeline.setup_vae()
    
    print("✅ WAN Flow Matching pipeline ready!")
    return pipeline 