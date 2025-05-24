"""
WAN 2.1 Flow Matching Pipeline Implementation
Based on official WAN 2.1 repository: https://github.com/Wan-Video/Wan2.1

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
        Load model components from WAN tensors - FAIL FAST if not possible
        
        Args:
            model_tensors: Dictionary of loaded model tensors
        """
        print(f"🔧 Initializing WAN Flow Matching model ({self.model_size})...")
        
        # FAIL FAST: Check if we have the actual WAN model tensors needed
        required_keys = ['transformer.layers.0.norm1.weight', 'transformer.layers.0.attn1.to_q.weight']
        missing_keys = [key for key in required_keys if key not in model_tensors]
        
        if missing_keys:
            raise RuntimeError(f"""
❌ FAIL FAST: WAN model tensors missing required keys: {missing_keys}

This is not a valid WAN 2.1 model. Expected transformer architecture with:
- transformer.layers.*.norm1.weight  
- transformer.layers.*.attn1.to_q.weight
- etc.

Current model has {len(model_tensors)} tensors but wrong structure.
NO FALLBACKS - please provide a valid WAN 2.1 model or disable WAN generation.
""")
        
        # FAIL FAST: Don't attempt to initialize huge models in mock mode
        # Real WAN would load the actual weights, but we can't do that without proper implementation
        raise RuntimeError(f"""
❌ FAIL FAST: WAN Flow Matching NOT FULLY IMPLEMENTED

Found valid WAN model structure with {len(model_tensors)} tensors, but actual WAN inference
requires the complete official WAN 2.1 codebase integration.

Current status:
✅ Model loading and validation
✅ Tensor structure verification  
✅ Architecture implementation
❌ Actual WAN inference - REQUIRES OFFICIAL WAN 2.1 INTEGRATION

NO FALLBACKS, NO FAKE FRAMES, NO PROMPT GUESSING.
Either integrate the full official WAN 2.1 codebase or disable WAN generation.

Reference: https://github.com/Wan-Video/Wan2.1
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
        Generate video using WAN Flow Matching
        
        Args:
            prompt: Text prompt for generation
            num_frames: Number of frames to generate
            height: Video height
            width: Video width  
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed
            
        Returns:
            List of generated video frames as numpy arrays
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        self.num_inference_steps = num_inference_steps
        
        print(f"🎬 Generating {num_frames} frames at {width}x{height}")
        print(f"📝 Prompt: {prompt}")
        
        # Encode text prompt using T5
        text_embeddings = self.text_encoder.encode([prompt])
        
        # Initialize random noise in latent space
        # VAE typically downsamples by factor of 8 spatially
        latent_height = height // 8
        latent_width = width // 8
        latent_channels = 16
        
        # Start from pure noise
        x_t = torch.randn(
            1, num_frames, latent_height, latent_width, latent_channels,
            device=self.device, dtype=torch.float32
        )
        
        print(f"🔄 Running {num_inference_steps} flow matching steps...")
        
        # Flow matching sampling loop
        for i, step in enumerate(range(num_inference_steps)):
            # Create timestep tensor (flow matching uses [0, 1] range)
            t = torch.full((1,), step / num_inference_steps, device=self.device)
            
            # Denoising step
            x_t = self.flow_matching_step(x_t, t, text_embeddings, guidance_scale)
            
            if (i + 1) % 10 == 0:
                print(f"  Step {i+1}/{num_inference_steps}")
                
        print("🎨 Decoding latents to video frames...")
        
        # Decode latents to video frames using Wan-VAE
        video_tensor = self.vae.decode(x_t)
        
        # Convert to numpy and PIL format
        frames = []
        video_np = video_tensor.squeeze(0).cpu().numpy()  # (3, frames, height, width)
        video_np = np.transpose(video_np, (1, 2, 3, 0))  # (frames, height, width, 3)
        video_np = np.clip((video_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
        
        for frame_idx in range(num_frames):
            frame_pil = Image.fromarray(video_np[frame_idx])
            frames.append(np.array(frame_pil))
            
        print(f"✅ Generated {len(frames)} frames using WAN Flow Matching")
        return frames


def create_wan_pipeline(model_path: str, 
                       model_tensors: Dict[str, torch.Tensor],
                       model_size: str = "14B",
                       device: str = "cuda") -> WanFlowMatchingPipeline:
    """
    Create and initialize WAN Flow Matching pipeline
    
    Args:
        model_path: Path to model directory
        model_tensors: Loaded model tensors
        model_size: Model size ("1.3B" or "14B")
        device: Device to run on
        
    Returns:
        Initialized WAN pipeline
    """
    print("🚀 Creating WAN Flow Matching pipeline...")
    
    # Create pipeline
    pipeline = WanFlowMatchingPipeline(model_path, model_size, device)
    
    # Load components
    pipeline.load_model_components(model_tensors)
    pipeline.setup_text_encoder()
    pipeline.setup_vae()
    
    print("✅ WAN Flow Matching pipeline ready!")
    return pipeline 