#!/usr/bin/env python3
"""
WAN Flash Attention Compatibility Layer
Handles flash attention compatibility without modifying the original WAN repository
"""

import torch
import warnings
import sys
import os
from pathlib import Path

class WanFlashAttentionCompatibility:
    """Compatibility layer to handle flash attention issues in WAN without modifying the original repo"""
    
    def __init__(self):
        self.flash_attn_2_available = self._check_flash_attn_2()
        self.flash_attn_3_available = self._check_flash_attn_3()
        self.patched = False
    
    def _check_flash_attn_2(self):
        """Check if flash attention 2 is available"""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _check_flash_attn_3(self):
        """Check if flash attention 3 is available"""
        try:
            import flash_attn_interface
            return True
        except ImportError:
            return False
    
    def patch_wan_attention(self):
        """Monkey patch WAN attention to handle missing flash attention gracefully"""
        if self.patched:
            return
        
        print("🔧 Applying WAN flash attention compatibility patch...")
        
        try:
            # Import the WAN attention module
            wan_repo_path = Path(__file__).parent.parent.parent / "Wan2.1"
            if str(wan_repo_path) not in sys.path:
                sys.path.insert(0, str(wan_repo_path))
            
            # Import wan attention module
            import wan.modules.attention as wan_attention
            
            # Store original functions
            original_flash_attention = wan_attention.flash_attention
            original_attention = wan_attention.attention
            
            # Create patched flash_attention function
            def patched_flash_attention(*args, **kwargs):
                """Patched flash_attention that gracefully falls back to standard attention"""
                if not self.flash_attn_2_available and not self.flash_attn_3_available:
                    print("⚠️ Flash attention not available, falling back to standard attention")
                    return original_attention(*args, **kwargs)
                return original_flash_attention(*args, **kwargs)
            
            # Create patched attention function that never fails
            def patched_attention(
                q, k, v,
                q_lens=None, k_lens=None,
                dropout_p=0., softmax_scale=None, q_scale=None,
                causal=False, window_size=(-1, -1),
                deterministic=False, dtype=torch.bfloat16,
                fa_version=None,
            ):
                """Patched attention function with robust fallback"""
                
                # Try flash attention first if available
                if self.flash_attn_2_available or self.flash_attn_3_available:
                    try:
                        return original_flash_attention(
                            q=q, k=k, v=v,
                            q_lens=q_lens, k_lens=k_lens,
                            dropout_p=dropout_p, softmax_scale=softmax_scale,
                            q_scale=q_scale, causal=causal,
                            window_size=window_size, deterministic=deterministic,
                            dtype=dtype, version=fa_version,
                        )
                    except Exception as e:
                        print(f"⚠️ Flash attention failed: {e}, falling back to standard attention")
                
                # Fallback to standard PyTorch attention
                if q_lens is not None or k_lens is not None:
                    warnings.warn(
                        'Padding mask is disabled when using scaled_dot_product_attention. '
                        'It can have a significant impact on performance.'
                    )
                
                attn_mask = None
                
                # Reshape for PyTorch attention: (B, L, N, D) -> (B, N, L, D)
                q = q.transpose(1, 2).to(dtype)
                k = k.transpose(1, 2).to(dtype) 
                v = v.transpose(1, 2).to(dtype)
                
                # Apply q_scale if provided
                if q_scale is not None:
                    q = q * q_scale
                
                # Use PyTorch's native attention
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=attn_mask, 
                    is_causal=causal, 
                    dropout_p=dropout_p
                )
                
                # Reshape back: (B, N, L, D) -> (B, L, N, D)
                out = out.transpose(1, 2).contiguous()
                return out
            
            # Apply patches
            wan_attention.flash_attention = patched_flash_attention
            wan_attention.attention = patched_attention
            
            # Update availability flags in the module
            wan_attention.FLASH_ATTN_2_AVAILABLE = self.flash_attn_2_available
            wan_attention.FLASH_ATTN_3_AVAILABLE = self.flash_attn_3_available
            
            self.patched = True
            print("✅ WAN flash attention compatibility patch applied successfully")
            
            if not self.flash_attn_2_available and not self.flash_attn_3_available:
                print("ℹ️ Flash attention not available - using PyTorch native attention fallback")
            
        except Exception as e:
            print(f"❌ Failed to patch WAN attention: {e}")
            raise RuntimeError(f"WAN compatibility patch failed: {e}")
    
    def ensure_wan_compatibility(self):
        """Ensure WAN is compatible before importing WAN modules"""
        if not self.patched:
            self.patch_wan_attention()
        
        # Additional compatibility checks
        self._check_pytorch_version()
        self._check_cuda_availability()
    
    def _check_pytorch_version(self):
        """Check PyTorch version compatibility"""
        torch_version = torch.__version__
        print(f"🔍 PyTorch version: {torch_version}")
        
        # WAN requires PyTorch >= 1.13 for scaled_dot_product_attention
        if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            raise RuntimeError(
                f"PyTorch {torch_version} does not support scaled_dot_product_attention. "
                "Please upgrade to PyTorch >= 1.13"
            )
    
    def _check_cuda_availability(self):
        """Check CUDA availability"""
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name()}")
        else:
            print("⚠️ CUDA not available - WAN will run on CPU (very slow)")


# Global compatibility instance
_wan_compatibility = WanFlashAttentionCompatibility()

def ensure_wan_compatibility():
    """Public function to ensure WAN compatibility"""
    return _wan_compatibility.ensure_wan_compatibility()

def get_flash_attention_status():
    """Get flash attention availability status"""
    return {
        'flash_attn_2': _wan_compatibility.flash_attn_2_available,
        'flash_attn_3': _wan_compatibility.flash_attn_3_available,
        'patched': _wan_compatibility.patched
    }

if __name__ == "__main__":
    print("🧪 Testing WAN Flash Attention Compatibility...")
    
    print(f"Flash Attention 2: {_wan_compatibility.flash_attn_2_available}")
    print(f"Flash Attention 3: {_wan_compatibility.flash_attn_3_available}")
    
    try:
        ensure_wan_compatibility()
        print("✅ WAN compatibility ensured")
        
        # Test importing WAN
        import wan
        print("✅ WAN import successful")
        
    except Exception as e:
        print(f"❌ Compatibility test failed: {e}")
