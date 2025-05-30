#!/usr/bin/env python3
"""
Flash Attention Patch for Wan
Patches the Wan attention module to use PyTorch native attention when Flash Attention is not available
"""

import torch
import warnings
import sys
import os
import importlib.util

# Global flag to track if patching has been applied
_PATCH_APPLIED = False

def force_disable_flash_attention():
    """Force disable flash attention by setting the flags to False in the attention module"""
    try:
        # First, ensure flash_attn modules are mocked
        if 'flash_attn' not in sys.modules:
            import types
            mock_flash_attn = types.ModuleType('flash_attn')
            
            def mock_flash_attn_varlen_func(*args, **kwargs):
                raise RuntimeError("Flash Attention not available - use PyTorch native attention")
            
            mock_flash_attn.flash_attn_varlen_func = mock_flash_attn_varlen_func
            sys.modules['flash_attn'] = mock_flash_attn
        
        if 'flash_attn_interface' not in sys.modules:
            import types
            mock_interface = types.ModuleType('flash_attn_interface')
            
            def mock_flash_attn_interface_func(*args, **kwargs):
                raise RuntimeError("Flash Attention interface not available - use PyTorch native attention")
            
            mock_interface.flash_attn_varlen_func = mock_flash_attn_interface_func
            sys.modules['flash_attn_interface'] = mock_interface
        
        # Now check if wan.modules.attention is already imported
        if 'wan.modules.attention' in sys.modules:
            attention_module = sys.modules['wan.modules.attention']
            attention_module.FLASH_ATTN_2_AVAILABLE = False
            attention_module.FLASH_ATTN_3_AVAILABLE = False
            print("✅ Disabled Flash Attention flags in already-loaded module")
    except Exception as e:
        print(f"⚠️ Could not disable flash attention flags: {e}")

def _create_patched_flash_attention():
    """Create the patched flash_attention function"""
    def patched_flash_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        version=None,
    ):
        """Patched flash_attention that uses PyTorch native attention"""
        
        # Use PyTorch native attention
        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        assert q.device.type == 'cuda' and q.size(-1) <= 256

        # params
        b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        # Simplified preprocessing for PyTorch attention
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        
        # Convert to proper format for PyTorch attention
        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype) 
        v = v.transpose(1, 2).to(dtype)
        
        if q_scale is not None:
            q = q * q_scale

        # Apply PyTorch native attention
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            is_causal=causal, 
            dropout_p=dropout_p,
            scale=softmax_scale
        )

        # Transpose back and ensure correct output format
        out = out.transpose(1, 2).contiguous()
        
        # Return with original dtype
        return out.type(out_dtype)
    
    # Mark as patched
    patched_flash_attention._is_patched_by_deforum = True
    return patched_flash_attention

def _create_patched_attention():
    """Create the patched attention function"""
    def patched_attention(
        q,
        k,
        v,
        q_lens=None,
        k_lens=None,
        dropout_p=0.,
        softmax_scale=None,
        q_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
        dtype=torch.bfloat16,
        fa_version=None,
    ):
        """Patched attention that always uses PyTorch native attention"""
        
        # Always use PyTorch fallback
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
            )
        attn_mask = None

        q = q.transpose(1, 2).to(dtype)
        k = k.transpose(1, 2).to(dtype)
        v = v.transpose(1, 2).to(dtype)

        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p, scale=softmax_scale)

        out = out.transpose(1, 2).contiguous()
        return out
    
    # Mark as patched
    patched_attention._is_patched_by_deforum = True
    return patched_attention

def _patch_wan_flash_attention_internal():
    """Internal patching function"""
    global _PATCH_APPLIED
    
    if _PATCH_APPLIED:
        print("✅ Flash Attention patch already applied")
        return True
    
    try:
        # Get the correct path to the Wan2.1 directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        extension_dir = os.path.dirname(os.path.dirname(current_dir))
        wan_dir = os.path.join(extension_dir, 'Wan2.1')
        
        # Add Wan2.1 to Python path if not already there
        if wan_dir not in sys.path:
            sys.path.insert(0, wan_dir)
        
        print("🔧 Attempting comprehensive Flash Attention patching...")
        
        # Force disable flash attention first
        force_disable_flash_attention()
        
        # Import wan.modules.attention
        from wan.modules import attention as wan_attention
        print("✅ Successfully imported Wan attention module")
        
        # Set flags to False
        wan_attention.FLASH_ATTN_2_AVAILABLE = False
        wan_attention.FLASH_ATTN_3_AVAILABLE = False
        print("✅ Disabled Flash Attention flags")
        
        # Create patched functions
        patched_flash_attention = _create_patched_flash_attention()
        patched_attention = _create_patched_attention()
        
        # Apply patches
        wan_attention.flash_attention = patched_flash_attention
        wan_attention.attention = patched_attention
        
        # Also update in sys.modules
        if 'wan.modules.attention' in sys.modules:
            sys.modules['wan.modules.attention'].flash_attention = patched_flash_attention
            sys.modules['wan.modules.attention'].attention = patched_attention
            sys.modules['wan.modules.attention'].FLASH_ATTN_2_AVAILABLE = False
            sys.modules['wan.modules.attention'].FLASH_ATTN_3_AVAILABLE = False
        
        # Patch the model module if it's already imported
        if 'wan.modules.model' in sys.modules:
            model_module = sys.modules['wan.modules.model']
            # Replace the imported flash_attention in model.py
            model_module.flash_attention = patched_flash_attention
            print("✅ Patched flash_attention in model module")
        
        _PATCH_APPLIED = True
        print("✅ Comprehensive Wan Flash Attention patch applied successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Comprehensive Flash Attention patch FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def ensure_flash_attention_patched():
    """Ensure flash attention is patched before any Wan model usage"""
    global _PATCH_APPLIED
    
    if not _PATCH_APPLIED:
        _patch_wan_flash_attention_internal()
    
    # Double-check the patch is still active
    try:
        if 'wan.modules.attention' in sys.modules:
            attention_module = sys.modules['wan.modules.attention']
            if attention_module.FLASH_ATTN_2_AVAILABLE or attention_module.FLASH_ATTN_3_AVAILABLE:
                print("⚠️ Flash Attention flags were re-enabled, re-applying patch...")
                attention_module.FLASH_ATTN_2_AVAILABLE = False
                attention_module.FLASH_ATTN_3_AVAILABLE = False
                
                # Re-apply function patches
                patched_flash_attention = _create_patched_flash_attention()
                patched_attention = _create_patched_attention()
                attention_module.flash_attention = patched_flash_attention
                attention_module.attention = patched_attention
                
                # Also patch model module if needed
                if 'wan.modules.model' in sys.modules:
                    sys.modules['wan.modules.model'].flash_attention = patched_flash_attention
    except Exception as e:
        print(f"⚠️ Could not verify patch status: {e}")

# Apply the patch immediately when this module is imported
_patch_wan_flash_attention_internal()

def apply_wan_compatibility_patches():
    """Public function to manually apply or re-apply patches if needed"""
    print("🔧 Manually calling comprehensive compatibility patches...")
    global _PATCH_APPLIED
    _PATCH_APPLIED = False  # Force re-application
    return _patch_wan_flash_attention_internal()

if __name__ == "__main__":
    print("🔧 Running wan_flash_attention_patch.py directly for testing...")
    apply_wan_compatibility_patches()
