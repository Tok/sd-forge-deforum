# -*- coding: utf-8 -*-
"""
Wan Flash Attention Monkey Patch
=================================

This module provides a monkey patch for the Wan flash attention implementation
to add PyTorch fallback support when flash attention is not available.

This ensures the original Wan repository remains untouched while providing
compatibility for systems without flash attention.
"""

import torch
import warnings
import sys


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
    """
    Patched flash attention function with PyTorch fallback support.
    
    This replaces the original flash_attention function to provide compatibility
    when flash attention libraries are not available.
    """
    # DEBUG: Uncomment next line to verify our function is being called
    # print("🔧 PATCHED flash_attention function called!")
    
    # Import flash attention modules dynamically
    try:
        import flash_attn_interface
        FLASH_ATTN_3_AVAILABLE = True
    except ModuleNotFoundError:
        FLASH_ATTN_3_AVAILABLE = False

    try:
        import flash_attn
        FLASH_ATTN_2_AVAILABLE = True
    except ModuleNotFoundError:
        FLASH_ATTN_2_AVAILABLE = False

    # Check global mode setting
    global _FLASH_ATTENTION_MODE
    mode_setting = _FLASH_ATTENTION_MODE
    
    # Override availability based on mode setting
    if mode_setting == "Force PyTorch Fallback":
        FLASH_ATTN_2_AVAILABLE = False
        FLASH_ATTN_3_AVAILABLE = False
    elif mode_setting == "Force Flash Attention":
        if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):
            raise RuntimeError("Flash Attention forced but not available!")
    # For Auto mode, use actual availability (checked above)

    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif FLASH_ATTN_2_AVAILABLE:
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))
    else:
        # PyTorch fallback when flash attention is not available
        warnings.warn(
            'Flash attention is not available, falling back to PyTorch scaled_dot_product_attention. '
            'Performance may be reduced but functionality will be preserved.'
        )
        
        # The input tensors are already flattened: q=[B*Lq, Nq, C], k=[B*Lk, Nk, C], v=[B*Lk, Nk, C2]
        # We need to reconstruct the batch structure for PyTorch attention
        
        # Get dimensions
        total_q_len, num_heads_q, head_dim = q.shape
        total_k_len, num_heads_k, _ = k.shape
        v_head_dim = v.shape[2]  # Get the head dimension from v tensor
        
        # Unflatten to batch structure for PyTorch attention: [B, Lq/Lk, Nq/Nk, C]
        q_unflat = q.view(b, lq, num_heads_q, head_dim)
        k_unflat = k.view(b, lk, num_heads_k, head_dim)  
        v_unflat = v.view(b, lk, num_heads_k, v_head_dim)
        
        # Transpose to PyTorch format: [B, Nq/Nk, Lq/Lk, C]
        q_pt = q_unflat.transpose(1, 2)
        k_pt = k_unflat.transpose(1, 2)
        v_pt = v_unflat.transpose(1, 2)
        
        # Apply PyTorch attention
        x_pt = torch.nn.functional.scaled_dot_product_attention(
            q_pt, k_pt, v_pt, 
            attn_mask=None, 
            dropout_p=dropout_p, 
            is_causal=causal,
            scale=softmax_scale
        )
        
        # Transpose back to [B, Lq, Nq, C] format to match flash attention output
        x_unflat = x_pt.transpose(1, 2).contiguous()  # [B, Lq, Nq, C]
        
        # Flash attention returns unflattened format [B, Lq, Nq, C] - match this exactly
        x = x_unflat   # [B, Lq, Nq, C_out]

    # output
    return x.type(out_dtype)


def apply_flash_attention_patch():
    """
    Apply the flash attention monkey patch to the Wan modules.
    
    This function dynamically patches the flash_attention function in the Wan
    attention module to provide PyTorch fallback support.
    
    Returns:
        bool: True if patch was applied successfully, False otherwise
    """
    try:
        # Try to import the Wan attention module
        if 'wan.modules.attention' in sys.modules:
            wan_attention = sys.modules['wan.modules.attention']
        else:
            # Try to find and import the module
            try:
                import wan.modules.attention as wan_attention
            except ImportError:
                print("⚠️ Wan attention module not found - skipping flash attention patch")
                return False
        
        # Check if the module has the flash_attention function
        if hasattr(wan_attention, 'flash_attention'):
            # Store the original function for reference
            if not hasattr(wan_attention, '_original_flash_attention'):
                wan_attention._original_flash_attention = wan_attention.flash_attention
            
            # CRITICAL: Also patch the module-level variables to allow fallback
            # This prevents the assertion error
            original_flash_attn_2 = getattr(wan_attention, 'FLASH_ATTN_2_AVAILABLE', False)
            original_flash_attn_3 = getattr(wan_attention, 'FLASH_ATTN_3_AVAILABLE', False)
            
            # Store originals
            wan_attention._original_FLASH_ATTN_2_AVAILABLE = original_flash_attn_2
            wan_attention._original_FLASH_ATTN_3_AVAILABLE = original_flash_attn_3
            
            # CRITICAL: Also ensure flash_attn is available in module namespace
            # If flash_attn is not defined, create a dummy module to prevent NameError
            if not hasattr(wan_attention, 'flash_attn') or wan_attention.flash_attn is None:
                try:
                    import flash_attn
                    wan_attention.flash_attn = flash_attn
                except ImportError:
                    # Create a dummy flash_attn module to prevent NameError
                    class DummyFlashAttn:
                        @staticmethod
                        def flash_attn_varlen_func(*args, **kwargs):
                            print("⚠️ WARNING: Original flash_attn called instead of patched function!")
                            # Fall back to our patched function logic
                            # This should not happen, but provides safety
                            return patched_flash_attention(*args, **kwargs).unflatten(0, args[0].shape[:2])
                    
                    wan_attention.flash_attn = DummyFlashAttn()
                    print("🔧 Added dummy flash_attn to module namespace to prevent NameError")
            
            # Same for flash_attn_interface
            if not hasattr(wan_attention, 'flash_attn_interface') or wan_attention.flash_attn_interface is None:
                try:
                    import flash_attn_interface
                    wan_attention.flash_attn_interface = flash_attn_interface
                except ImportError:
                    # Create a dummy flash_attn_interface module
                    class DummyFlashAttnInterface:
                        @staticmethod
                        def flash_attn_varlen_func(*args, **kwargs):
                            raise RuntimeError("Flash Attention 3 not available - this should not be called with patched function")
                    
                    wan_attention.flash_attn_interface = DummyFlashAttnInterface()
            
            # Override based on mode
            global _FLASH_ATTENTION_MODE
            if _FLASH_ATTENTION_MODE == "Force PyTorch Fallback":
                # Force use of PyTorch fallback
                wan_attention.FLASH_ATTN_2_AVAILABLE = False
                wan_attention.FLASH_ATTN_3_AVAILABLE = False
                print("🔧 Forced Flash Attention to False for PyTorch fallback")
            elif _FLASH_ATTENTION_MODE == "Force Flash Attention":
                # Keep original values but will fail if flash attention not available
                wan_attention.FLASH_ATTN_2_AVAILABLE = original_flash_attn_2
                wan_attention.FLASH_ATTN_3_AVAILABLE = original_flash_attn_3
                if not (original_flash_attn_2 or original_flash_attn_3):
                    print("⚠️ Force Flash Attention requested but Flash Attention not available!")
            else:  # Auto mode
                # Set FLASH_ATTN_2_AVAILABLE = True to bypass assertion, our patched function handles fallback
                wan_attention.FLASH_ATTN_2_AVAILABLE = True
                wan_attention.FLASH_ATTN_3_AVAILABLE = original_flash_attn_3
                print("🔧 Auto mode: Set FLASH_ATTN_2_AVAILABLE = True for assertion bypass")
            
            # Apply the patch - replace the function EVERYWHERE
            wan_attention.flash_attention = patched_flash_attention
            
            # Also patch any cached references if they exist
            if hasattr(wan_attention, '_flash_attention'):
                wan_attention._flash_attention = patched_flash_attention
                
            # Replace in globals() if needed  
            if 'flash_attention' in wan_attention.__dict__:
                wan_attention.__dict__['flash_attention'] = patched_flash_attention
                
            print("🔧 Patched flash_attention function comprehensively")
            
            # CRITICAL: Also patch the model.py module which imports flash_attention directly
            # This is the key fix - model.py has "from .attention import flash_attention"
            try:
                if 'wan.modules.model' in sys.modules:
                    wan_model = sys.modules['wan.modules.model']
                    if hasattr(wan_model, 'flash_attention'):
                        wan_model._original_flash_attention = wan_model.flash_attention
                        wan_model.flash_attention = patched_flash_attention
                        print("🔧 Also patched flash_attention reference in wan.modules.model")
                else:
                    # Try to import and patch
                    try:
                        import wan.modules.model as wan_model
                        if hasattr(wan_model, 'flash_attention'):
                            wan_model._original_flash_attention = wan_model.flash_attention
                            wan_model.flash_attention = patched_flash_attention
                            print("🔧 Also patched flash_attention reference in wan.modules.model")
                    except ImportError:
                        print("⚠️ Could not import wan.modules.model for patching")
            except Exception as e:
                print(f"⚠️ Could not patch wan.modules.model: {e}")
            
            # ALSO patch the attention function if it exists to ensure it uses our patched version
            if hasattr(wan_attention, 'attention'):
                if not hasattr(wan_attention, '_original_attention'):
                    wan_attention._original_attention = wan_attention.attention
                
                def patched_attention(
                    q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None,
                    q_scale=None, causal=False, window_size=(-1, -1), deterministic=False,
                    dtype=torch.bfloat16, fa_version=None,
                ):
                    """Patched attention that uses our patched flash_attention"""
                    # Always use our patched flash_attention function
                    return patched_flash_attention(
                        q=q, k=k, v=v, q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
                        softmax_scale=softmax_scale, q_scale=q_scale, causal=causal,
                        window_size=window_size, deterministic=deterministic, dtype=dtype,
                        version=fa_version,
                    )
                
                wan_attention.attention = patched_attention
                print("   ✅ Also patched attention function")
            
            print("✅ Applied flash attention monkey patch successfully")
            print(f"   📊 FLASH_ATTN_2_AVAILABLE: {wan_attention.FLASH_ATTN_2_AVAILABLE}")
            print(f"   📊 FLASH_ATTN_3_AVAILABLE: {wan_attention.FLASH_ATTN_3_AVAILABLE}")
            print(f"   🔧 Mode: {_FLASH_ATTENTION_MODE}")
            return True
        else:
            print("⚠️ flash_attention function not found in Wan attention module")
            return False
            
    except Exception as e:
        print(f"❌ Failed to apply flash attention patch: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_flash_attention_availability():
    """
    Check which flash attention implementations are available.
    
    Returns:
        dict: Status of different flash attention implementations
    """
    status = {
        'flash_attn_2': False,
        'flash_attn_3': False,
        'pytorch_fallback': True
    }
    
    try:
        import flash_attn
        status['flash_attn_2'] = True
    except ImportError:
        pass
    
    try:
        import flash_attn_interface
        status['flash_attn_3'] = True
    except ImportError:
        pass
    
    return status


def get_flash_attention_status_html():
    """
    Get HTML formatted flash attention status for UI display.
    
    Returns:
        str: HTML formatted status string
    """
    status = check_flash_attention_availability()
    
    if status['flash_attn_3'] or status['flash_attn_2']:
        if status['flash_attn_3']:
            return "⚡ <span style='color: #4CAF50;'>Flash Attention 3 Available</span>"
        else:
            return "⚡ <span style='color: #2196F3;'>Flash Attention 2 Available</span>"
    else:
        return "🔄 <span style='color: #FF9800;'>PyTorch Fallback Only</span>"


def update_patched_flash_attention_mode(mode="Auto (Recommended)"):
    """
    Update the flash attention mode for the patched function.
    
    Args:
        mode (str): One of "Auto (Recommended)", "Force Flash Attention", "Force PyTorch Fallback"
    """
    global _FLASH_ATTENTION_MODE
    _FLASH_ATTENTION_MODE = mode
    print(f"🔧 Flash Attention mode set to: {mode}")


# Global variable to track flash attention mode
_FLASH_ATTENTION_MODE = "Auto (Recommended)"


if __name__ == "__main__":
    # Test the patch application
    print("🔍 Flash Attention Availability:")
    status = check_flash_attention_availability()
    for impl, available in status.items():
        print(f"   {impl}: {'✅' if available else '❌'}")
    
    print(f"\n🔧 Testing monkey patch (won't apply without Wan module loaded)...")
    success = apply_flash_attention_patch()
    if success:
        print("✅ Patch applied successfully!")
    else:
        print("❌ Patch could not be applied (normal when Wan not loaded)") 