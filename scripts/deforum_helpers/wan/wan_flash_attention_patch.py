#!/usr/bin/env python3
"""
Flash Attention Patch for Wan
Attempts to enable Flash Attention with proper error handling and fallback to PyTorch native attention
"""

import torch
import warnings
import sys
import os
import importlib.util

# Global flags to track patching status
_PATCH_APPLIED = False
_FLASH_ATTENTION_AVAILABLE = None
_FLASH_ATTENTION_ERROR = None

def check_flash_attention_availability():
    """Check if Flash Attention is actually available and working"""
    global _FLASH_ATTENTION_AVAILABLE, _FLASH_ATTENTION_ERROR
    
    if _FLASH_ATTENTION_AVAILABLE is not None:
        return _FLASH_ATTENTION_AVAILABLE
    
    try:
        # Try to import flash_attn
        import flash_attn
        from flash_attn import flash_attn_varlen_func
        
        # Test with a small tensor to see if it actually works
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if device.type != 'cuda':
            _FLASH_ATTENTION_ERROR = "Flash Attention requires CUDA"
            _FLASH_ATTENTION_AVAILABLE = False
            return False
        
        # Try a simple flash attention call
        try:
            test_q = torch.randn(1, 32, 64, dtype=torch.float16, device=device)
            test_k = torch.randn(1, 32, 64, dtype=torch.float16, device=device)
            test_v = torch.randn(1, 32, 64, dtype=torch.float16, device=device)
            
            # Try calling flash attention
            _ = torch.nn.functional.scaled_dot_product_attention(test_q, test_k, test_v)
            
            # If that works, try flash_attn_varlen_func with minimal setup
            q_flattened = test_q.view(-1, 64)
            k_flattened = test_k.view(-1, 64) 
            v_flattened = test_v.view(-1, 64)
            cu_seqlens = torch.tensor([0, 32], dtype=torch.int32, device=device)
            
            _ = flash_attn_varlen_func(
                q_flattened, k_flattened, v_flattened,
                cu_seqlens, cu_seqlens,
                max_seqlen_q=32, max_seqlen_k=32,
                dropout_p=0.0,
                causal=False
            )
            
            _FLASH_ATTENTION_AVAILABLE = True
            _FLASH_ATTENTION_ERROR = None
            print("✅ Flash Attention test passed - Flash Attention is available")
            return True
            
        except Exception as test_e:
            _FLASH_ATTENTION_ERROR = f"Flash Attention test failed: {test_e}"
            _FLASH_ATTENTION_AVAILABLE = False
            print(f"⚠️ Flash Attention test failed: {test_e}")
            return False
            
    except ImportError as e:
        _FLASH_ATTENTION_ERROR = f"Flash Attention import failed: {e}"
        _FLASH_ATTENTION_AVAILABLE = False
        print(f"⚠️ Flash Attention not available: {e}")
        return False
    except Exception as e:
        _FLASH_ATTENTION_ERROR = f"Flash Attention check failed: {e}"
        _FLASH_ATTENTION_AVAILABLE = False
        print(f"⚠️ Flash Attention availability check failed: {e}")
        return False

def _create_smart_flash_attention():
    """Create a smart flash_attention function that tries Flash Attention first, falls back to PyTorch"""
    
    def smart_flash_attention(
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
        """Smart flash_attention that tries Flash Attention first, then falls back to PyTorch"""
        
        # Check if Flash Attention is available
        if check_flash_attention_availability():
            try:
                # Try to use real Flash Attention
                from flash_attn import flash_attn_varlen_func
                
                # Prepare inputs for flash attention
                batch_size, seq_len_q, head_dim = q.shape
                seq_len_k = k.shape[1]
                
                # Convert to flat format for flash_attn_varlen_func
                q_flat = q.contiguous().view(-1, head_dim).to(dtype)
                k_flat = k.contiguous().view(-1, head_dim).to(dtype)
                v_flat = v.contiguous().view(-1, head_dim).to(dtype)
                
                # Create sequence length arrays
                if q_lens is None:
                    q_lens = [seq_len_q] * batch_size
                if k_lens is None:
                    k_lens = [seq_len_k] * batch_size
                
                # Create cumulative sequence lengths
                cu_seqlens_q = torch.cat([
                    torch.tensor([0], dtype=torch.int32, device=q.device),
                    torch.cumsum(torch.tensor(q_lens, dtype=torch.int32, device=q.device), dim=0)
                ])
                cu_seqlens_k = torch.cat([
                    torch.tensor([0], dtype=torch.int32, device=k.device),
                    torch.cumsum(torch.tensor(k_lens, dtype=torch.int32, device=k.device), dim=0)
                ])
                
                max_seqlen_q = max(q_lens)
                max_seqlen_k = max(k_lens)
                
                # Apply q_scale if provided
                if q_scale is not None:
                    q_flat = q_flat * q_scale
                
                # Call Flash Attention
                out_flat = flash_attn_varlen_func(
                    q_flat, k_flat, v_flat,
                    cu_seqlens_q, cu_seqlens_k,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    dropout_p=dropout_p,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    deterministic=deterministic
                )
                
                # Reshape back to original format
                out = out_flat.view(batch_size, seq_len_q, head_dim).contiguous()
                
                # Convert back to original dtype
                return out.to(q.dtype)
                
            except Exception as fa_error:
                # Flash Attention failed, log warning and fall back
                print(f"⚠️ Flash Attention failed, falling back to PyTorch: {fa_error}")
                # Continue to PyTorch fallback below
        
        # PyTorch native attention fallback
        if q_lens is not None or k_lens is not None:
            warnings.warn(
                '⚠️ Using PyTorch attention fallback. Padding masks disabled for performance.',
                UserWarning
            )
        
        # Convert to proper format for PyTorch attention
        q = q.transpose(-3, -2).to(dtype)  # (batch, heads, seq, dim)
        k = k.transpose(-3, -2).to(dtype)
        v = v.transpose(-3, -2).to(dtype)
        
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
        out = out.transpose(-3, -2).contiguous()
        
        # Return with original dtype
        return out.to(q.dtype)
    
    # Mark as patched
    smart_flash_attention._is_patched_by_deforum = True
    smart_flash_attention._uses_flash_attention = check_flash_attention_availability()
    return smart_flash_attention

def _create_smart_attention():
    """Create a smart attention function that tries Flash Attention first"""
    
    def smart_attention(
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
        """Smart attention that tries Flash Attention first, then falls back to PyTorch"""
        
        # Use the same logic as smart_flash_attention
        return _create_smart_flash_attention()(
            q, k, v, q_lens, k_lens, dropout_p, softmax_scale, q_scale,
            causal, window_size, deterministic, dtype, fa_version
        )
    
    # Mark as patched
    smart_attention._is_patched_by_deforum = True
    smart_attention._uses_flash_attention = check_flash_attention_availability()
    return smart_attention

def force_disable_flash_attention():
    """Force disable flash attention flags as a last resort"""
    try:
        # Mock flash_attn modules if they're not available
        if 'flash_attn' not in sys.modules:
            import types
            mock_flash_attn = types.ModuleType('flash_attn')
            
            def mock_flash_attn_varlen_func(*args, **kwargs):
                raise RuntimeError("Flash Attention not available - use PyTorch native attention")
            
            mock_flash_attn.flash_attn_varlen_func = mock_flash_attn_varlen_func
            sys.modules['flash_attn'] = mock_flash_attn
        
        # Set flags to False in attention module if loaded
        if 'wan.modules.attention' in sys.path:
            attention_module = sys.modules['wan.modules.attention']
            attention_module.FLASH_ATTN_2_AVAILABLE = False
            attention_module.FLASH_ATTN_3_AVAILABLE = False
            print("🔧 Set Flash Attention flags to False in loaded module")
            
    except Exception as e:
        print(f"⚠️ Could not modify flash attention flags: {e}")

def _patch_wan_flash_attention_internal():
    """Internal patching function with smart Flash Attention support"""
    global _PATCH_APPLIED
    
    if _PATCH_APPLIED:
        print("✅ Flash Attention patch already applied")
        return True
    
    try:
        print("🔧 Attempting Flash Attention patching with smart fallback...")
        
        # Check Flash Attention availability first
        flash_available = check_flash_attention_availability()
        
        if flash_available:
            print("⚡ Flash Attention is available - will use Flash Attention with PyTorch fallback")
        else:
            print(f"⚠️ Flash Attention not available ({_FLASH_ATTENTION_ERROR}) - will use PyTorch attention")
        
        # Try to find and add Wan2.1 path
        wan_dir = None
        
        # Method 1: Use the path from flash attention patch file location
        current_dir = os.path.dirname(os.path.abspath(__file__))
        extension_dir = os.path.dirname(os.path.dirname(current_dir))
        potential_wan_dir = os.path.join(extension_dir, 'Wan2.1')
        
        if os.path.exists(potential_wan_dir) and os.path.exists(os.path.join(potential_wan_dir, 'wan')):
            wan_dir = potential_wan_dir
            
        # Method 2: Check if Wan2.1 is already in sys.path
        if not wan_dir:
            for path in sys.path:
                if 'Wan2.1' in path and os.path.exists(os.path.join(path, 'wan')):
                    wan_dir = path
                    break
        
        # Method 3: Search common locations
        if not wan_dir:
            search_paths = [
                os.path.join(extension_dir, 'Wan2.1'),
                os.path.join(os.path.dirname(extension_dir), 'Wan2.1'),
                './Wan2.1',
                '../Wan2.1'
            ]
            for search_path in search_paths:
                abs_path = os.path.abspath(search_path)
                if os.path.exists(abs_path) and os.path.exists(os.path.join(abs_path, 'wan')):
                    wan_dir = abs_path
                    break
        
        # Add Wan2.1 to Python path if found
        if wan_dir and wan_dir not in sys.path:
            sys.path.insert(0, wan_dir)
            print(f"✅ Added Wan repo to path for Flash Attention patch: {wan_dir}")
        elif wan_dir:
            print(f"✅ Wan repo already in path: {wan_dir}")
        else:
            print("⚠️ Wan repository not found - proceeding without repo-specific patches")
        
        # Try to import and patch the attention module
        try:
            import wan.modules.attention as attention_module
            print("✅ Successfully imported Wan attention module")
            
            # Apply smart patching - try Flash Attention first, fall back to PyTorch
            if hasattr(attention_module, 'flash_attention'):
                attention_module.flash_attention = _create_smart_flash_attention()
                print("✅ Patched flash_attention with smart Flash Attention support")
            
            if hasattr(attention_module, 'attention'):
                attention_module.attention = _create_smart_attention()
                print("✅ Patched attention with smart Flash Attention support")
            
            # Set availability flags based on actual Flash Attention availability
            if hasattr(attention_module, 'FLASH_ATTN_2_AVAILABLE'):
                attention_module.FLASH_ATTN_2_AVAILABLE = flash_available
            if hasattr(attention_module, 'FLASH_ATTN_3_AVAILABLE'):
                attention_module.FLASH_ATTN_3_AVAILABLE = flash_available
                
            if flash_available:
                print("⚡ Flash Attention flags set to True - will attempt Flash Attention")
            else:
                print("🔧 Flash Attention flags set to False - will use PyTorch attention")
            
        except ImportError:
            print("⚠️ Wan attention module not found - will patch when imported")
        except Exception as e:
            print(f"⚠️ Error patching attention module: {e}")
        
        _PATCH_APPLIED = True
        
        if flash_available:
            print("✅ Smart Flash Attention patch applied successfully! (Flash Attention enabled)")
        else:
            print("✅ Smart Flash Attention patch applied successfully! (PyTorch fallback mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ Flash Attention patching failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def ensure_flash_attention_patched():
    """Ensure Flash Attention compatibility is patched"""
    return _patch_wan_flash_attention_internal()

def apply_wan_flash_attention_when_imported():
    """Apply Flash Attention patches after Wan module is imported"""
    try:
        if 'wan.modules.attention' in sys.modules:
            print("🔧 Applying delayed Flash Attention patches to imported Wan module...")
            attention_module = sys.modules['wan.modules.attention']
            
            # Check Flash Attention availability
            flash_available = check_flash_attention_availability()
            
            # Apply smart patching
            if hasattr(attention_module, 'flash_attention'):
                attention_module.flash_attention = _create_smart_flash_attention()
                print("✅ Patched flash_attention with smart support (delayed)")
            
            if hasattr(attention_module, 'attention'):
                attention_module.attention = _create_smart_attention()
                print("✅ Patched attention with smart support (delayed)")
            
            # Set flags based on availability
            if hasattr(attention_module, 'FLASH_ATTN_2_AVAILABLE'):
                attention_module.FLASH_ATTN_2_AVAILABLE = flash_available
            if hasattr(attention_module, 'FLASH_ATTN_3_AVAILABLE'):
                attention_module.FLASH_ATTN_3_AVAILABLE = flash_available
            
            if flash_available:
                print("⚡ Delayed Flash Attention patches applied successfully! (Flash Attention enabled)")
            else:
                print("🔧 Delayed Flash Attention patches applied successfully! (PyTorch fallback)")
                
        else:
            print("⚠️ Wan attention module not yet imported for delayed patching")
            
    except Exception as e:
        print(f"⚠️ Error applying delayed Flash Attention patches: {e}")

def apply_wan_compatibility_patches():
    """Apply all Wan compatibility patches"""
    try:
        print("🔧 Applying Wan compatibility patches...")
        
        # Apply Flash Attention patches
        _patch_wan_flash_attention_internal()
        
        # Also try delayed patching if module is already loaded
        apply_wan_flash_attention_when_imported()
        
        print("✅ Flash Attention compatibility patch completed!")
        
    except Exception as e:
        print(f"❌ Flash Attention compatibility patching failed: {e}")
        # Don't raise - continue with other patches

# Auto-apply patches when module is imported
if __name__ != "__main__":
    try:
        ensure_flash_attention_patched()
    except Exception:
        pass  # Silent fail on import
