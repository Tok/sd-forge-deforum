"""
WAN Tensor Adapter - Fixes tensor shape mismatches between DiT/Flux models and WAN architecture
Converts existing DiT/Flux model weights to work with WAN Flow Matching pipeline
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import re
from pathlib import Path


class DiTToWanTensorAdapter:
    """
    Adapter that converts DiT/Flux model tensors to WAN-compatible format
    Handles tensor shape mismatches and naming differences
    """
    
    def __init__(self, model_size: str = "14B"):
        self.model_size = model_size
        
        # WAN model configurations
        if model_size == "14B":
            self.wan_dim = 5120
            self.wan_heads = 40
            self.wan_layers = 40
            self.wan_feedforward_dim = 13824
        else:  # 1.3B
            self.wan_dim = 1536
            self.wan_heads = 12
            self.wan_layers = 30
            self.wan_feedforward_dim = 8960
            
        # Cross-attention dimension (T5 encoder output)
        self.cross_attention_dim = 768
        
    def detect_model_type(self, tensor_names: List[str]) -> str:
        """
        Detect the actual model type from tensor names
        
        Args:
            tensor_names: List of tensor names from the model
            
        Returns:
            Model type: 'dit', 'flux', 'wan', or 'unknown'
        """
        # Count patterns to identify model type
        patterns = {
            'dit': len([name for name in tensor_names if 
                       'blocks.' in name and 'cross_attn' in name]),
            'flux': len([name for name in tensor_names if 
                        'single_blocks' in name or 'double_blocks' in name]),
            'wan': len([name for name in tensor_names if 
                       'wan' in name.lower() or 't2v' in name.lower()]),
            'vace': len([name for name in tensor_names if 
                        'vace_blocks' in name])
        }
        
        print(f"🔍 Model type detection:")
        print(f"   DiT patterns: {patterns['dit']}")
        print(f"   Flux patterns: {patterns['flux']}")
        print(f"   WAN patterns: {patterns['wan']}")
        print(f"   VACE patterns: {patterns['vace']}")
        
        # Determine model type
        if patterns['wan'] > 10:
            return 'wan'
        elif patterns['vace'] > 0:
            return 'dit_vace'  # DiT with Video Auto-encoder Conditioning Extension
        elif patterns['dit'] > patterns['flux']:
            return 'dit'
        elif patterns['flux'] > 0:
            return 'flux'
        else:
            return 'unknown'
    
    def get_tensor_dimensions(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, Tuple[int, ...]]:
        """
        Analyze tensor dimensions to understand the actual model architecture
        
        Args:
            tensors: Dictionary of model tensors
            
        Returns:
            Dictionary mapping tensor names to their shapes
        """
        dimensions = {}
        
        # Look for key tensors that indicate model dimensions
        key_patterns = [
            r'blocks\.0\.self_attn\.q\.weight',
            r'blocks\.0\.cross_attn\.q\.weight', 
            r'blocks\.0\.ffn\.0\.weight',
            r'text_embedding\.0\.weight',
            r'patch_embedding\.weight'
        ]
        
        for name, tensor in tensors.items():
            for pattern in key_patterns:
                if re.match(pattern, name):
                    dimensions[name] = tensor.shape
                    print(f"📐 Key tensor {name}: {tensor.shape}")
                    
        return dimensions
    
    def infer_actual_model_config(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Infer the actual model configuration from tensor shapes
        
        Args:
            tensors: Dictionary of model tensors
            
        Returns:
            Dictionary with inferred model configuration
        """
        dimensions = self.get_tensor_dimensions(tensors)
        
        # Try to find the model dimension from key tensors
        actual_dim = None
        actual_heads = None
        actual_layers = 0
        
        # Look for self-attention query weight to get model dimension
        for name, shape in dimensions.items():
            if 'self_attn.q.weight' in name:
                # For multi-head attention: (dim, dim) where dim = num_heads * head_dim
                actual_dim = shape[0]
                break
                
        # Count transformer blocks
        block_numbers = set()
        for name in tensors.keys():
            match = re.match(r'blocks\.(\d+)\.', name)
            if match:
                block_numbers.add(int(match.group(1)))
                
        actual_layers = len(block_numbers)
        
        # Try to infer number of heads
        if actual_dim:
            # Common head dimensions are 64, 80, 128
            for head_dim in [64, 80, 128]:
                if actual_dim % head_dim == 0:
                    actual_heads = actual_dim // head_dim
                    break
                    
        config = {
            'dim': actual_dim,
            'num_heads': actual_heads,
            'num_layers': actual_layers,
            'model_type': self.detect_model_type(list(tensors.keys()))
        }
        
        print(f"🔧 Inferred model config:")
        print(f"   Dimension: {actual_dim}")
        print(f"   Heads: {actual_heads}")
        print(f"   Layers: {actual_layers}")
        print(f"   Type: {config['model_type']}")
        
        return config
    
    def create_tensor_mapping(self, source_config: Dict[str, Any]) -> Dict[str, str]:
        """
        Create mapping from source tensor names to WAN tensor names
        
        Args:
            source_config: Configuration of the source model
            
        Returns:
            Dictionary mapping source names to WAN names
        """
        mapping = {}
        
        if source_config['model_type'] in ['dit', 'dit_vace']:
            # Map DiT tensors to WAN tensors
            for i in range(min(source_config['num_layers'], self.wan_layers)):
                # Self-attention mapping
                mapping[f'blocks.{i}.self_attn.q.weight'] = f'blocks.{i}.self_attn.q.weight'
                mapping[f'blocks.{i}.self_attn.k.weight'] = f'blocks.{i}.self_attn.k.weight'
                mapping[f'blocks.{i}.self_attn.v.weight'] = f'blocks.{i}.self_attn.v.weight'
                mapping[f'blocks.{i}.self_attn.o.weight'] = f'blocks.{i}.self_attn.o.weight'
                
                # Cross-attention mapping  
                mapping[f'blocks.{i}.cross_attn.q.weight'] = f'blocks.{i}.cross_attn.q.weight'
                mapping[f'blocks.{i}.cross_attn.k.weight'] = f'blocks.{i}.cross_attn.k.weight'
                mapping[f'blocks.{i}.cross_attn.v.weight'] = f'blocks.{i}.cross_attn.v.weight'
                mapping[f'blocks.{i}.cross_attn.o.weight'] = f'blocks.{i}.cross_attn.o.weight'
                
                # FFN mapping
                mapping[f'blocks.{i}.ffn.0.weight'] = f'blocks.{i}.ffn.0.weight'
                mapping[f'blocks.{i}.ffn.2.weight'] = f'blocks.{i}.ffn.2.weight'
                
                # Norm mapping
                mapping[f'blocks.{i}.norm3.weight'] = f'blocks.{i}.norm3.weight'
                
                # Time modulation mapping (this might need adaptation)
                mapping[f'blocks.{i}.modulation'] = f'blocks.{i}.time_bias'
                
        return mapping
    
    def adapt_tensor_shape(self, source_tensor: torch.Tensor, 
                          source_shape: Tuple[int, ...], 
                          target_shape: Tuple[int, ...],
                          tensor_name: str) -> torch.Tensor:
        """
        Adapt tensor shape from source to target dimensions
        
        Args:
            source_tensor: Original tensor
            source_shape: Original tensor shape
            target_shape: Target tensor shape  
            tensor_name: Name of the tensor for debugging
            
        Returns:
            Adapted tensor with target shape
        """
        if source_shape == target_shape:
            return source_tensor
            
        print(f"🔧 Adapting tensor {tensor_name}: {source_shape} → {target_shape}")
        
        # Handle different adaptation strategies based on tensor type
        if 'weight' in tensor_name:
            if len(source_shape) == 2 and len(target_shape) == 2:
                # Linear layer weight adaptation
                source_out, source_in = source_shape
                target_out, target_in = target_shape
                
                if source_out != target_out or source_in != target_in:
                    # Use interpolation or truncation/padding
                    adapted_tensor = torch.zeros(target_shape, dtype=source_tensor.dtype, device=source_tensor.device)
                    
                    # Copy overlapping region
                    min_out = min(source_out, target_out)
                    min_in = min(source_in, target_in)
                    adapted_tensor[:min_out, :min_in] = source_tensor[:min_out, :min_in]
                    
                    # Initialize remaining weights with small random values
                    if target_out > source_out:
                        nn.init.xavier_uniform_(adapted_tensor[source_out:, :])
                    if target_in > source_in:
                        nn.init.xavier_uniform_(adapted_tensor[:, source_in:])
                        
                    return adapted_tensor
                    
        elif 'bias' in tensor_name:
            if len(source_shape) == 1 and len(target_shape) == 1:
                # Bias adaptation
                source_dim = source_shape[0]
                target_dim = target_shape[0]
                
                if source_dim != target_dim:
                    adapted_tensor = torch.zeros(target_shape, dtype=source_tensor.dtype, device=source_tensor.device)
                    min_dim = min(source_dim, target_dim)
                    adapted_tensor[:min_dim] = source_tensor[:min_dim]
                    return adapted_tensor
                    
        elif 'modulation' in tensor_name:
            # Time modulation tensor adaptation
            if source_shape != target_shape:
                # Create new modulation tensor with correct shape
                adapted_tensor = torch.zeros(target_shape, dtype=source_tensor.dtype, device=source_tensor.device)
                
                # Initialize with small random values
                nn.init.normal_(adapted_tensor, std=0.02)
                
                return adapted_tensor
                
        # Default: return source tensor if no specific adaptation
        print(f"⚠️ No specific adaptation for {tensor_name}, using source tensor")
        return source_tensor
    
    def convert_tensors_to_wan_format(self, source_tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert source model tensors to WAN-compatible format
        
        Args:
            source_tensors: Original model tensors
            
        Returns:
            Dictionary of WAN-compatible tensors
        """
        print("🔄 Converting tensors to WAN format...")
        
        # Infer source model configuration
        source_config = self.infer_actual_model_config(source_tensors)
        
        if source_config['model_type'] == 'wan':
            print("✅ Source model is already WAN format")
            return source_tensors
            
        if source_config['model_type'] == 'unknown':
            print("⚠️ Unknown model type, attempting best-effort conversion")
            
        # Create tensor mapping
        tensor_mapping = self.create_tensor_mapping(source_config)
        
        # Convert tensors
        wan_tensors = {}
        converted_count = 0
        
        for source_name, wan_name in tensor_mapping.items():
            if source_name in source_tensors:
                source_tensor = source_tensors[source_name]
                
                # Define target shape based on WAN architecture
                target_shape = self.get_wan_tensor_shape(wan_name)
                
                if target_shape:
                    # Adapt tensor shape
                    adapted_tensor = self.adapt_tensor_shape(
                        source_tensor, 
                        source_tensor.shape, 
                        target_shape, 
                        wan_name
                    )
                    wan_tensors[wan_name] = adapted_tensor
                    converted_count += 1
                else:
                    # Use original tensor if target shape unknown
                    wan_tensors[wan_name] = source_tensor
                    converted_count += 1
                    
        print(f"✅ Converted {converted_count} tensors to WAN format")
        
        # Add any missing WAN-specific tensors
        self.add_missing_wan_tensors(wan_tensors)
        
        return wan_tensors
    
    def get_wan_tensor_shape(self, tensor_name: str) -> Optional[Tuple[int, ...]]:
        """
        Get expected tensor shape for WAN architecture
        
        Args:
            tensor_name: Name of the WAN tensor
            
        Returns:
            Expected tensor shape or None if unknown
        """
        # Define expected shapes for WAN tensors
        shapes = {
            # Self-attention
            'self_attn.q.weight': (self.wan_dim, self.wan_dim),
            'self_attn.k.weight': (self.wan_dim, self.wan_dim),
            'self_attn.v.weight': (self.wan_dim, self.wan_dim),
            'self_attn.o.weight': (self.wan_dim, self.wan_dim),
            'self_attn.q.bias': (self.wan_dim,),
            'self_attn.k.bias': (self.wan_dim,),
            'self_attn.v.bias': (self.wan_dim,),
            'self_attn.o.bias': (self.wan_dim,),
            
            # Cross-attention
            'cross_attn.q.weight': (self.wan_dim, self.wan_dim),
            'cross_attn.k.weight': (self.wan_dim, self.cross_attention_dim),
            'cross_attn.v.weight': (self.wan_dim, self.cross_attention_dim),
            'cross_attn.o.weight': (self.wan_dim, self.wan_dim),
            'cross_attn.q.bias': (self.wan_dim,),
            'cross_attn.k.bias': (self.wan_dim,),
            'cross_attn.v.bias': (self.wan_dim,),
            'cross_attn.o.bias': (self.wan_dim,),
            
            # FFN
            'ffn.0.weight': (self.wan_feedforward_dim, self.wan_dim),
            'ffn.2.weight': (self.wan_dim, self.wan_feedforward_dim),
            'ffn.0.bias': (self.wan_feedforward_dim,),
            'ffn.2.bias': (self.wan_dim,),
            
            # Norm
            'norm3.weight': (self.wan_dim,),
            'norm3.bias': (self.wan_dim,),
            
            # Time modulation  
            'time_bias': (6, self.wan_dim),
        }
        
        # Find matching shape
        for pattern, shape in shapes.items():
            if pattern in tensor_name:
                return shape
                
        return None
    
    def add_missing_wan_tensors(self, wan_tensors: Dict[str, torch.Tensor]):
        """
        Add any missing tensors required by WAN architecture
        
        Args:
            wan_tensors: Dictionary of WAN tensors to augment
        """
        # Check for missing essential tensors and create them
        required_tensors = [
            'input_proj.weight',
            'input_proj.bias', 
            'output_proj.weight',
            'output_proj.bias',
            'time_embedding.time_mlp.0.weight',
            'time_embedding.time_mlp.0.bias',
            'time_embedding.time_mlp.2.weight', 
            'time_embedding.time_mlp.2.bias',
        ]
        
        for tensor_name in required_tensors:
            if tensor_name not in wan_tensors:
                shape = self.get_wan_tensor_shape(tensor_name) or self.infer_missing_tensor_shape(tensor_name)
                if shape:
                    # Initialize missing tensor
                    tensor = torch.zeros(shape, dtype=torch.float32)
                    nn.init.xavier_uniform_(tensor) if 'weight' in tensor_name else nn.init.zeros_(tensor)
                    wan_tensors[tensor_name] = tensor
                    print(f"➕ Added missing tensor: {tensor_name} {shape}")
    
    def infer_missing_tensor_shape(self, tensor_name: str) -> Optional[Tuple[int, ...]]:
        """
        Infer shape for missing tensors based on WAN architecture
        
        Args:
            tensor_name: Name of missing tensor
            
        Returns:
            Inferred tensor shape
        """
        if 'input_proj' in tensor_name:
            return (self.wan_dim, 16) if 'weight' in tensor_name else (self.wan_dim,)
        elif 'output_proj' in tensor_name:
            return (16, self.wan_dim) if 'weight' in tensor_name else (16,)
        elif 'time_mlp.0' in tensor_name:
            return (self.wan_dim * 4, 256) if 'weight' in tensor_name else (self.wan_dim * 4,)
        elif 'time_mlp.2' in tensor_name:
            return (self.wan_dim * 6, self.wan_dim * 4) if 'weight' in tensor_name else (self.wan_dim * 6,)
            
        return None


def create_tensor_adapter(model_tensors: Dict[str, torch.Tensor], model_size: str = "14B") -> Dict[str, torch.Tensor]:
    """
    Create tensor adapter and convert model tensors for WAN compatibility
    
    Args:
        model_tensors: Original model tensors  
        model_size: Target WAN model size ("14B" or "1.3B")
        
    Returns:
        WAN-compatible tensors
    """
    print("🔧 Creating WAN tensor adapter...")
    
    adapter = DiTToWanTensorAdapter(model_size)
    wan_tensors = adapter.convert_tensors_to_wan_format(model_tensors)
    
    print(f"✅ Tensor adaptation complete: {len(wan_tensors)} WAN-compatible tensors")
    return wan_tensors


def validate_wan_tensors(wan_tensors: Dict[str, torch.Tensor], model_size: str = "14B") -> bool:
    """
    Validate that WAN tensors have correct shapes and types
    
    Args:
        wan_tensors: WAN tensors to validate
        model_size: Expected model size
        
    Returns:
        True if tensors are valid
    """
    adapter = DiTToWanTensorAdapter(model_size)
    
    errors = []
    for name, tensor in wan_tensors.items():
        expected_shape = adapter.get_wan_tensor_shape(name)
        if expected_shape and tensor.shape != expected_shape:
            errors.append(f"Tensor {name}: expected {expected_shape}, got {tensor.shape}")
            
    if errors:
        print("❌ Tensor validation errors:")
        for error in errors[:5]:  # Show first 5 errors
            print(f"   {error}")
        if len(errors) > 5:
            print(f"   ... and {len(errors) - 5} more errors")
        return False
    else:
        print(f"✅ All {len(wan_tensors)} WAN tensors validated successfully")
        return True
