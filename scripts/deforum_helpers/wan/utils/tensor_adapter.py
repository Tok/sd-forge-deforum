"""
Wan Tensor Adapter - Simplified
Handles basic tensor compatibility for Wan models
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class WanTensorValidator:
    """
    Simplified tensor validator for Wan models
    """
    
    def __init__(self, model_size: str = "14B"):
        self.model_size = model_size
        
        # Basic Wan model configurations
        if model_size == "14B":
            self.wan_dim = 5120
            self.wan_layers = 40
        else:  # 1.3B
            self.wan_dim = 1536
            self.wan_layers = 30
    
    def validate_model_tensors(self, tensors: Dict[str, torch.Tensor]) -> bool:
        """
        Basic validation of model tensors
        
        Args:
            tensors: Dictionary of model tensors
            
        Returns:
            True if tensors appear valid for Wan
        """
        if not tensors:
            print("❌ No tensors provided")
            return False
        
        print(f"🔍 Validating {len(tensors)} tensors for Wan {self.model_size} model")
        
        # Check for basic requirements
        has_weights = any('weight' in name for name in tensors.keys())
        has_reasonable_sizes = all(
            tensor.numel() > 0 and tensor.numel() < 1e10 
            for tensor in tensors.values()
        )
        
        if not has_weights:
            print("❌ No weight tensors found")
            return False
            
        if not has_reasonable_sizes:
            print("❌ Some tensors have unreasonable sizes")
            return False
        
        print(f"✅ Basic tensor validation passed")
        print(f"📊 Total parameters: {sum(t.numel() for t in tensors.values()):,}")
        
        return True
    
    def get_model_info(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Get basic information about the model tensors"""
        if not tensors:
            return {'error': 'No tensors provided'}
        
        total_params = sum(t.numel() for t in tensors.values())
        total_size_gb = sum(t.numel() * t.element_size() for t in tensors.values()) / (1024**3)
        
        # Analyze tensor names to infer model type
        tensor_names = list(tensors.keys())
        model_type = 'unknown'
        
        if any('diffusion_pytorch_model' in name for name in tensor_names):
            model_type = 'diffusion_transformer'
        elif any('flux' in name.lower() for name in tensor_names):
            model_type = 'flux'
        elif any('wan' in name.lower() for name in tensor_names):
            model_type = 'wan'
            
        return {
            'tensor_count': len(tensors),
            'total_parameters': total_params,
            'size_gb': round(total_size_gb, 2),
            'model_type': model_type,
            'sample_tensors': list(tensors.keys())[:5]
        }


def validate_wan_model(model_tensors: Dict[str, torch.Tensor], model_size: str = "14B") -> bool:
    """
    Validate model tensors for Wan compatibility - simplified
    
    Args:
        model_tensors: Model tensor dictionary
        model_size: Expected model size
        
    Returns:
        True if tensors are valid
    """
    validator = WanTensorValidator(model_size)
    return validator.validate_model_tensors(model_tensors)


def get_wan_model_info(model_tensors: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """
    Get information about Wan model tensors - simplified
    
    Args:
        model_tensors: Model tensor dictionary
        
    Returns:
        Dictionary with model information
    """
    validator = WanTensorValidator()
    return validator.get_model_info(model_tensors)


def estimate_model_size(model_tensors: Dict[str, torch.Tensor]) -> str:
    """
    Estimate model size category from tensor count/size
    
    Args:
        model_tensors: Model tensor dictionary  
        
    Returns:
        Estimated size: "1.3B" or "14B"
    """
    if not model_tensors:
        return "unknown"
    
    total_params = sum(t.numel() for t in model_tensors.values())
    
    # Rough estimation based on parameter count
    if total_params < 5e9:  # Less than 5B parameters
        return "1.3B"
    else:
        return "14B"


# For compatibility with existing code
create_tensor_adapter = lambda tensors, size: tensors  # No-op for simplified approach
DiTToWanTensorAdapter = WanTensorValidator  # Alias for compatibility
