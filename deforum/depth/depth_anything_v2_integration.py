"""
Depth-Anything-V2 Integration - Primary Depth Estimation Method

Depth-Anything-V2 is the default and recommended depth estimation method for Deforum.
It provides superior accuracy and performance compared to MiDaS.
"""

import torch
import numpy as np
from torchvision import transforms
from typing import Optional, Literal, Union
from PIL import Image

try:
    from transformers import pipeline
except ImportError:
    print("Warning: transformers not available. Depth-Anything-V2 will not work.")
    pipeline = None


ModelSize = Literal['small', 'base', 'large']

# Model configurations - prioritizing quality and performance
DEPTH_ANYTHING_MODELS = {
    'small': 'depth-anything/Depth-Anything-V2-Small-hf',
    'base': 'depth-anything/Depth-Anything-V2-Base-hf', 
    'large': 'depth-anything/Depth-Anything-V2-Large-hf'
}

# Default model (good balance of speed and quality)
DEFAULT_MODEL_SIZE = 'base'


class DepthAnything:
    """
    Depth-Anything-V2 depth estimation model.
    
    This is the primary depth estimation method for Deforum, providing
    superior accuracy compared to MiDaS while maintaining good performance.
    """
    
    def __init__(self, device: str = 'auto', model_size: ModelSize = DEFAULT_MODEL_SIZE):
        """
        Initialize Depth-Anything-V2 model.
        
        Args:
            device: Device to run on ('auto', 'cuda', 'cpu')
            model_size: Model size ('small', 'base', 'large')
        """
        if pipeline is None:
            raise ImportError("transformers package required for Depth-Anything-V2")
            
        self.device = self._get_device(device)
        self.model_size = model_size
        self.model_name = DEPTH_ANYTHING_MODELS[model_size]
        
        print(f"🔍 Loading Depth-Anything-V2 ({model_size}) from {self.model_name}...")
        print(f"📱 Using device: {self.device}")
        
        try:
            self.pipe = pipeline(
                task='depth-estimation', 
                model=self.model_name, 
                device=self.device
            )
            self.pipe.model.to(self.device)
            print("✅ Depth-Anything-V2 loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load Depth-Anything-V2: {e}")
            raise
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def predict(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Predict depth for an image.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Depth tensor with shape (1, H, W)
        """
        try:
            # Ensure image is PIL Image
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            # Get depth prediction
            result = self.pipe(image)
            depth_tensor = result['depth']
            
            # Convert to tensor format expected by Deforum
            if isinstance(depth_tensor, Image.Image):
                depth_tensor = transforms.ToTensor()(depth_tensor)
            elif isinstance(depth_tensor, np.ndarray):
                depth_tensor = torch.from_numpy(depth_tensor).float()
            
            # Ensure correct shape (1, H, W)
            if depth_tensor.dim() == 2:
                depth_tensor = depth_tensor.unsqueeze(0)
            elif depth_tensor.dim() == 3 and depth_tensor.shape[0] == 3:
                # Convert RGB to grayscale if needed
                depth_tensor = depth_tensor.mean(dim=0, keepdim=True)
            
            return depth_tensor
            
        except Exception as e:
            print(f"❌ Depth prediction failed: {e}")
            raise


def load_depth_anything_model(device: str = 'auto', model_size: ModelSize = DEFAULT_MODEL_SIZE) -> DepthAnything:
    """
    Load Depth-Anything-V2 model.
    
    Args:
        device: Device to run on ('auto', 'cuda', 'cpu')
        model_size: Model size ('small', 'base', 'large')
        
    Returns:
        DepthAnything model instance
    """
    return DepthAnything(device=device, model_size=model_size)


def estimate_depth_anything(image: Union[Image.Image, np.ndarray], 
                          model: Optional[DepthAnything] = None,
                          device: str = 'auto',
                          model_size: ModelSize = DEFAULT_MODEL_SIZE) -> torch.Tensor:
    """
    Estimate depth using Depth-Anything-V2.
    
    Args:
        image: Input image
        model: Pre-loaded model (optional)
        device: Device to use if loading new model
        model_size: Model size if loading new model
        
    Returns:
        Depth tensor
    """
    if model is None:
        model = load_depth_anything_model(device=device, model_size=model_size)
    
    return model.predict(image)


def get_default_depth_estimator(device: str = 'auto') -> DepthAnything:
    """
    Get the default depth estimator (Depth-Anything-V2 base model).
    
    Args:
        device: Device to run on
        
    Returns:
        Default depth estimator
    """
    return load_depth_anything_model(device=device, model_size=DEFAULT_MODEL_SIZE)


# Convenience function for backward compatibility
def get_depth_anything_model(device: str = 'auto') -> DepthAnything:
    """Legacy function name for backward compatibility."""
    return get_default_depth_estimator(device)
