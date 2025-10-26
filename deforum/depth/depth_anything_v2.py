from torchvision import transforms
# noinspection PyUnresolvedReferences
from transformers import pipeline  # provided by Forge
from deforum.utils.system.logging import get_logger

# Initialize logger
logger = get_logger()



class DepthAnything:
    def __init__(self, device, model_size='small'):
        """
        Initialize Depth Anything V2 model

        Args:
            device: torch device (cpu/cuda)
            model_size: 'small', 'base', or 'large' (default: 'small')
        """
        model_map = {
            'small': 'depth-anything/Depth-Anything-V2-Small-hf',
            'base': 'depth-anything/Depth-Anything-V2-Base-hf',
            'large': 'depth-anything/Depth-Anything-V2-Large-hf'
        }
        model_name = model_map.get(model_size.lower(), model_map['small'])
        logger.info(f"Loading Depth Anything V2 model ({model_size}) from {model_name}...")
        self.pipe = pipeline(task='depth-estimation', model=model_name, device=device)
        self.pipe.model.to(device)

    def predict(self, image):
        depth_tensor = self.pipe(image)['depth']
        return transforms.ToTensor()(depth_tensor).unsqueeze(0)
