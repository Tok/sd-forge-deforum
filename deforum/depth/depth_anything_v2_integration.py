from torchvision import transforms
# noinspection PyUnresolvedReferences
from transformers import pipeline  # provided by Forge


class DepthAnything:
    def __init__(self, device):
        model_name = "depth-anything/Depth-Anything-V2-Small-hf"
        print(f"Loading Depth Anything model from {model_name}...")
        self.pipe = pipeline(task='depth-estimation', model=model_name, device=device)
        self.pipe.model.to(device)

    def predict(self, image):
        depth_tensor = self.pipe(image)['depth']
        return transforms.ToTensor()(depth_tensor).unsqueeze(0)
