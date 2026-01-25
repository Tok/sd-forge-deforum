"""Diagnostic script to test DA3 depth value normalization."""
import sys
import os
import torch
import numpy as np
from PIL import Image

# Add extension to path
ext_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ext_path)

from deforum.depth.depth_anything_v3 import DepthAnythingV3

def test_da3_normalization():
    """Test that DA3 returns properly normalized depth values."""
    print("Testing DA3 depth normalization...")

    # Create test image
    test_image = Image.new('RGB', (512, 512), color=(128, 128, 128))

    # Initialize DA3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = DepthAnythingV3(device, model_size='small', variant='mono')

    # Predict depth
    depth = model.predict(test_image)

    # Check depth tensor properties
    print(f"\nDepth tensor shape: {depth.shape}")
    print(f"Depth dtype: {depth.dtype}")
    print(f"Depth device: {depth.device}")

    # Get depth statistics
    depth_min = depth.min().item()
    depth_max = depth.max().item()
    depth_mean = depth.mean().item()
    depth_std = depth.std().item()

    print(f"\nDepth statistics:")
    print(f"  Min: {depth_min:.6f}")
    print(f"  Max: {depth_max:.6f}")
    print(f"  Mean: {depth_mean:.6f}")
    print(f"  Std: {depth_std:.6f}")
    print(f"  Range: {depth_max - depth_min:.6f}")

    # Check normalization
    if depth_min < 0.0 or depth_max > 1.0:
        print(f"\n⚠️  WARNING: Depth values outside [0, 1] range!")
        print(f"   Expected: [0.0, 1.0]")
        print(f"   Got: [{depth_min:.6f}, {depth_max:.6f}]")
        return False

    # Check if range is reasonable
    depth_range = depth_max - depth_min
    if depth_range < 0.1:
        print(f"\n⚠️  WARNING: Depth range too small ({depth_range:.6f})")
        print(f"   This will cause poor depth warping!")
        return False

    print(f"\n✓ Depth normalization looks good!")
    print(f"  Values in [0, 1] range: ✓")
    print(f"  Sufficient range ({depth_range:.6f}): ✓")

    return True

if __name__ == "__main__":
    try:
        success = test_da3_normalization()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
