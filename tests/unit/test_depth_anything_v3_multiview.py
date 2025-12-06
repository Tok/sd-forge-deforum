"""Unit tests for DA3 multiview and 3DGS capabilities.

Tests Phase 2 (multi-view) and Phase 3 (3D Gaussian Splatting) features.
"""

import pytest
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from unittest.mock import Mock, patch

# Add extension root to path for direct module import (avoiding Forge deps)
extension_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(extension_root))

# Mock ALL external dependencies BEFORE importing depth_anything_v3
# This prevents ModuleNotFoundError in CI environment

# Mock logger module
mock_logger_module = Mock()
mock_logger_instance = Mock()
mock_logger_module.get_logger.return_value = mock_logger_instance
sys.modules['deforum.utils.system.logging'] = mock_logger_module

# Mock depth_anything_3 module (not installed in CI)
mock_da3_module = Mock()
mock_da3_class = Mock()
mock_da3_module.DepthAnything3 = mock_da3_class
mock_da3_module.api = Mock()
mock_da3_module.api.DepthAnything3 = mock_da3_class
sys.modules['depth_anything_3'] = mock_da3_module
sys.modules['depth_anything_3.api'] = mock_da3_module.api

# Import directly from depth_anything_v3 module (not through deforum.depth package)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "depth_anything_v3",
    extension_root / "deforum" / "depth" / "depth_anything_v3.py"
)
da3_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(da3_module)

# Extract what we need
DepthAnythingV3 = da3_module.DepthAnythingV3


class TestMultiviewPrediction:
    """Test multi-view depth prediction (Phase 2)."""

    @patch('depth_anything_3.api.DepthAnything3')
    def test_multiview_with_anyview_model(self, mock_da3_class):
        """Test multiview prediction with any-view variant."""
        # Mock model
        mock_model = Mock()
        mock_result = Mock()
        mock_result.depth = [np.random.rand(480, 640) for _ in range(3)]
        mock_result.conf = [np.ones((480, 640)) for _ in range(3)]
        mock_result.extrinsics = np.random.rand(3, 4, 4)
        mock_result.intrinsics = np.random.rand(3, 3, 3)
        mock_model.inference.return_value = mock_result
        mock_da3_class.from_pretrained.return_value = mock_model

        # Create instance with any-view variant
        device = torch.device('cpu')
        da3 = DepthAnythingV3(device, model_size='small', variant='any-view')

        # Create test images
        images = [Image.new('RGB', (640, 480)) for _ in range(3)]

        # Run multiview prediction
        result = da3.predict_multiview(images)

        # Check result
        assert result == mock_result
        mock_model.inference.assert_called_once()

    @patch('depth_anything_3.api.DepthAnything3')
    def test_multiview_fallback_with_mono_model(self, mock_da3_class):
        """Test multiview falls back to single-view with mono variant."""
        # Mock model for mono variant
        mock_model = Mock()
        mock_result = Mock()
        mock_result.depth = [np.random.rand(480, 640)]  # Single image result
        mock_model.inference.return_value = mock_result
        mock_da3_class.from_pretrained.return_value = mock_model

        # Create instance with mono variant (NOT any-view)
        device = torch.device('cpu')
        da3 = DepthAnythingV3(device, model_size='small', variant='mono')

        # Create test images
        images = [Image.new('RGB', (640, 480)) for _ in range(3)]

        # Run multiview prediction (should fallback)
        result = da3.predict_multiview(images)

        # Check result has fallback structure
        assert 'depth' in result
        assert 'confidence' in result
        assert 'camera_extrinsics' in result
        assert 'camera_intrinsics' in result

        # Check fallback values
        assert len(result['depth']) == 3  # One per image
        assert len(result['confidence']) == 3
        assert result['camera_extrinsics'] is None  # No multi-view data
        assert result['camera_intrinsics'] is None

        # Check confidence is all ones
        for conf in result['confidence']:
            assert torch.all(conf == 1.0)

    @patch('depth_anything_3.api.DepthAnything3')
    def test_multiview_with_numpy_images(self, mock_da3_class):
        """Test multiview with numpy array images (BGR)."""
        # Mock model
        mock_model = Mock()
        mock_result = Mock()
        mock_result.depth = [np.random.rand(480, 640) for _ in range(2)]
        mock_model.inference.return_value = mock_result
        mock_da3_class.from_pretrained.return_value = mock_model

        # Create instance
        device = torch.device('cpu')
        da3 = DepthAnythingV3(device, variant='any-view')

        # Create numpy BGR images
        images = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
        images[0][:, :, 0] = 255  # Blue channel in BGR

        # Run multiview prediction
        result = da3.predict_multiview(images)

        # Check model was called with PIL images
        call_args = mock_model.inference.call_args
        pil_images = call_args[0][0]
        assert all(isinstance(img, Image.Image) for img in pil_images)

        # Check color conversion (blue should be at index 2 in RGB)
        rgb_array = np.array(pil_images[0])
        assert rgb_array[0, 0, 2] == 255  # Blue in RGB position

    @patch('depth_anything_3.api.DepthAnything3')
    def test_multiview_with_mixed_images(self, mock_da3_class):
        """Test multiview with mixed numpy and PIL images."""
        # Mock model
        mock_model = Mock()
        mock_result = Mock()
        mock_result.depth = [np.random.rand(480, 640) for _ in range(3)]
        mock_model.inference.return_value = mock_result
        mock_da3_class.from_pretrained.return_value = mock_model

        # Create instance
        device = torch.device('cpu')
        da3 = DepthAnythingV3(device, variant='any-view')

        # Create mixed images
        images = [
            np.zeros((480, 640, 3), dtype=np.uint8),  # Numpy BGR
            Image.new('RGB', (640, 480)),              # PIL RGB
            np.ones((480, 640, 3), dtype=np.uint8),   # Numpy BGR
        ]

        # Run multiview prediction
        result = da3.predict_multiview(images)

        # Check all images were converted to PIL
        call_args = mock_model.inference.call_args
        pil_images = call_args[0][0]
        assert len(pil_images) == 3
        assert all(isinstance(img, Image.Image) for img in pil_images)


class Test3DGaussianSplatting:
    """Test 3D Gaussian Splatting estimation (Phase 3)."""

    @patch('depth_anything_3.api.DepthAnything3')
    def test_3dgs_estimation_success(self, mock_da3_class):
        """Test successful 3DGS estimation."""
        # Mock model with 3DGS support
        mock_model = Mock()
        mock_result = Mock()
        mock_result.gaussians = Mock(
            means=np.random.rand(1000, 3),
            rotations=np.random.rand(1000, 4),
            scales=np.random.rand(1000, 3),
            opacities=np.random.rand(1000, 1),
            colors=np.random.rand(1000, 3)
        )
        mock_model.inference.return_value = mock_result
        mock_da3_class.from_pretrained.return_value = mock_model

        # Create instance
        device = torch.device('cpu')
        da3 = DepthAnythingV3(device, variant='any-view')

        # Create keyframe images
        images = [Image.new('RGB', (640, 480)) for _ in range(5)]

        # Run 3DGS estimation
        result = da3.estimate_3d_gaussians(images)

        # Check result
        assert result == mock_result
        assert hasattr(result, 'gaussians')

        # Check inference was called with infer_gs=True
        mock_model.inference.assert_called_once()
        call_kwargs = mock_model.inference.call_args[1]
        assert call_kwargs.get('infer_gs') is True

    @patch('depth_anything_3.api.DepthAnything3')
    def test_3dgs_estimation_failure(self, mock_da3_class):
        """Test 3DGS estimation failure (model doesn't support it)."""
        # Mock model that raises error for 3DGS
        mock_model = Mock()
        mock_model.inference.side_effect = Exception("3DGS not supported by this model")
        mock_da3_class.from_pretrained.return_value = mock_model

        # Create instance
        device = torch.device('cpu')
        da3 = DepthAnythingV3(device)

        # Create images
        images = [Image.new('RGB', (640, 480)) for _ in range(3)]

        # Run 3DGS estimation (should return None on failure)
        result = da3.estimate_3d_gaussians(images)

        # Check result is None
        assert result is None

    @patch('depth_anything_3.api.DepthAnything3')
    def test_3dgs_with_numpy_images(self, mock_da3_class):
        """Test 3DGS estimation with numpy images."""
        # Mock model
        mock_model = Mock()
        mock_result = Mock()
        mock_result.gaussians = Mock()
        mock_model.inference.return_value = mock_result
        mock_da3_class.from_pretrained.return_value = mock_model

        # Create instance
        device = torch.device('cpu')
        da3 = DepthAnythingV3(device)

        # Create numpy BGR images
        images = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.ones((480, 640, 3), dtype=np.uint8) * 128,
        ]

        # Run 3DGS estimation
        result = da3.estimate_3d_gaussians(images)

        # Check images were converted to PIL
        call_args = mock_model.inference.call_args
        pil_images = call_args[0][0]
        assert len(pil_images) == 2
        assert all(isinstance(img, Image.Image) for img in pil_images)

    @patch('depth_anything_3.api.DepthAnything3')
    def test_3dgs_empty_image_list(self, mock_da3_class):
        """Test 3DGS with empty image list."""
        # Mock model
        mock_model = Mock()
        mock_result = Mock()
        mock_model.inference.return_value = mock_result
        mock_da3_class.from_pretrained.return_value = mock_model

        # Create instance
        device = torch.device('cpu')
        da3 = DepthAnythingV3(device)

        # Run with empty list
        result = da3.estimate_3d_gaussians([])

        # Should still call inference with empty list
        call_args = mock_model.inference.call_args
        assert call_args[0][0] == []


class TestModelVariantBehavior:
    """Test behavior differences between mono and any-view variants."""

    @patch('depth_anything_3.api.DepthAnything3')
    def test_mono_loads_correct_model(self, mock_da3_class):
        """Test that mono variant loads correct model."""
        mock_model = Mock()
        mock_da3_class.from_pretrained.return_value = mock_model

        device = torch.device('cpu')
        da3 = DepthAnythingV3(device, model_size='large', variant='mono')

        # Check correct model was loaded (mono/large maps to DA3MONO-LARGE)
        mock_da3_class.from_pretrained.assert_called_with(
            'depth-anything/DA3MONO-LARGE'
        )

    @patch('depth_anything_3.api.DepthAnything3')
    def test_anyview_loads_correct_model(self, mock_da3_class):
        """Test that any-view variant loads correct model."""
        mock_model = Mock()
        mock_da3_class.from_pretrained.return_value = mock_model

        device = torch.device('cpu')
        da3 = DepthAnythingV3(device, model_size='base', variant='any-view')

        # Check correct model was loaded (any-view uses all caps)
        mock_da3_class.from_pretrained.assert_called_with(
            'depth-anything/DA3-BASE'
        )

    @patch('depth_anything_3.api.DepthAnything3')
    def test_default_variant_is_mono(self, mock_da3_class):
        """Test that default variant is mono."""
        mock_model = Mock()
        mock_da3_class.from_pretrained.return_value = mock_model

        device = torch.device('cpu')
        da3 = DepthAnythingV3(device)

        # Check default is mono/small (maps to DA3MONO-LARGE - only mono variant)
        mock_da3_class.from_pretrained.assert_called_with(
            'depth-anything/DA3MONO-LARGE'
        )
        assert da3.variant == 'mono'

    @patch('depth_anything_3.api.DepthAnything3')
    def test_default_size_is_small(self, mock_da3_class):
        """Test that default model size is small."""
        mock_model = Mock()
        mock_da3_class.from_pretrained.return_value = mock_model

        device = torch.device('cpu')
        da3 = DepthAnythingV3(device)

        assert da3.model_size == 'small'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
