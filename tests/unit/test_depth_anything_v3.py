"""Unit tests for Depth Anything V3 integration.

Tests all pure helper functions and the main predict() pipeline.
"""

import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import Mock, MagicMock, patch

# Import functions to test
from deforum.depth.depth_anything_v3 import (
    _get_model_name,
    _convert_bgr_to_rgb_pil,
    _extract_image_dimensions,
    _prepare_image_for_inference,
    _convert_depth_to_tensor,
    _normalize_depth_range,
    _resize_depth_to_match_image,
    _convert_images_to_pil,
    DepthAnythingV3,
)


class TestModelNameMapping:
    """Test model name detection and mapping."""

    def test_mono_small(self):
        """Test mono small model mapping."""
        assert _get_model_name('mono', 'small') == 'depth-anything/Depth-Anything-V3-Small'

    def test_mono_base(self):
        """Test mono base model mapping."""
        assert _get_model_name('mono', 'base') == 'depth-anything/Depth-Anything-V3-Base'

    def test_mono_large(self):
        """Test mono large model mapping."""
        assert _get_model_name('mono', 'large') == 'depth-anything/Depth-Anything-V3-Large'

    def test_anyview_small(self):
        """Test any-view small model mapping."""
        assert _get_model_name('any-view', 'small') == 'depth-anything/DA3-Small'

    def test_anyview_base(self):
        """Test any-view base model mapping."""
        assert _get_model_name('any-view', 'base') == 'depth-anything/DA3-Base'

    def test_anyview_large(self):
        """Test any-view large model mapping."""
        assert _get_model_name('any-view', 'large') == 'depth-anything/DA3-Large'

    def test_case_insensitive(self):
        """Test that variant and size are case-insensitive."""
        assert _get_model_name('MONO', 'LARGE') == 'depth-anything/Depth-Anything-V3-Large'
        assert _get_model_name('Any-View', 'Base') == 'depth-anything/DA3-Base'

    def test_invalid_variant_fallback(self):
        """Test fallback to mono/small for invalid variant."""
        assert _get_model_name('invalid', 'large') == 'depth-anything/Depth-Anything-V3-Small'

    def test_invalid_size_fallback(self):
        """Test fallback to mono/small for invalid size."""
        assert _get_model_name('mono', 'invalid') == 'depth-anything/Depth-Anything-V3-Small'


class TestImageConversion:
    """Test image format conversion functions."""

    def test_convert_bgr_to_rgb_pil(self):
        """Test BGR numpy array to RGB PIL Image conversion."""
        # Create BGR image (blue channel = 255)
        bgr_image = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 255  # Blue in BGR

        # Convert to RGB PIL
        pil_image = _convert_bgr_to_rgb_pil(bgr_image)

        # Check it's PIL Image
        assert isinstance(pil_image, Image.Image)

        # Check RGB values (blue channel should now be at index 2)
        rgb_array = np.array(pil_image)
        assert rgb_array[0, 0, 2] == 255  # Blue in RGB
        assert rgb_array[0, 0, 0] == 0    # Red
        assert rgb_array[0, 0, 1] == 0    # Green

    def test_extract_dimensions_from_numpy(self):
        """Test dimension extraction from numpy array."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        h, w = _extract_image_dimensions(image)
        assert h == 480
        assert w == 640

    def test_extract_dimensions_from_pil(self):
        """Test dimension extraction from PIL Image."""
        image = Image.new('RGB', (640, 480))
        h, w = _extract_image_dimensions(image)
        assert h == 480
        assert w == 640

    def test_prepare_numpy_image(self):
        """Test preparing numpy array for inference."""
        # Create BGR numpy image
        bgr_image = np.zeros((480, 640, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 100  # Blue channel

        pil_image, h, w = _prepare_image_for_inference(bgr_image)

        # Check outputs
        assert isinstance(pil_image, Image.Image)
        assert h == 480
        assert w == 640

        # Check color conversion happened (blue should be in RGB position)
        rgb_array = np.array(pil_image)
        assert rgb_array[0, 0, 2] == 100  # Blue in RGB

    def test_prepare_pil_image(self):
        """Test preparing PIL Image for inference (no conversion needed)."""
        original_image = Image.new('RGB', (640, 480), color=(255, 0, 0))

        pil_image, h, w = _prepare_image_for_inference(original_image)

        # Check outputs
        assert pil_image is original_image  # Should be same object
        assert h == 480
        assert w == 640


class TestDepthTensorConversion:
    """Test depth data to tensor conversion."""

    def test_convert_numpy_2d(self):
        """Test converting 2D numpy array to tensor."""
        depth_np = np.random.rand(480, 640).astype(np.float32)
        depth_tensor = _convert_depth_to_tensor(depth_np)

        # Check shape
        assert depth_tensor.shape == (1, 1, 480, 640)
        assert isinstance(depth_tensor, torch.Tensor)

    def test_convert_tensor_2d(self):
        """Test converting 2D tensor to standard format."""
        depth_tensor = torch.rand(480, 640)
        result = _convert_depth_to_tensor(depth_tensor)

        # Check shape
        assert result.shape == (1, 1, 480, 640)

    def test_convert_tensor_3d(self):
        """Test converting 3D tensor to standard format."""
        depth_tensor = torch.rand(1, 480, 640)
        result = _convert_depth_to_tensor(depth_tensor)

        # Check shape
        assert result.shape == (1, 1, 480, 640)

    def test_convert_tensor_4d(self):
        """Test that 4D tensor is returned as-is."""
        depth_tensor = torch.rand(1, 1, 480, 640)
        result = _convert_depth_to_tensor(depth_tensor)

        # Should be unchanged
        assert result.shape == (1, 1, 480, 640)


class TestDepthNormalization:
    """Test depth value normalization."""

    def test_normalize_arbitrary_range(self):
        """Test normalizing depth from arbitrary range to [0, 1]."""
        # Create depth with range [0.94, 1.04] (typical DA3 output)
        depth = torch.tensor([[[[0.94, 1.0, 1.04]]]])

        normalized = _normalize_depth_range(depth)

        # Check normalization
        assert normalized.min().item() == pytest.approx(0.0, abs=1e-6)
        assert normalized.max().item() == pytest.approx(1.0, abs=1e-6)

    def test_normalize_zero_range(self):
        """Test that constant depth is handled gracefully."""
        # All same value
        depth = torch.ones(1, 1, 10, 10)

        normalized = _normalize_depth_range(depth)

        # Should remain unchanged (all ones)
        assert torch.all(normalized == 1.0)

    def test_normalize_negative_range(self):
        """Test normalizing negative depth values."""
        depth = torch.tensor([[[[-5.0, 0.0, 5.0]]]])

        normalized = _normalize_depth_range(depth)

        # Check normalization
        assert normalized.min().item() == pytest.approx(0.0, abs=1e-6)
        assert normalized.max().item() == pytest.approx(1.0, abs=1e-6)
        assert normalized[0, 0, 0, 1].item() == pytest.approx(0.5, abs=1e-6)  # Middle value


class TestDepthResizing:
    """Test depth map resizing."""

    def test_resize_upscale(self):
        """Test upscaling depth map."""
        # Small depth map
        depth = torch.rand(1, 1, 100, 100)

        # Upscale to larger size
        resized = _resize_depth_to_match_image(depth, 200, 200)

        # Check new size
        assert resized.shape == (1, 1, 200, 200)

    def test_resize_downscale(self):
        """Test downscaling depth map."""
        # Large depth map
        depth = torch.rand(1, 1, 1000, 1000)

        # Downscale to smaller size
        resized = _resize_depth_to_match_image(depth, 500, 500)

        # Check new size
        assert resized.shape == (1, 1, 500, 500)

    def test_no_resize_needed(self):
        """Test that matching dimensions returns same tensor."""
        depth = torch.rand(1, 1, 480, 640)

        resized = _resize_depth_to_match_image(depth, 480, 640)

        # Should be same tensor (no resize needed)
        assert torch.equal(resized, depth)

    def test_resize_preserves_range(self):
        """Test that resizing preserves value range."""
        # Normalized depth
        depth = torch.rand(1, 1, 100, 100)
        original_min = depth.min()
        original_max = depth.max()

        # Resize
        resized = _resize_depth_to_match_image(depth, 200, 200)

        # Range should be approximately preserved
        assert resized.min() >= original_min - 0.01
        assert resized.max() <= original_max + 0.01


class TestBatchImageConversion:
    """Test batch image conversion to PIL."""

    def test_convert_mixed_list(self):
        """Test converting list with both numpy and PIL images."""
        # Create mixed list
        bgr_image = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr_image[:, :, 0] = 255  # Blue in BGR

        pil_image = Image.new('RGB', (10, 10), color=(255, 0, 0))

        images = [bgr_image, pil_image]

        # Convert
        result = _convert_images_to_pil(images)

        # Check all PIL Images
        assert len(result) == 2
        assert all(isinstance(img, Image.Image) for img in result)

        # Check first image was converted from BGR
        rgb_array = np.array(result[0])
        assert rgb_array[0, 0, 2] == 255  # Blue in RGB position

    def test_convert_empty_list(self):
        """Test converting empty list."""
        result = _convert_images_to_pil([])
        assert result == []

    def test_convert_all_pil(self):
        """Test that all PIL images are preserved."""
        images = [Image.new('RGB', (10, 10)) for _ in range(3)]

        result = _convert_images_to_pil(images)

        # Should be same objects
        assert result == images


class TestDepthAnythingV3Mock:
    """Test DepthAnythingV3 class with mocked model."""

    @patch('depth_anything_3.api.DepthAnything3')
    def test_init_success(self, mock_da3_class):
        """Test successful initialization."""
        mock_model = Mock()
        mock_da3_class.from_pretrained.return_value = mock_model

        device = torch.device('cpu')
        da3 = DepthAnythingV3(device, model_size='small', variant='mono')

        # Check initialization
        assert da3.device == device
        assert da3.model_size == 'small'
        assert da3.variant == 'mono'
        assert da3.model == mock_model

        # Check model was loaded and moved to device
        mock_da3_class.from_pretrained.assert_called_once_with('depth-anything/Depth-Anything-V3-Small')
        mock_model.to.assert_called_once_with(device)

    def test_init_import_error(self):
        """Test ImportError when DA3 package not installed."""
        # This test requires depth-anything-3 package to NOT be installed
        # Since the package is likely installed in the test environment,
        # we can only verify the error message format is correct
        #
        # To actually test: uninstall depth-anything-3, run this test, reinstall
        # For now, we'll skip this test in environments where DA3 is available
        try:
            from depth_anything_3.api import DepthAnything3
            pytest.skip("depth-anything-3 package is installed, cannot test ImportError")
        except ImportError:
            # Package not installed - test should raise our custom ImportError
            device = torch.device('cpu')
            with pytest.raises(ImportError, match="depth-anything-3 package required"):
                DepthAnythingV3(device)

    @patch('depth_anything_3.api.DepthAnything3')
    def test_predict_with_numpy_bgr(self, mock_da3_class):
        """Test predict() with numpy BGR image."""
        # Mock model and inference result
        mock_model = Mock()
        mock_result = Mock()
        mock_result.depth = [np.random.rand(240, 320).astype(np.float32)]  # Downsampled
        mock_model.inference.return_value = mock_result
        mock_da3_class.from_pretrained.return_value = mock_model

        # Create instance
        device = torch.device('cpu')
        da3 = DepthAnythingV3(device, model_size='small', variant='mono')

        # Create test image (original size)
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)

        # Run prediction
        depth = da3.predict(test_image)

        # Check output
        assert depth.shape == (1, 1, 480, 640)  # Resized to match original
        assert isinstance(depth, torch.Tensor)

        # Check depth is normalized [0, 1]
        assert depth.min() >= 0.0
        assert depth.max() <= 1.0

    @patch('depth_anything_3.api.DepthAnything3')
    def test_predict_with_pil_image(self, mock_da3_class):
        """Test predict() with PIL Image."""
        # Mock model and inference result
        mock_model = Mock()
        mock_result = Mock()
        mock_result.depth = [np.random.rand(240, 320).astype(np.float32)]
        mock_model.inference.return_value = mock_result
        mock_da3_class.from_pretrained.return_value = mock_model

        # Create instance
        device = torch.device('cpu')
        da3 = DepthAnythingV3(device)

        # Create PIL Image
        test_image = Image.new('RGB', (640, 480))

        # Run prediction
        depth = da3.predict(test_image)

        # Check output
        assert depth.shape == (1, 1, 480, 640)
        assert isinstance(depth, torch.Tensor)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
