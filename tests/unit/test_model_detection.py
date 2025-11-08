"""Unit tests for model detection utilities."""

import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

# Add parent directory to path to allow direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deforum.utils.model_detection import (
    is_lumina_model,
    is_flux_model,
    get_model_name,
    get_recommended_cfg_scale,
    get_recommended_steps,
    _get_model_class_name,
    _get_checkpoint_name,
    _check_diffusion_model_class,
    _detect_flux_variant,
    LUMINA_CONFIG,
    FLUX_CONFIG,
    DEFAULT_CONFIG,
)


class TestModelConfig:
    """Test suite for ModelConfig constants."""

    def test_lumina_config(self):
        """Test Lumina configuration values."""
        assert LUMINA_CONFIG.name == "Lumina 2.0"
        assert LUMINA_CONFIG.cfg_range == (4.0, 5.5)
        assert LUMINA_CONFIG.recommended_steps == 30

    def test_flux_config(self):
        """Test Flux configuration values."""
        assert FLUX_CONFIG.name == "Flux"
        assert FLUX_CONFIG.cfg_range == (1.0, 3.5)
        assert FLUX_CONFIG.recommended_steps == 20

    def test_default_config(self):
        """Test default configuration values."""
        assert DEFAULT_CONFIG.name == "Unknown"
        assert DEFAULT_CONFIG.cfg_range == (7.0, 12.0)
        assert DEFAULT_CONFIG.recommended_steps == 20


class TestHelperFunctions:
    """Test suite for internal helper functions."""

    def test_get_model_class_name_with_class(self):
        """Test extracting class name from object with __class__."""
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "TestModelClass"

        result = _get_model_class_name(mock_model)
        assert result == "TestModelClass"

    def test_get_model_class_name_without_class(self):
        """Test extracting class name from object without __class__."""
        # Create object without __class__ attribute
        mock_model = {}  # dict doesn't expose __class__ the same way

        result = _get_model_class_name(mock_model)
        # Should still work with dict's __class__
        assert result == "dict"

    def test_get_checkpoint_name_success(self):
        """Test extracting checkpoint name from shared module."""
        shared = SimpleNamespace()
        shared.opts = SimpleNamespace()
        shared.opts.sd_model_checkpoint = "Flux-Dev-v1.0.safetensors"

        result = _get_checkpoint_name(shared)
        assert result == "flux-dev-v1.0.safetensors"

    def test_get_checkpoint_name_no_opts(self):
        """Test checkpoint name extraction when opts missing."""
        shared = SimpleNamespace()
        result = _get_checkpoint_name(shared)
        assert result is None

    def test_get_checkpoint_name_no_checkpoint(self):
        """Test checkpoint name extraction when checkpoint missing."""
        shared = SimpleNamespace()
        shared.opts = SimpleNamespace()
        result = _get_checkpoint_name(shared)
        assert result is None

    def test_check_diffusion_model_class_match(self):
        """Test diffusion model class checking with match."""
        # Create actual class instance instead of trying to assign __class__
        Lumina2NextDiT = type('Lumina2NextDiT', (), {})
        diff_model_instance = Lumina2NextDiT()

        model = SimpleNamespace()
        model.forge_objects = SimpleNamespace()
        model.forge_objects.unet = SimpleNamespace()
        model.forge_objects.unet.model = SimpleNamespace()
        model.forge_objects.unet.model.diffusion_model = diff_model_instance

        result = _check_diffusion_model_class(model, 'Lumina2NextDiT')
        assert result is True

    def test_check_diffusion_model_class_no_match(self):
        """Test diffusion model class checking without match."""
        # Create actual class instance
        OtherModel = type('OtherModel', (), {})
        diff_model_instance = OtherModel()

        model = SimpleNamespace()
        model.forge_objects = SimpleNamespace()
        model.forge_objects.unet = SimpleNamespace()
        model.forge_objects.unet.model = SimpleNamespace()
        model.forge_objects.unet.model.diffusion_model = diff_model_instance

        result = _check_diffusion_model_class(model, 'Lumina2NextDiT')
        assert result is False

    def test_check_diffusion_model_class_no_forge_objects(self):
        """Test diffusion model class checking when forge_objects missing."""
        model = SimpleNamespace()
        result = _check_diffusion_model_class(model, 'Lumina2NextDiT')
        assert result is False

    def test_detect_flux_variant_schnell(self):
        """Test Flux variant detection for Schnell."""
        result = _detect_flux_variant("flux-schnell-v1.0")
        assert result == "Flux Schnell"

    def test_detect_flux_variant_dev(self):
        """Test Flux variant detection for Dev."""
        result = _detect_flux_variant("flux-dev-v1.0")
        assert result == "Flux Dev"

    def test_detect_flux_variant_no_name(self):
        """Test Flux variant detection with no checkpoint name."""
        result = _detect_flux_variant(None)
        assert result == "Flux"


class TestLuminaDetection:
    """Test suite for Lumina model detection."""

    @patch('deforum.utils.model_detection._get_shared_module')
    def test_is_lumina_via_class_name(self, mock_get_shared):
        """Test Lumina detection via model class name."""
        mock_shared = SimpleNamespace()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "LuminaModel"
        mock_shared.sd_model = mock_model
        mock_get_shared.return_value = mock_shared

        assert is_lumina_model() is True

    @patch('deforum.utils.model_detection._get_shared_module')
    def test_is_lumina_via_diffusion_class(self, mock_get_shared):
        """Test Lumina detection via diffusion model class."""
        # Create actual class instances
        Lumina2NextDiT = type('Lumina2NextDiT', (), {})
        diff_model_instance = Lumina2NextDiT()

        OtherClass = type('OtherClass', (), {})
        mock_model = OtherClass()

        # Set up forge_objects structure
        mock_model.forge_objects = SimpleNamespace()
        mock_model.forge_objects.unet = SimpleNamespace()
        mock_model.forge_objects.unet.model = SimpleNamespace()
        mock_model.forge_objects.unet.model.diffusion_model = diff_model_instance

        mock_shared = SimpleNamespace()
        mock_shared.sd_model = mock_model
        mock_get_shared.return_value = mock_shared

        assert is_lumina_model() is True

    @patch('deforum.utils.model_detection._get_shared_module')
    def test_is_lumina_via_checkpoint_name_lumina(self, mock_get_shared):
        """Test Lumina detection via 'lumina' in checkpoint name."""
        mock_shared = SimpleNamespace()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "OtherModel"
        mock_shared.sd_model = mock_model
        mock_shared.opts = SimpleNamespace()
        mock_shared.opts.sd_model_checkpoint = "lumina-2.0.safetensors"
        mock_get_shared.return_value = mock_shared

        assert is_lumina_model() is True

    @patch('deforum.utils.model_detection._get_shared_module')
    def test_is_lumina_via_checkpoint_name_neta(self, mock_get_shared):
        """Test Lumina detection via 'neta' in checkpoint name."""
        mock_shared = SimpleNamespace()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "OtherModel"
        mock_shared.sd_model = mock_model
        mock_shared.opts = SimpleNamespace()
        mock_shared.opts.sd_model_checkpoint = "neta-lumina-v1.0.safetensors"
        mock_get_shared.return_value = mock_shared

        assert is_lumina_model() is True

    @patch('deforum.utils.model_detection._get_shared_module')
    def test_is_lumina_false_for_other_model(self, mock_get_shared):
        """Test Lumina detection returns False for other models."""
        mock_shared = SimpleNamespace()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "FluxModel"
        mock_shared.sd_model = mock_model
        mock_shared.opts = SimpleNamespace()
        mock_shared.opts.sd_model_checkpoint = "flux-dev.safetensors"
        mock_get_shared.return_value = mock_shared

        assert is_lumina_model() is False

    @patch('deforum.utils.model_detection._get_shared_module')
    def test_is_lumina_false_when_no_model(self, mock_get_shared):
        """Test Lumina detection returns False when no model loaded."""
        mock_shared = SimpleNamespace()
        mock_get_shared.return_value = mock_shared

        assert is_lumina_model() is False

    @patch('deforum.utils.model_detection._get_shared_module')
    def test_is_lumina_false_when_shared_none(self, mock_get_shared):
        """Test Lumina detection returns False when shared module unavailable."""
        mock_get_shared.return_value = None

        assert is_lumina_model() is False


class TestFluxDetection:
    """Test suite for Flux model detection."""

    @patch('deforum.utils.model_detection._get_shared_module')
    def test_is_flux_via_class_name(self, mock_get_shared):
        """Test Flux detection via model class name."""
        mock_shared = SimpleNamespace()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "FluxModel"
        mock_shared.sd_model = mock_model
        mock_get_shared.return_value = mock_shared

        assert is_flux_model() is True

    @patch('deforum.utils.model_detection._get_shared_module')
    def test_is_flux_via_checkpoint_name(self, mock_get_shared):
        """Test Flux detection via checkpoint name."""
        mock_shared = SimpleNamespace()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "OtherModel"
        mock_shared.sd_model = mock_model
        mock_shared.opts = SimpleNamespace()
        mock_shared.opts.sd_model_checkpoint = "flux-dev-v1.0.safetensors"
        mock_get_shared.return_value = mock_shared

        assert is_flux_model() is True

    @patch('deforum.utils.model_detection._get_shared_module')
    def test_is_flux_false_for_other_model(self, mock_get_shared):
        """Test Flux detection returns False for other models."""
        mock_shared = SimpleNamespace()
        mock_model = MagicMock()
        mock_model.__class__.__name__ = "LuminaModel"
        mock_shared.sd_model = mock_model
        mock_shared.opts = SimpleNamespace()
        mock_shared.opts.sd_model_checkpoint = "lumina.safetensors"
        mock_get_shared.return_value = mock_shared

        assert is_flux_model() is False

    @patch('deforum.utils.model_detection._get_shared_module')
    def test_is_flux_false_when_no_model(self, mock_get_shared):
        """Test Flux detection returns False when no model loaded."""
        mock_shared = SimpleNamespace()
        mock_get_shared.return_value = mock_shared

        assert is_flux_model() is False

    @patch('deforum.utils.model_detection._get_shared_module')
    def test_is_flux_false_when_shared_none(self, mock_get_shared):
        """Test Flux detection returns False when shared module unavailable."""
        mock_get_shared.return_value = None

        assert is_flux_model() is False


class TestModelName:
    """Test suite for get_model_name function."""

    @patch('deforum.utils.model_detection.is_lumina_model')
    def test_get_model_name_lumina(self, mock_is_lumina):
        """Test model name for Lumina."""
        mock_is_lumina.return_value = True
        assert get_model_name() == "Lumina 2.0"

    @patch('deforum.utils.model_detection.is_lumina_model')
    @patch('deforum.utils.model_detection.is_flux_model')
    @patch('deforum.utils.model_detection._get_shared_module')
    def test_get_model_name_flux_schnell(
        self, mock_get_shared, mock_is_flux, mock_is_lumina
    ):
        """Test model name for Flux Schnell."""
        mock_is_lumina.return_value = False
        mock_is_flux.return_value = True

        mock_shared = SimpleNamespace()
        mock_shared.opts = SimpleNamespace()
        mock_shared.opts.sd_model_checkpoint = "flux-schnell.safetensors"
        mock_get_shared.return_value = mock_shared

        assert get_model_name() == "Flux Schnell"

    @patch('deforum.utils.model_detection.is_lumina_model')
    @patch('deforum.utils.model_detection.is_flux_model')
    @patch('deforum.utils.model_detection._get_shared_module')
    def test_get_model_name_flux_dev(
        self, mock_get_shared, mock_is_flux, mock_is_lumina
    ):
        """Test model name for Flux Dev."""
        mock_is_lumina.return_value = False
        mock_is_flux.return_value = True

        mock_shared = SimpleNamespace()
        mock_shared.opts = SimpleNamespace()
        mock_shared.opts.sd_model_checkpoint = "flux-dev.safetensors"
        mock_get_shared.return_value = mock_shared

        assert get_model_name() == "Flux Dev"

    @patch('deforum.utils.model_detection.is_lumina_model')
    @patch('deforum.utils.model_detection.is_flux_model')
    def test_get_model_name_unknown(self, mock_is_flux, mock_is_lumina):
        """Test model name for unknown models."""
        mock_is_lumina.return_value = False
        mock_is_flux.return_value = False

        assert get_model_name() == "Unknown"


class TestRecommendedSettings:
    """Test suite for recommended settings functions."""

    @patch('deforum.utils.model_detection.is_lumina_model')
    @patch('deforum.utils.model_detection.is_flux_model')
    def test_get_recommended_cfg_scale_lumina(self, mock_is_flux, mock_is_lumina):
        """Test recommended CFG scale for Lumina."""
        mock_is_lumina.return_value = True
        mock_is_flux.return_value = False

        assert get_recommended_cfg_scale() == (4.0, 5.5)

    @patch('deforum.utils.model_detection.is_lumina_model')
    @patch('deforum.utils.model_detection.is_flux_model')
    def test_get_recommended_cfg_scale_flux(self, mock_is_flux, mock_is_lumina):
        """Test recommended CFG scale for Flux."""
        mock_is_lumina.return_value = False
        mock_is_flux.return_value = True

        assert get_recommended_cfg_scale() == (1.0, 3.5)

    @patch('deforum.utils.model_detection.is_lumina_model')
    @patch('deforum.utils.model_detection.is_flux_model')
    def test_get_recommended_cfg_scale_default(self, mock_is_flux, mock_is_lumina):
        """Test recommended CFG scale for unknown models."""
        mock_is_lumina.return_value = False
        mock_is_flux.return_value = False

        assert get_recommended_cfg_scale() == (7.0, 12.0)

    @patch('deforum.utils.model_detection.is_lumina_model')
    @patch('deforum.utils.model_detection.is_flux_model')
    def test_get_recommended_steps_lumina(self, mock_is_flux, mock_is_lumina):
        """Test recommended steps for Lumina."""
        mock_is_lumina.return_value = True
        mock_is_flux.return_value = False

        assert get_recommended_steps() == 30

    @patch('deforum.utils.model_detection.is_lumina_model')
    @patch('deforum.utils.model_detection.is_flux_model')
    def test_get_recommended_steps_flux(self, mock_is_flux, mock_is_lumina):
        """Test recommended steps for Flux."""
        mock_is_lumina.return_value = False
        mock_is_flux.return_value = True

        assert get_recommended_steps() == 20

    @patch('deforum.utils.model_detection.is_lumina_model')
    @patch('deforum.utils.model_detection.is_flux_model')
    def test_get_recommended_steps_default(self, mock_is_flux, mock_is_lumina):
        """Test recommended steps for unknown models."""
        mock_is_lumina.return_value = False
        mock_is_flux.return_value = False

        assert get_recommended_steps() == 20
