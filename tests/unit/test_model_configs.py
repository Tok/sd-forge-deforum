"""Unit tests for model configuration and validation system.

Tests model detection, configuration retrieval, and settings validation.
"""

import pytest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

# Add extension root to path to import model_configs directly (avoiding Forge deps)
extension_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(extension_root))

# Mock the logger module BEFORE importing model_configs
# This prevents ImportError when model_configs tries to import get_logger
mock_logger_module = Mock()
mock_logger_instance = Mock()
mock_logger_module.get_logger.return_value = mock_logger_instance
sys.modules['deforum.utils.system.logging'] = mock_logger_module

# Import directly from model_configs module (not through deforum.config package)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "model_configs",
    extension_root / "deforum" / "config" / "model_configs.py"
)
model_configs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_configs)

# Extract what we need
detect_model_type_extended = model_configs.detect_model_type_extended
get_model_config = model_configs.get_model_config
validate_settings = model_configs.validate_settings
MODEL_CONFIGS = model_configs.MODEL_CONFIGS


class TestModelDetection:
    """Test model type detection from filenames."""

    def test_detect_flux_dev(self):
        """Test Flux Dev detection."""
        assert detect_model_type_extended("Flux/flux1-dev-bnb-nf4-v2.safetensors") == "flux_dev"
        assert detect_model_type_extended("flux-dev.safetensors") == "flux_dev"
        assert detect_model_type_extended("FLUX-DEV.ckpt") == "flux_dev"

    def test_detect_flux_schnell(self):
        """Test Flux Schnell detection (more specific than dev)."""
        assert detect_model_type_extended("flux1-schnell.safetensors") == "flux_schnell"
        assert detect_model_type_extended("Flux/FLUX-SCHNELL.ckpt") == "flux_schnell"

    def test_detect_lumina(self):
        """Test Lumina 2.0 detection."""
        assert detect_model_type_extended("neta-lumina-v1.0.safetensors") == "lumina"
        assert detect_model_type_extended("Lumina/model.ckpt") == "lumina"
        assert detect_model_type_extended("neta-art-model.safetensors") == "lumina"

    def test_detect_z_image(self):
        """Test Z-Image-Turbo detection."""
        assert detect_model_type_extended("Z-Image/model.safetensors") == "z_image"
        assert detect_model_type_extended("zimage-turbo.ckpt") == "z_image"
        assert detect_model_type_extended("stabilityai/z-image.safetensors") == "z_image"

    def test_detect_sdxl(self):
        """Test SDXL detection."""
        assert detect_model_type_extended("sdxl-base.safetensors") == "sdxl"
        assert detect_model_type_extended("stable-diffusion-xl.ckpt") == "sdxl"
        assert detect_model_type_extended("SDXL-1.0.safetensors") == "sdxl"
        assert detect_model_type_extended("sd_xl_base.ckpt") == "sdxl"

    def test_detect_sd15(self):
        """Test SD 1.5 detection."""
        assert detect_model_type_extended("sd15-model.ckpt") == "sd15"
        assert detect_model_type_extended("sd-1.5.safetensors") == "sd15"
        assert detect_model_type_extended("SD1.5.ckpt") == "sd15"

    def test_detect_unknown_model(self):
        """Test unknown model handling."""
        assert detect_model_type_extended("unknown-model.ckpt") == "unknown"

    def test_detect_empty_string(self):
        """Test that empty string returns unknown."""
        assert detect_model_type_extended("") == "unknown"

    def test_case_insensitive(self):
        """Test that detection is case-insensitive."""
        assert detect_model_type_extended("FLUX-DEV.SAFETENSORS") == "flux_dev"
        assert detect_model_type_extended("lumina.CKPT") == "lumina"


class TestModelConfigRetrieval:
    """Test retrieving model configurations."""

    def test_get_flux_dev_config(self):
        """Test retrieving Flux Dev configuration."""
        config = get_model_config("flux-dev.safetensors")

        assert config.model_type == "flux_dev"
        assert config.display_name == "Flux.1 Dev"
        assert config.recommended_steps == 20
        assert config.uses_distilled_cfg is True
        assert config.distilled_cfg_scale_default == 3.5
        assert config.uses_cfg is False

    def test_get_lumina_config(self):
        """Test retrieving Lumina configuration."""
        config = get_model_config("lumina-model.ckpt")

        assert config.model_type == "lumina"
        assert config.display_name == "Lumina 2.0"
        assert config.recommended_steps == 30
        assert config.uses_cfg is True
        assert config.cfg_scale_default == 5.0
        assert config.uses_distilled_cfg is False

    def test_get_z_image_config(self):
        """Test retrieving Z-Image-Turbo configuration."""
        config = get_model_config("z-image.safetensors")

        assert config.model_type == "z_image"
        assert config.display_name == "Z-Image-Turbo"
        assert config.recommended_steps == 9
        assert config.uses_cfg is True
        assert config.cfg_scale_default == 2.0
        assert config.uses_distilled_cfg is False

    def test_get_sdxl_config(self):
        """Test retrieving SDXL configuration."""
        config = get_model_config("sdxl-base.safetensors")

        assert config.model_type == "sdxl"
        assert config.display_name == "SDXL"
        assert config.recommended_steps == 25
        assert config.uses_cfg is True
        assert config.cfg_scale_default == 7.5
        assert config.uses_distilled_cfg is False

    def test_get_sd15_config(self):
        """Test retrieving SD 1.5 configuration."""
        config = get_model_config("sd15-model.ckpt")

        assert config.model_type == "sd15"
        assert config.display_name == "SD 1.5"
        assert config.recommended_steps == 25
        assert config.uses_cfg is True
        assert config.cfg_scale_default == 7.5
        assert config.uses_distilled_cfg is False

    def test_all_configs_exist(self):
        """Test that all model types have configs."""
        required_types = ["flux_dev", "flux_schnell", "lumina", "z_image", "sdxl", "sd15", "unknown"]

        for model_type in required_types:
            assert model_type in MODEL_CONFIGS


class TestSettingsValidation:
    """Test settings validation against model configs."""

    def test_flux_with_correct_settings(self):
        """Test Flux with optimal settings (no warnings)."""
        args = SimpleNamespace(
            steps=20,
            cfg_scale=1.0,
            sampler='euler',
            sampler_schedule_type='simple',
            anim_args=SimpleNamespace(
                distilled_cfg_scale_schedule='0: (3.5)'
            )
        )

        warnings = validate_settings(args, "flux-dev.safetensors")

        # Should have no warnings (all optimal)
        assert len(warnings) == 0

    def test_flux_with_wrong_cfg(self):
        """Test Flux with traditional CFG (should warn)."""
        args = SimpleNamespace(
            steps=20,
            cfg_scale=7.5,  # Wrong! Flux ignores this
            sampler='euler',
            sampler_schedule_type='simple',
            anim_args=SimpleNamespace(
                distilled_cfg_scale_schedule='0: (3.5)'
            )
        )

        warnings = validate_settings(args, "flux-dev.safetensors")

        # Should warn about CFG being ignored
        assert len(warnings) > 0
        assert any("ignores traditional CFG" in w for w in warnings)

    def test_z_image_with_flux_settings(self):
        """Test Z-Image with Flux distilled_cfg (should warn when non-default)."""
        args = SimpleNamespace(
            steps=4,
            cfg_scale=2.0,  # Correct for Z-Image
            sampler='euler',
            sampler_schedule_type='simple',
            anim_args=SimpleNamespace(
                distilled_cfg_scale_schedule='0: (7.0)'  # Non-default! Z-Image doesn't use this
            )
        )

        warnings = validate_settings(args, "z-image.safetensors")

        # Should warn about distilled_cfg being ignored (when changed from default)
        assert len(warnings) > 0
        assert any("ignores distilled CFG" in w for w in warnings)

    def test_steps_too_low(self):
        """Test warning for steps below minimum."""
        args = SimpleNamespace(
            steps=2,  # Too low for Flux Dev (min 8)
            cfg_scale=1.0,
            sampler='euler',
            sampler_schedule_type='simple',
            anim_args=SimpleNamespace(
                distilled_cfg_scale_schedule='0: (3.5)'
            )
        )

        warnings = validate_settings(args, "flux-dev.safetensors")

        # Should warn about low steps
        assert len(warnings) > 0
        assert any("Steps too low" in w for w in warnings)

    def test_steps_too_high(self):
        """Test warning for steps above maximum."""
        args = SimpleNamespace(
            steps=100,  # Too high for Flux Dev (max 50)
            cfg_scale=1.0,
            sampler='euler',
            sampler_schedule_type='simple',
            anim_args=SimpleNamespace(
                distilled_cfg_scale_schedule='0: (3.5)'
            )
        )

        warnings = validate_settings(args, "flux-dev.safetensors")

        # Should warn about wasted computation
        assert len(warnings) > 0
        assert any("Steps too high" in w for w in warnings)

    def test_incompatible_scheduler(self):
        """Test warning for incompatible scheduler."""
        args = SimpleNamespace(
            steps=20,
            cfg_scale=1.0,
            sampler='euler',
            sampler_schedule_type='linear_quadratic',  # Wrong for Flux
            anim_args=SimpleNamespace(
                distilled_cfg_scale_schedule='0: (3.5)'
            )
        )

        warnings = validate_settings(args, "flux-dev.safetensors")

        # Should warn about scheduler
        assert len(warnings) > 0
        assert any("Scheduler" in w for w in warnings)

    def test_sampler_case_insensitive(self):
        """Test that sampler validation is case-insensitive."""
        args = SimpleNamespace(
            steps=4,
            cfg_scale=2.0,
            sampler='Euler',  # Capitalized should match 'euler'
            sampler_schedule_type='simple',
            anim_args=None
        )

        warnings = validate_settings(args, "z-image.safetensors")

        # Should NOT warn about sampler (case-insensitive match)
        sampler_warnings = [w for w in warnings if "Sampler" in w]
        assert len(sampler_warnings) == 0

    def test_cfg_scale_out_of_range(self):
        """Test warning for CFG scale outside recommended range."""
        args = SimpleNamespace(
            steps=30,
            cfg_scale=10.0,  # Too high for Lumina (max 5.5)
            sampler='euler',
            sampler_schedule_type='linear_quadratic',
            anim_args=None
        )

        warnings = validate_settings(args, "lumina.ckpt")

        # Should warn about CFG range
        assert len(warnings) > 0
        assert any("CFG scale out of range" in w for w in warnings)

    def test_lumina_with_optimal_settings(self):
        """Test Lumina with optimal settings."""
        args = SimpleNamespace(
            steps=30,
            cfg_scale=5.0,
            sampler='euler',
            sampler_schedule_type='linear_quadratic',
            anim_args=SimpleNamespace(
                distilled_cfg_scale_schedule='0: (3.5)'  # Ignored but not warned if default
            )
        )

        warnings = validate_settings(args, "lumina-model.ckpt")

        # Should have no warnings
        assert len(warnings) == 0


class TestConfigParameters:
    """Test specific config parameter values."""

    def test_flux_schnell_steps(self):
        """Test Flux Schnell has low step count."""
        config = MODEL_CONFIGS["flux_schnell"]

        assert config.recommended_steps == 4
        assert config.min_steps == 1
        assert config.max_steps == 8

    def test_lumina_scheduler(self):
        """Test Lumina requires linear_quadratic scheduler."""
        config = MODEL_CONFIGS["lumina"]

        assert config.recommended_scheduler == "linear_quadratic"
        assert "linear_quadratic" in config.compatible_schedulers

    def test_all_configs_have_required_fields(self):
        """Test all configs have required fields."""
        required_fields = [
            'model_type', 'display_name', 'recommended_steps',
            'uses_cfg', 'uses_distilled_cfg', 'recommended_scheduler',
            'recommended_sampler', 'notes'
        ]

        for model_type, config in MODEL_CONFIGS.items():
            for field in required_fields:
                assert hasattr(config, field), f"{model_type} missing {field}"

    def test_step_ranges_valid(self):
        """Test all configs have valid step ranges."""
        for model_type, config in MODEL_CONFIGS.items():
            assert config.min_steps <= config.recommended_steps <= config.max_steps, \
                f"{model_type} has invalid step range"

    def test_cfg_ranges_valid(self):
        """Test all configs have valid CFG ranges."""
        for model_type, config in MODEL_CONFIGS.items():
            if config.uses_cfg:
                assert config.cfg_scale_min <= config.cfg_scale_default <= config.cfg_scale_max, \
                    f"{model_type} has invalid CFG range"

            if config.uses_distilled_cfg:
                assert config.distilled_cfg_scale_min <= config.distilled_cfg_scale_default <= config.distilled_cfg_scale_max, \
                    f"{model_type} has invalid distilled CFG range"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
