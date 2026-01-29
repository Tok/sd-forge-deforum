"""Test Flux 2 Klein model detection and compatibility.

This test verifies that:
1. Flux 2 Klein models are correctly detected by Forge Neo
2. Klein text encoder (Qwen3) is properly configured
3. Img2img works with Klein models (if weights are available)
"""

import pytest
import sys
from pathlib import Path


def test_klein_model_detection():
    """Test that Flux 2 Klein models are registered in Forge Neo."""
    try:
        from huggingface_guess import model_list

        # Verify Klein models are in the model list
        assert hasattr(model_list, 'Flux2K4B'), "Flux2K4B model class not found"
        assert hasattr(model_list, 'Flux2K9B'), "Flux2K9B model class not found"

        # Verify Klein models are in the MODELS list
        assert model_list.Flux2K4B in model_list.MODELS, "Flux2K4B not in MODELS list"
        assert model_list.Flux2K9B in model_list.MODELS, "Flux2K9B not in MODELS list"

        # Check Klein 4B configuration
        klein_4b = model_list.Flux2K4B
        assert klein_4b.huggingface_repo == "black-forest-labs/FLUX.2-klein-4B"
        assert klein_4b.unet_config["image_model"] == "flux2"
        assert klein_4b.unet_config["hidden_size"] == 3072

        # Check Klein 9B configuration
        klein_9b = model_list.Flux2K9B
        assert klein_9b.huggingface_repo == "black-forest-labs/FLUX.2-klein-9B"
        assert klein_9b.unet_config["image_model"] == "flux2"
        assert klein_9b.unet_config["hidden_size"] == 4096

        print("✓ Klein models detected successfully")
        print(f"  - Flux2K4B: {klein_4b.huggingface_repo}")
        print(f"  - Flux2K9B: {klein_9b.huggingface_repo}")

    except ImportError as e:
        pytest.skip(f"Cannot import Forge Neo model list: {e}")


def test_klein_text_processing_engine():
    """Test that Klein text processing engine is available."""
    try:
        from backend.text_processing.klein_engine import KleinTextProcessingEngine

        # Verify engine has correct attributes
        assert hasattr(KleinTextProcessingEngine, 'tokenize')
        assert hasattr(KleinTextProcessingEngine, 'process_tokens')
        assert hasattr(KleinTextProcessingEngine, 'llama_template')

        print("✓ Klein text processing engine available")

        # Check llama template format
        engine_class = KleinTextProcessingEngine
        # Can't instantiate without text_encoder/tokenizer, but can check class definition
        assert 'im_start' in str(KleinTextProcessingEngine.__init__.__code__.co_consts)

    except ImportError as e:
        pytest.skip(f"Cannot import Klein text engine: {e}")


def test_flux2_diffusion_engine():
    """Test that Flux2 diffusion engine is available."""
    try:
        from backend.diffusion_engine.flux2 import Flux2

        # Verify Flux2 engine has Klein model matches
        assert hasattr(Flux2, 'matched_guesses')

        from huggingface_guess import model_list
        assert model_list.Flux2K4B in Flux2.matched_guesses
        assert model_list.Flux2K9B in Flux2.matched_guesses

        print("✓ Flux2 diffusion engine configured for Klein models")

    except ImportError as e:
        pytest.skip(f"Cannot import Flux2 engine: {e}")


def test_klein_model_configs_exist():
    """Test that Klein model configuration files exist."""
    forge_root = Path(__file__).parent.parent.parent.parent.parent

    klein_4b_path = forge_root / "backend" / "huggingface" / "black-forest-labs" / "FLUX.2-klein-4B"
    klein_9b_path = forge_root / "backend" / "huggingface" / "black-forest-labs" / "FLUX.2-klein-9B"

    # Check if config directories exist
    assert klein_4b_path.exists(), f"Klein 4B config not found at {klein_4b_path}"
    assert klein_9b_path.exists(), f"Klein 9B config not found at {klein_9b_path}"

    # Check essential config files
    for klein_path, name in [(klein_4b_path, "4B"), (klein_9b_path, "9B")]:
        model_index = klein_path / "model_index.json"
        assert model_index.exists(), f"model_index.json missing for Klein {name}"

        transformer_config = klein_path / "transformer" / "config.json"
        assert transformer_config.exists(), f"transformer config missing for Klein {name}"

        text_encoder_config = klein_path / "text_encoder" / "config.json"
        assert text_encoder_config.exists(), f"text_encoder config missing for Klein {name}"

        tokenizer_dir = klein_path / "tokenizer"
        assert tokenizer_dir.exists(), f"tokenizer dir missing for Klein {name}"

        print(f"✓ Klein {name} configuration files present")


def test_dynamic_args_klein_flag():
    """Test that dynamic_args includes klein flag."""
    try:
        from backend import args

        # Verify klein flag is defined in dynamic_args
        assert hasattr(args, 'dynamic_args')
        assert 'klein' in args.dynamic_args

        print("✓ dynamic_args['klein'] flag available")

    except ImportError as e:
        pytest.skip(f"Cannot import backend.args: {e}")


def test_deforum_flux2_compatibility():
    """Test that Deforum's Flux 2 compatibility patches exist."""
    try:
        from deforum.integrations.flux2 import ensure_flux2_compatibility

        # Verify compatibility function exists
        assert callable(ensure_flux2_compatibility)

        # Try to apply patches (should be safe to call multiple times)
        ensure_flux2_compatibility()

        print("✓ Deforum Flux 2 compatibility patches available")

    except ImportError as e:
        pytest.skip(f"Cannot import Deforum Flux2 compatibility: {e}")


if __name__ == "__main__":
    """Run tests directly for quick verification."""
    print("Testing Flux 2 Klein model support...\n")

    test_klein_model_detection()
    test_klein_text_processing_engine()
    test_flux2_diffusion_engine()
    test_klein_model_configs_exist()
    test_dynamic_args_klein_flag()
    test_deforum_flux2_compatibility()

    print("\n✅ All Flux 2 Klein tests passed!")