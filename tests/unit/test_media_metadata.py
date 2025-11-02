"""Unit tests for deforum.media.metadata module."""

import base64
import json
import pytest
import sys
import importlib.util
from pathlib import Path

# Import metadata module directly bypassing package __init__.py
# (avoids heavy dependencies in deforum.media.__init__.py)
metadata_path = Path(__file__).parent.parent.parent / "deforum" / "media" / "metadata.py"
spec = importlib.util.spec_from_file_location("metadata", metadata_path)
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)

# Import specific functions and constants
encode_settings_for_metadata = metadata.encode_settings_for_metadata
decode_settings_from_metadata = metadata.decode_settings_from_metadata
create_essential_metadata = metadata.create_essential_metadata
create_ffmpeg_metadata_args = metadata.create_ffmpeg_metadata_args
METADATA_PREFIX = metadata.METADATA_PREFIX
METADATA_VERSION = metadata.METADATA_VERSION


class TestEncodeDecodeSettings:
    """Test settings encoding and decoding."""

    def test_encode_adds_prefix(self):
        """Encoded string should start with DEFORUM_SETTINGS: prefix."""
        settings = {"test": "value"}
        encoded = encode_settings_for_metadata(settings)
        assert encoded.startswith(METADATA_PREFIX)

    def test_encode_includes_version(self):
        """Encoded metadata should include version field."""
        settings = {"test": "value"}
        encoded = encode_settings_for_metadata(settings)

        # Decode to check structure
        decoded = decode_settings_from_metadata(encoded)
        assert "version" in decoded
        assert decoded["version"] == METADATA_VERSION

    def test_encode_preserves_settings(self):
        """Original settings should be preserved in encoded data."""
        settings = {"fps": 24, "max_frames": 100, "seed": 12345}
        encoded = encode_settings_for_metadata(settings)

        decoded = decode_settings_from_metadata(encoded)
        assert decoded["settings"] == settings

    def test_encode_handles_complex_data(self):
        """Encoding should handle nested dictionaries and various types."""
        settings = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"}
        }
        encoded = encode_settings_for_metadata(settings)
        decoded = decode_settings_from_metadata(encoded)
        assert decoded["settings"] == settings

    def test_decode_requires_prefix(self):
        """Decoding should fail without correct prefix."""
        with pytest.raises(ValueError, match="Not a Deforum settings string"):
            decode_settings_from_metadata("INVALID_PREFIX:data")

    def test_decode_requires_valid_base64(self):
        """Decoding should fail with invalid base64."""
        with pytest.raises(Exception):  # base64.binascii.Error or similar
            decode_settings_from_metadata(f"{METADATA_PREFIX}not_valid_base64!!!")

    def test_decode_requires_valid_json(self):
        """Decoding should fail with invalid JSON."""
        invalid_json = base64.b64encode(b"not json").decode('ascii')
        with pytest.raises(json.JSONDecodeError):
            decode_settings_from_metadata(f"{METADATA_PREFIX}{invalid_json}")

    def test_decode_requires_version_field(self):
        """Decoded data must have version field."""
        # Create metadata without version
        invalid_metadata = {"settings": {"test": "value"}}  # Missing "version"
        encoded_json = json.dumps(invalid_metadata)
        encoded_b64 = base64.b64encode(encoded_json.encode('utf-8')).decode('ascii')

        with pytest.raises(ValueError, match="Invalid metadata structure"):
            decode_settings_from_metadata(f"{METADATA_PREFIX}{encoded_b64}")

    def test_roundtrip_preserves_data(self):
        """Encode then decode should return original data."""
        original = {
            "render_mode": "New 3D",
            "fps": 60,
            "max_frames": 240,
            "seed": 987654321,
            "model": "flux1-dev-bnb-nf4-v2"
        }

        encoded = encode_settings_for_metadata(original)
        decoded = decode_settings_from_metadata(encoded)

        assert decoded["settings"] == original


class TestCreateEssentialMetadata:
    """Test essential metadata creation."""

    def test_creates_all_required_fields(self):
        """Should create metadata with all required fields."""
        metadata = create_essential_metadata(
            render_mode="New 3D",
            fps=24,
            max_frames=100,
            width=1024,
            height=576,
            seed=12345,
            steps=20,
            cfg_scale=7.5
        )

        required_fields = [
            "commit_id", "render_mode", "model", "scheduler",
            "steps", "cfg_scale", "seed", "fps", "total_frames", "width", "height"
        ]

        for field in required_fields:
            assert field in metadata

    def test_stores_width_height_separately(self):
        """Width and height should be stored as separate fields."""
        metadata = create_essential_metadata(
            render_mode="New 3D",
            fps=24,
            max_frames=100,
            width=1920,
            height=1080,
            seed=12345,
            steps=20,
            cfg_scale=7.5
        )

        assert metadata["width"] == 1920
        assert metadata["height"] == 1080

    def test_uses_provided_model_name(self):
        """Should use provided model name."""
        metadata = create_essential_metadata(
            render_mode="New 3D",
            fps=24,
            max_frames=100,
            width=1024,
            height=576,
            seed=12345,
            steps=20,
            cfg_scale=7.5,
            model_name="Flux1-Dev-Bnb-Nf4"
        )

        assert metadata["model"] == "Flux1-Dev-Bnb-Nf4"

    def test_defaults_to_unknown_model_and_scheduler(self):
        """Should default to 'Unknown' for model and scheduler if not provided."""
        metadata = create_essential_metadata(
            render_mode="New 3D",
            fps=24,
            max_frames=100,
            width=1024,
            height=576,
            seed=12345,
            steps=20,
            cfg_scale=7.5
        )

        assert metadata["model"] == "Unknown"
        assert metadata["scheduler"] == "Unknown"

    def test_includes_prompts_when_provided(self):
        """Should include prompts dict when provided."""
        prompts = {0: "a cat", 50: "a dog", 100: "a bird"}
        metadata = create_essential_metadata(
            render_mode="New 3D",
            fps=24,
            max_frames=100,
            width=1024,
            height=576,
            seed=12345,
            steps=20,
            cfg_scale=7.5,
            prompts=prompts
        )

        assert "prompts" in metadata
        assert metadata["prompts"] == prompts

    def test_omits_prompts_when_not_provided(self):
        """Should not include prompts field when not provided."""
        metadata = create_essential_metadata(
            render_mode="New 3D",
            fps=24,
            max_frames=100,
            width=1024,
            height=576,
            seed=12345,
            steps=20,
            cfg_scale=7.5
        )

        assert "prompts" not in metadata


class TestCreateFFmpegMetadataArgs:
    """Test ffmpeg metadata argument generation."""

    def test_returns_list_of_strings(self):
        """Should return list of strings."""
        settings = {"fps": 24}
        args = create_ffmpeg_metadata_args(settings)

        assert isinstance(args, list)
        assert all(isinstance(arg, str) for arg in args)

    def test_includes_metadata_flags(self):
        """Should include -metadata flags."""
        settings = {"fps": 24}
        args = create_ffmpeg_metadata_args(settings)

        assert '-metadata' in args

    def test_includes_comment_with_encoded_settings(self):
        """Comment field should contain encoded settings."""
        settings = {"fps": 24, "seed": 12345}
        args = create_ffmpeg_metadata_args(settings)

        # Find comment field
        comment_value = None
        for i, arg in enumerate(args):
            if arg == '-metadata' and i + 1 < len(args):
                if args[i + 1].startswith('comment='):
                    comment_value = args[i + 1][8:]  # Remove 'comment=' prefix
                    break

        assert comment_value is not None
        assert comment_value.startswith(METADATA_PREFIX)

        # Verify it decodes correctly
        decoded = decode_settings_from_metadata(comment_value)
        assert decoded["settings"]["fps"] == 24
        assert decoded["settings"]["seed"] == 12345

    def test_only_embeds_comment_field(self):
        """Should only embed settings in comment field, no branding."""
        settings = {"fps": 24, "seed": 12345}
        args = create_ffmpeg_metadata_args(settings)

        # Should only have one -metadata flag (for comment)
        metadata_count = args.count('-metadata')
        assert metadata_count == 1

        # Should not include title, artist, copyright, encoder
        args_str = ' '.join(args)
        assert 'title=' not in args_str
        assert 'artist=' not in args_str
        assert 'copyright=' not in args_str
        assert 'encoder=' not in args_str
        assert 'description=' not in args_str

    def test_args_alternate_flag_and_value(self):
        """Arguments should alternate between -metadata and value."""
        settings = {"fps": 24}
        args = create_ffmpeg_metadata_args(settings)

        # Every odd index should be '-metadata'
        metadata_indices = [i for i, arg in enumerate(args) if arg == '-metadata']

        # Each -metadata should be followed by a value
        for idx in metadata_indices:
            assert idx + 1 < len(args)
            assert not args[idx + 1].startswith('-')  # Value shouldn't start with flag


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_full_workflow(self):
        """Test complete workflow: create settings, encode, create args, decode."""
        # Create settings with prompts
        prompts = {0: "scene A", 50: "scene B", 100: "scene C"}
        essential = create_essential_metadata(
            render_mode="Keyframes Only",
            fps=24,
            max_frames=100,
            width=1024,
            height=576,
            seed=999,
            steps=20,
            cfg_scale=7.0,
            model_name="TestModel",
            scheduler="euler_a",
            prompts=prompts
        )

        # Create ffmpeg args
        args = create_ffmpeg_metadata_args(essential)

        # Extract comment value
        comment_value = None
        for i, arg in enumerate(args):
            if arg == '-metadata' and i + 1 < len(args):
                if args[i + 1].startswith('comment='):
                    comment_value = args[i + 1][8:]
                    break

        # Decode and verify all fields
        decoded = decode_settings_from_metadata(comment_value)
        settings = decoded["settings"]
        assert settings["render_mode"] == "Keyframes Only"
        assert settings["fps"] == 24
        assert settings["seed"] == 999
        assert settings["model"] == "TestModel"
        assert settings["scheduler"] == "euler_a"
        assert settings["steps"] == 20
        assert settings["cfg_scale"] == 7.0
        assert settings["width"] == 1024
        assert settings["height"] == 576
        assert settings["total_frames"] == 100
        # Note: JSON converts int dict keys to strings during serialization
        assert settings["prompts"] == {str(k): v for k, v in prompts.items()}
