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
create_comprehensive_metadata = metadata.create_comprehensive_metadata
create_ffmpeg_metadata_args = metadata.create_ffmpeg_metadata_args
METADATA_PREFIX = metadata.METADATA_PREFIX
METADATA_VERSION = metadata.METADATA_VERSION

# Constants used in tests (fallback values for when general.py can't be imported)
GITHUB_URL = "https://github.com/Tok/sd-forge-deforum"
FORK_NAME = "Zirteq's Fluxabled Fork of the Deforum Extension for Forge Neo Fork of Forge WebUI Fork of Automatic1111"


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


class TestCreateComprehensiveMetadata:
    """Test comprehensive metadata creation."""

    def test_adds_commit_id_and_github_url(self):
        """Should add commit_id and github_url to settings."""
        settings = {
            "fps": 24,
            "seed": 12345,
        }
        metadata = create_comprehensive_metadata(settings)

        assert "commit_id" in metadata
        assert "github_url" in metadata
        assert metadata["github_url"] == GITHUB_URL

    def test_preserves_all_provided_settings(self):
        """Should preserve all settings provided."""
        settings = {
            "fps": 24,
            "seed": 12345,
            "W": 1280,
            "H": 720,
            "steps": 20,
            "cfg_scale": 7.5,
            "distilled_cfg_scale": 3.5,
        }
        metadata = create_comprehensive_metadata(settings)

        # All original settings should be preserved
        for key, value in settings.items():
            assert metadata[key] == value


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

    def test_embeds_both_encoded_and_readable_fields(self):
        """Should embed both base64-encoded comment and human-readable fields."""
        settings = {
            "fps": 24,
            "seed": 12345,
            "W": 1280,
            "H": 720,
            "github_url": "https://github.com/Tok/sd-forge-deforum",
            "commit_id": "abc123",
        }
        args = create_ffmpeg_metadata_args(settings)

        # Should have multiple -metadata flags (comment + readable fields)
        metadata_count = args.count('-metadata')
        assert metadata_count > 1  # At least comment + some readable fields

        args_str = ' '.join(args)

        # Should include encoded comment
        assert 'comment=' in args_str
        assert METADATA_PREFIX in args_str

        # Should include human-readable fields with deforum_ prefix
        assert 'deforum_fps=24' in args_str
        assert 'deforum_seed=12345' in args_str
        assert 'deforum_resolution=1280x720' in args_str
        assert 'deforum_github=https://github.com/Tok/sd-forge-deforum' in args_str
        assert 'deforum_commit=abc123' in args_str

        # Should NOT include branding fields
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
        # Create comprehensive settings
        settings = {
            "render_mode": "Keyframes Only",
            "fps": 24,
            "max_frames": 100,
            "W": 1024,
            "H": 576,
            "seed": 999,
            "steps": 20,
            "cfg_scale": 7.0,
            "sd_model_checkpoint": "TestModel",
            "scheduler": "euler_a",
            "animation_prompts": {0: "scene A", 50: "scene B", 100: "scene C"},
        }
        comprehensive = create_comprehensive_metadata(settings)

        # Create ffmpeg args
        args = create_ffmpeg_metadata_args(comprehensive)

        # Extract comment value
        comment_value = None
        for i, arg in enumerate(args):
            if arg == '-metadata' and i + 1 < len(args):
                if args[i + 1].startswith('comment='):
                    comment_value = args[i + 1][8:]
                    break

        # Decode and verify all fields from base64 comment
        decoded = decode_settings_from_metadata(comment_value)
        settings_decoded = decoded["settings"]
        assert settings_decoded["render_mode"] == "Keyframes Only"
        assert settings_decoded["fps"] == 24
        assert settings_decoded["seed"] == 999
        assert settings_decoded["sd_model_checkpoint"] == "TestModel"
        assert settings_decoded["scheduler"] == "euler_a"
        assert settings_decoded["steps"] == 20
        assert settings_decoded["cfg_scale"] == 7.0
        assert settings_decoded["W"] == 1024
        assert settings_decoded["H"] == 576
        assert settings_decoded["max_frames"] == 100
        assert settings_decoded["commit_id"]  # Should be present
        assert settings_decoded["github_url"] == GITHUB_URL

        # Verify human-readable metadata fields are also present
        args_str = ' '.join(args)
        assert 'deforum_resolution=1024x576' in args_str
        assert 'deforum_fps=24' in args_str
        assert 'deforum_seed=999' in args_str
        assert 'deforum_model=TestModel' in args_str
        assert f'deforum_github={GITHUB_URL}' in args_str
