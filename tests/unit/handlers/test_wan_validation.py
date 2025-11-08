"""Unit tests for Wan validation helpers.

Tests the helper functions extracted from validate_wan_generation().
"""

import json
import pytest
import sys
from pathlib import Path

# Add extension root to path
extension_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(extension_root))

# Import directly from module file, bypassing deforum.ui.__init__.py
# This avoids cascade of Forge module imports
import importlib.util
spec = importlib.util.spec_from_file_location(
    "wan_validation",
    extension_root / "deforum" / "ui" / "handlers" / "wan_validation.py"
)
wan_validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wan_validation)

_is_empty_prompts = wan_validation._is_empty_prompts
_has_placeholder_text = wan_validation._has_placeholder_text
_parse_prompts_json = wan_validation._parse_prompts_json
_has_default_prompts = wan_validation._has_default_prompts
_build_validation_message = wan_validation._build_validation_message


class TestIsEmptyPrompts:
    """Test _is_empty_prompts() function."""

    def test_empty_string(self):
        """Should return True for empty string."""
        assert _is_empty_prompts("") is True

    def test_whitespace_only(self):
        """Should return True for whitespace only."""
        assert _is_empty_prompts("   ") is True
        assert _is_empty_prompts("\n\t  ") is True

    def test_none_value(self):
        """Should return True for None."""
        assert _is_empty_prompts(None) is True

    def test_valid_content(self):
        """Should return False for non-empty content."""
        assert _is_empty_prompts("some content") is False
        assert _is_empty_prompts('{"0": "prompt"}') is False


class TestHasPlaceholderText:
    """Test _has_placeholder_text() function."""

    def test_required_placeholder(self):
        """Should detect 'required:' placeholder."""
        assert _has_placeholder_text("required: load prompts") is True
        assert _has_placeholder_text("REQUIRED: Load prompts") is True

    def test_load_prompts_placeholder(self):
        """Should detect 'load prompts' placeholder."""
        assert _has_placeholder_text("Please load prompts first") is True
        assert _has_placeholder_text("LOAD PROMPTS now") is True

    def test_placeholder_keyword(self):
        """Should detect 'placeholder' keyword."""
        assert _has_placeholder_text("This is a placeholder") is True
        assert _has_placeholder_text("PLACEHOLDER text") is True

    def test_no_placeholder(self):
        """Should return False for normal prompts."""
        assert _has_placeholder_text('{"0": "a beautiful landscape"}') is False
        assert _has_placeholder_text("Regular prompt text") is False


class TestParsePromptsJson:
    """Test _parse_prompts_json() function."""

    def test_valid_json(self):
        """Should successfully parse valid JSON."""
        prompts = '{"0": "prompt one", "60": "prompt two"}'
        is_valid, prompts_dict, error = _parse_prompts_json(prompts)

        assert is_valid is True
        assert prompts_dict == {"0": "prompt one", "60": "prompt two"}
        assert error == ""

    def test_empty_json_object(self):
        """Should detect empty JSON object."""
        prompts = "{}"
        is_valid, prompts_dict, error = _parse_prompts_json(prompts)

        assert is_valid is False
        assert prompts_dict is None
        assert error == "empty"

    def test_invalid_json(self):
        """Should detect invalid JSON syntax."""
        prompts = '{"0": "missing closing brace"'
        is_valid, prompts_dict, error = _parse_prompts_json(prompts)

        assert is_valid is False
        assert prompts_dict is None
        assert error == "invalid_json"

    def test_malformed_json(self):
        """Should detect malformed JSON."""
        prompts = "not json at all"
        is_valid, prompts_dict, error = _parse_prompts_json(prompts)

        assert is_valid is False
        assert prompts_dict is None
        assert error == "invalid_json"

    def test_single_prompt(self):
        """Should handle single prompt."""
        prompts = '{"0": "single prompt"}'
        is_valid, prompts_dict, error = _parse_prompts_json(prompts)

        assert is_valid is True
        assert len(prompts_dict) == 1
        assert error == ""


class TestHasDefaultPrompts:
    """Test _has_default_prompts() function."""

    def test_prompt_text_placeholder(self):
        """Should detect 'prompt text' placeholder."""
        prompts_dict = {"0": "prompt text here"}
        assert _has_default_prompts(prompts_dict) is True

    def test_beautiful_landscape_placeholder(self):
        """Should detect 'beautiful landscape' placeholder."""
        prompts_dict = {"0": "a beautiful landscape"}
        assert _has_default_prompts(prompts_dict) is True

    def test_load_prompts_placeholder(self):
        """Should detect 'load prompts' placeholder."""
        prompts_dict = {"0": "load prompts first"}
        assert _has_default_prompts(prompts_dict) is True

    def test_case_insensitive(self):
        """Should be case insensitive."""
        prompts_dict = {"0": "PROMPT TEXT"}
        assert _has_default_prompts(prompts_dict) is True

    def test_real_prompt(self):
        """Should return False for real prompts."""
        prompts_dict = {"0": "a futuristic city at night"}
        assert _has_default_prompts(prompts_dict) is False

    def test_empty_dict(self):
        """Should handle empty dict."""
        assert _has_default_prompts({}) is False

    def test_none_value(self):
        """Should handle None value."""
        assert _has_default_prompts(None) is False


class TestBuildValidationMessage:
    """Test _build_validation_message() function."""

    @pytest.fixture
    def emojis(self):
        """Mock emoji dictionary."""
        return {
            'warning': '⚠️',
            'cross': '❌',
            'check': '✅',
            'memo': '📝',
            'movie_camera': '🎥',
            'fire': '🔥',
            'zap': '⚡',
            'wrench': '🔧',
        }

    def test_empty_status(self, emojis):
        """Should build empty prompts message."""
        message = _build_validation_message("empty", None, emojis)

        assert "⚠️" in message
        assert "Prompts Required" in message
        assert "Load from Deforum Prompts" in message

    def test_placeholder_status(self, emojis):
        """Should build placeholder text message."""
        message = _build_validation_message("placeholder", None, emojis)

        assert "⚠️" in message
        assert "Load Real Prompts" in message
        assert "Replace placeholder text" in message

    def test_default_status(self, emojis):
        """Should build default prompts message."""
        message = _build_validation_message("default", None, emojis)

        assert "⚠️" in message
        assert "Default/Placeholder Prompts Detected" in message
        assert "Load your real prompts" in message

    def test_invalid_json_status(self, emojis):
        """Should build invalid JSON message."""
        message = _build_validation_message("invalid_json", None, emojis)

        assert "❌" in message
        assert "Invalid JSON Format" in message
        assert "Fix the format" in message

    def test_ready_status_single_prompt(self, emojis):
        """Should build ready message for single prompt."""
        prompts_dict = {"0": "test prompt"}
        message = _build_validation_message("ready", prompts_dict, emojis)

        assert "✅" in message
        assert "Ready to Generate!" in message
        assert "Found 1 prompt" in message
        assert "prompts" not in message  # Should not pluralize

    def test_ready_status_multiple_prompts(self, emojis):
        """Should build ready message for multiple prompts."""
        prompts_dict = {"0": "prompt one", "60": "prompt two", "120": "prompt three"}
        message = _build_validation_message("ready", prompts_dict, emojis)

        assert "✅" in message
        assert "Ready to Generate!" in message
        assert "Found 3 prompts" in message  # Should pluralize

    def test_unknown_status(self, emojis):
        """Should handle unknown status."""
        message = _build_validation_message("unknown", None, emojis)

        assert "❌" in message
        assert "Unknown validation status" in message


class TestIntegration:
    """Integration tests combining multiple helpers."""

    def test_full_validation_flow_valid(self):
        """Test complete validation flow with valid prompts."""
        prompts = '{"0": "a futuristic city", "60": "neon lights"}'

        # Should pass all checks
        assert not _is_empty_prompts(prompts)
        assert not _has_placeholder_text(prompts)

        is_valid, prompts_dict, error = _parse_prompts_json(prompts)
        assert is_valid
        assert not _has_default_prompts(prompts_dict)

    def test_full_validation_flow_empty(self):
        """Test complete validation flow with empty prompts."""
        prompts = ""

        # Should fail at first check
        assert _is_empty_prompts(prompts)

    def test_full_validation_flow_placeholder(self):
        """Test complete validation flow with placeholder."""
        prompts = "required: load prompts"

        # Should pass empty check but fail placeholder check
        assert not _is_empty_prompts(prompts)
        assert _has_placeholder_text(prompts)

    def test_full_validation_flow_invalid_json(self):
        """Test complete validation flow with invalid JSON."""
        prompts = '{"0": "missing brace"'

        # Should pass empty and placeholder checks but fail JSON parsing
        assert not _is_empty_prompts(prompts)
        assert not _has_placeholder_text(prompts)

        is_valid, prompts_dict, error = _parse_prompts_json(prompts)
        assert not is_valid
        assert error == "invalid_json"

    def test_full_validation_flow_default_prompts(self):
        """Test complete validation flow with default prompts."""
        prompts = '{"0": "a beautiful landscape"}'

        # Should pass all checks except default check
        assert not _is_empty_prompts(prompts)
        assert not _has_placeholder_text(prompts)

        is_valid, prompts_dict, error = _parse_prompts_json(prompts)
        assert is_valid
        assert _has_default_prompts(prompts_dict)
