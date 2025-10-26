"""Unit tests for deforum.utils.parsing.prompts module.

Tests pure functions for prompt parsing and transformation.
All functions here are side-effect free and easily testable.
"""

import json
from deforum.utils.parsing.prompts import (
    remove_negative_prompt,
    convert_deforum_to_wan_prompts,
    format_prompts_as_multiline,
    format_prompts_as_json,
    create_error_prompt,
    parse_prompts_json,
    validate_prompts_not_empty,
    create_fallback_prompts,
)


# ============================================================================
# Negative Prompt Removal
# ============================================================================


class TestRemoveNegativePrompt:
    """Test removal of negative prompts."""

    def test_prompt_with_negative(self):
        """Test removing negative section."""
        prompt = "beautiful landscape --neg ugly, distorted"
        result = remove_negative_prompt(prompt)

        assert result == "beautiful landscape"

    def test_prompt_without_negative(self):
        """Test prompt without negative section unchanged."""
        prompt = "beautiful landscape"
        result = remove_negative_prompt(prompt)

        assert result == "beautiful landscape"

    def test_prompt_with_trailing_spaces(self):
        """Test that trailing spaces are stripped."""
        prompt = "beautiful landscape  --neg ugly"
        result = remove_negative_prompt(prompt)

        assert result == "beautiful landscape"

    def test_empty_prompt(self):
        """Test empty prompt returns empty string."""
        result = remove_negative_prompt("")

        assert result == ""

    def test_only_negative(self):
        """Test prompt that's only negative section."""
        prompt = "--neg ugly, distorted"
        result = remove_negative_prompt(prompt)

        assert result == ""


# ============================================================================
# Deforum to Wan Conversion
# ============================================================================


class TestConvertDeforumToWanPrompts:
    """Test conversion from Deforum to Wan format."""

    def test_simple_conversion(self):
        """Test basic conversion."""
        deforum = {"0": "scene one", "60": "scene two"}

        wan = convert_deforum_to_wan_prompts(deforum)

        assert wan == {"0": "scene one", "60": "scene two"}

    def test_removes_negatives(self):
        """Test that negative prompts are removed."""
        deforum = {"0": "beautiful landscape --neg ugly", "60": "sunset scene --neg dark, gloomy"}

        wan = convert_deforum_to_wan_prompts(deforum)

        assert wan == {"0": "beautiful landscape", "60": "sunset scene"}

    def test_empty_prompts(self):
        """Test empty prompts dict."""
        wan = convert_deforum_to_wan_prompts({})

        assert wan == {}

    def test_preserves_frame_keys(self):
        """Test that frame keys are preserved."""
        deforum = {"0": "start", "100": "middle", "200": "end"}

        wan = convert_deforum_to_wan_prompts(deforum)

        assert set(wan.keys()) == {"0", "100", "200"}


# ============================================================================
# Multiline Formatting
# ============================================================================


class TestFormatPromptsAsMultiline:
    """Test formatting prompts as multiline string."""

    def test_simple_prompts(self):
        """Test basic multiline formatting."""
        prompts = {"0": "start", "60": "end"}

        result = format_prompts_as_multiline(prompts)

        assert result == "0: start\n60: end"

    def test_sorts_by_frame(self):
        """Test that frames are sorted."""
        prompts = {"120": "end", "0": "start", "60": "middle"}

        result = format_prompts_as_multiline(prompts)

        assert result == "0: start\n60: middle\n120: end"

    def test_empty_prompts(self):
        """Test empty prompts dict."""
        result = format_prompts_as_multiline({})

        assert result == "0: "

    def test_single_prompt(self):
        """Test single prompt."""
        prompts = {"0": "only one"}

        result = format_prompts_as_multiline(prompts)

        assert result == "0: only one"


# ============================================================================
# JSON Formatting
# ============================================================================


class TestFormatPromptsAsJson:
    """Test formatting prompts as JSON."""

    def test_simple_json(self):
        """Test basic JSON formatting."""
        prompts = {"0": "start", "60": "end"}

        result = format_prompts_as_json(prompts)

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed == prompts

    def test_preserves_unicode(self):
        """Test that unicode is preserved (ensure_ascii=False)."""
        prompts = {"0": "café ☕", "60": "日本語"}

        result = format_prompts_as_json(prompts)

        assert "café" in result
        assert "☕" in result
        assert "日本語" in result

    def test_custom_indent(self):
        """Test custom indentation."""
        prompts = {"0": "test"}

        result = format_prompts_as_json(prompts, indent=4)

        # Check that indentation is applied
        assert "    " in result  # 4 spaces

    def test_empty_prompts(self):
        """Test empty prompts dict."""
        result = format_prompts_as_json({})

        parsed = json.loads(result)
        assert parsed == {}


# ============================================================================
# Error Prompt Creation
# ============================================================================


class TestCreateErrorPrompt:
    """Test creation of error prompts."""

    def test_creates_valid_json(self):
        """Test that error prompt is valid JSON."""
        result = create_error_prompt("Something went wrong")

        parsed = json.loads(result)
        assert "0" in parsed

    def test_includes_error_message(self):
        """Test that error message is included."""
        error_msg = "Test error message"
        result = create_error_prompt(error_msg)

        parsed = json.loads(result)
        assert parsed["0"] == error_msg

    def test_special_characters(self):
        """Test error messages with special characters."""
        error_msg = "Error: \"quotes\" and 'apostrophes'"
        result = create_error_prompt(error_msg)

        # Should be valid JSON
        parsed = json.loads(result)
        assert parsed["0"] == error_msg


# ============================================================================
# JSON Parsing
# ============================================================================


class TestParsePromptsJson:
    """Test JSON parsing with error handling."""

    def test_valid_json(self):
        """Test parsing valid JSON."""
        json_str = '{"0": "test", "60": "test2"}'

        prompts, error = parse_prompts_json(json_str)

        assert error is None
        assert prompts == {"0": "test", "60": "test2"}

    def test_invalid_json(self):
        """Test parsing invalid JSON."""
        json_str = '{"0": "test"'  # Missing closing brace

        prompts, error = parse_prompts_json(json_str)

        assert error is not None
        assert "Invalid JSON" in error
        assert prompts == {}

    def test_non_dict_json(self):
        """Test JSON that's not a dictionary."""
        json_str = '["array", "not", "dict"]'

        prompts, error = parse_prompts_json(json_str)

        assert error is not None
        assert "dictionary" in error.lower()
        assert prompts == {}

    def test_custom_default_on_error(self):
        """Test custom default returned on error."""
        json_str = "invalid"
        default = {"0": "fallback"}

        prompts, error = parse_prompts_json(json_str, default_on_error=default)

        assert error is not None
        assert prompts == default

    def test_empty_string(self):
        """Test empty string."""
        prompts, error = parse_prompts_json("")

        assert error is not None
        assert prompts == {}


# ============================================================================
# Validation
# ============================================================================


class TestValidatePromptsNotEmpty:
    """Test validation of non-empty prompts."""

    def test_valid_non_empty(self):
        """Test valid non-empty prompts."""
        is_valid, error = validate_prompts_not_empty('{"0": "test"}')

        assert is_valid is True
        assert error == ""

    def test_empty_string(self):
        """Test empty string is invalid."""
        is_valid, error = validate_prompts_not_empty("")

        assert is_valid is False
        assert "No prompts" in error

    def test_whitespace_only(self):
        """Test whitespace-only string is invalid."""
        is_valid, error = validate_prompts_not_empty("   \n  ")

        assert is_valid is False
        assert "No prompts" in error

    def test_valid_whitespace_padded(self):
        """Test valid JSON with padding is valid."""
        is_valid, error = validate_prompts_not_empty('  {"0": "test"}  ')

        assert is_valid is True


# ============================================================================
# Fallback Creation
# ============================================================================


class TestCreateFallbackPrompts:
    """Test creation of fallback prompts."""

    def test_returns_dict(self):
        """Test that fallback returns a dict."""
        result = create_fallback_prompts()

        assert isinstance(result, dict)

    def test_has_frame_0(self):
        """Test that fallback includes frame 0."""
        result = create_fallback_prompts()

        assert "0" in result

    def test_has_multiple_frames(self):
        """Test that fallback has multiple frames."""
        result = create_fallback_prompts()

        assert len(result) >= 2

    def test_prompts_not_empty(self):
        """Test that fallback prompts are not empty strings."""
        result = create_fallback_prompts()

        for prompt in result.values():
            assert prompt != ""
            assert len(prompt) > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestPromptUtilsIntegration:
    """Integration tests for complete workflows."""

    def test_deforum_to_wan_workflow(self):
        """Test complete Deforum to Wan conversion workflow."""
        # Start with Deforum prompts
        deforum = {
            "0": "beautiful landscape --neg ugly, dark",
            "60": "sunset scene --neg gloomy",
            "120": "night sky",
        }

        # Convert to Wan
        wan = convert_deforum_to_wan_prompts(deforum)

        # Format as JSON
        json_str = format_prompts_as_json(wan)

        # Parse back
        parsed, error = parse_prompts_json(json_str)

        assert error is None
        assert parsed == {"0": "beautiful landscape", "60": "sunset scene", "120": "night sky"}

    def test_error_handling_workflow(self):
        """Test error handling workflow."""
        # Invalid JSON
        invalid_json = "not valid json"

        # Parse with fallback
        prompts, error = parse_prompts_json(
            invalid_json, default_on_error=create_fallback_prompts()
        )

        # Should have error but valid fallback
        assert error is not None
        assert prompts is not None
        assert "0" in prompts

    def test_multiline_format_workflow(self):
        """Test multiline formatting workflow."""
        # Parse JSON
        json_str = '{"0": "start", "120": "end", "60": "middle"}'
        prompts, _ = parse_prompts_json(json_str)

        # Format as multiline
        multiline = format_prompts_as_multiline(prompts)

        # Should be sorted
        lines = multiline.split("\n")
        assert lines[0].startswith("0:")
        assert lines[1].startswith("60:")
        assert lines[2].startswith("120:")
