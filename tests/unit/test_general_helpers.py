"""
Tests for refactored helper functions in deforum/utils/general.py

Tests the small, pure helper functions extracted during complexity refactoring.
These functions were broken out from high-complexity functions to improve
testability and reduce cyclomatic complexity.

NOTE: Imports are done inside test methods to avoid import errors during test
collection, since deforum.utils.general requires modules.shared which is mocked
in conftest.py. This pattern matches test_general.py.
"""

import pytest
from unittest.mock import Mock


class TestIsValidFrameFile:
    """Test _is_valid_frame_file helper (extracted from duplicate_pngs_from_folder)."""

    def test_valid_png_with_batch_id(self):
        """Test valid PNG file matching batch ID."""
        from deforum.utils.general import _is_valid_frame_file
        assert _is_valid_frame_file("batch_001.png", "batch") is True

    def test_valid_jpg_with_batch_id(self):
        """Test valid JPG file matching batch ID."""
        from deforum.utils.general import _is_valid_frame_file
        assert _is_valid_frame_file("batch_001.jpg", "batch") is True

    def test_valid_numeric_frame_no_batch_id(self):
        """Test numeric frame name with no batch ID requirement."""
        from deforum.utils.general import _is_valid_frame_file
        assert _is_valid_frame_file("0001.png", None) is True

    def test_valid_numeric_frame_with_batch_id_none(self):
        """Test numeric frame accepted when batch_id is None."""
        from deforum.utils.general import _is_valid_frame_file
        assert _is_valid_frame_file("12345.png", None) is True

    def test_reject_non_image_file(self):
        """Test rejection of non-image files."""
        from deforum.utils.general import _is_valid_frame_file
        assert _is_valid_frame_file("file.txt", None) is False
        assert _is_valid_frame_file("data.json", None) is False
        assert _is_valid_frame_file("README.md", None) is False

    def test_reject_depth_files(self):
        """Test rejection of depth map files."""
        from deforum.utils.general import _is_valid_frame_file
        assert _is_valid_frame_file("0001_depth_0001.png", None) is False
        assert _is_valid_frame_file("frame_depth_map.jpg", None) is False

    def test_reject_intermediate_files(self):
        """Test rejection of files with hyphens (intermediate files)."""
        from deforum.utils.general import _is_valid_frame_file
        assert _is_valid_frame_file("0001-0002.png", None) is False
        assert _is_valid_frame_file("tween-frame.jpg", None) is False

    def test_reject_empty_filename(self):
        """Test rejection of empty filename."""
        from deforum.utils.general import _is_valid_frame_file
        assert _is_valid_frame_file("", None) is False
        assert _is_valid_frame_file("", "batch") is False

    def test_batch_id_mismatch(self):
        """Test rejection when batch ID doesn't match."""
        from deforum.utils.general import _is_valid_frame_file
        assert _is_valid_frame_file("other_001.png", "batch") is False
        assert _is_valid_frame_file("wrong_prefix.jpg", "correct") is False

    def test_numeric_frame_with_matching_batch_id(self):
        """Test numeric frame when batch ID is specified."""
        from deforum.utils.general import _is_valid_frame_file
        # Numeric frames should still be accepted even with batch_id set
        assert _is_valid_frame_file("0001.png", "batch") is True
        assert _is_valid_frame_file("99999.jpg", "prefix") is True

    def test_both_png_and_jpg_extensions(self):
        """Test that both PNG and JPG extensions are valid."""
        from deforum.utils.general import _is_valid_frame_file
        assert _is_valid_frame_file("frame.png", None) is True
        assert _is_valid_frame_file("frame.jpg", None) is True
        assert _is_valid_frame_file("frame.jpeg", None) is False  # Only png/jpg

    def test_batch_id_prefix_matching(self):
        """Test that batch ID matching uses startswith."""
        from deforum.utils.general import _is_valid_frame_file
        assert _is_valid_frame_file("myBatch_0001.png", "myBatch") is True
        assert _is_valid_frame_file("myBatch_frame.jpg", "myBatch") is True
        assert _is_valid_frame_file("_myBatch_0001.png", "myBatch") is False


class TestFindResumeTimestring:
    """Test _find_resume_timestring helper (extracted from substitute_placeholders)."""

    def test_find_both_attributes(self):
        """Test finding both resume attributes."""
        from deforum.utils.general import _find_resume_timestring
        arg1 = Mock(resume_from_timestring=True, resume_timestring="20231129")
        args = [arg1]

        resume_from, resume_time = _find_resume_timestring(args)

        assert resume_from is True
        assert resume_time == "20231129"

    def test_find_from_multiple_objects(self):
        """Test finding attributes across multiple objects."""
        from deforum.utils.general import _find_resume_timestring
        class MockArg1:
            pass  # No attributes

        class MockArg2:
            resume_from_timestring = True

        class MockArg3:
            resume_timestring = "20231130"

        arg1 = MockArg1()
        arg2 = MockArg2()
        arg3 = MockArg3()
        args = [arg1, arg2, arg3]

        resume_from, resume_time = _find_resume_timestring(args)

        assert resume_from is True
        assert resume_time == "20231130"

    def test_missing_attributes_return_defaults(self):
        from deforum.utils.general import _find_resume_timestring
        """Test defaults when attributes are missing."""
        arg1 = Mock(spec=[])  # No resume attributes
        args = [arg1]

        resume_from, resume_time = _find_resume_timestring(args)

        assert resume_from is False
        assert resume_time is None

    def test_empty_list_returns_defaults(self):
        from deforum.utils.general import _find_resume_timestring
        """Test defaults when arg_list is empty."""
        resume_from, resume_time = _find_resume_timestring([])

        assert resume_from is False
        assert resume_time is None

    def test_resume_from_false(self):
        from deforum.utils.general import _find_resume_timestring
        """Test when resume_from_timestring is False."""
        arg1 = Mock(resume_from_timestring=False, resume_timestring="20231129")
        args = [arg1]

        resume_from, resume_time = _find_resume_timestring(args)

        assert resume_from is False
        assert resume_time == "20231129"

    def test_uses_first_matching_attribute(self):
        from deforum.utils.general import _find_resume_timestring
        """Test that next() returns first matching attribute."""
        arg1 = Mock(resume_from_timestring=True)
        arg2 = Mock(resume_from_timestring=False)  # Should be ignored
        args = [arg1, arg2]

        resume_from, resume_time = _find_resume_timestring(args)

        assert resume_from is True  # From arg1, not arg2


class TestUpdateTimestringsForResume:
    """Test _update_timestrings_for_resume helper (extracted from substitute_placeholders)."""

    def test_update_single_object(self):
        """Test updating timestring on single object."""
        from deforum.utils.general import _update_timestrings_for_resume
        arg1 = Mock(timestring="old")
        args = [arg1]

        _update_timestrings_for_resume(args, "new")

        assert arg1.timestring == "new"

    def test_update_multiple_objects(self):
        """Test updating timestrings on multiple objects."""
        from deforum.utils.general import _update_timestrings_for_resume
        arg1 = Mock(timestring="old1")
        arg2 = Mock(timestring="old2")
        arg3 = Mock(timestring="old3")
        args = [arg1, arg2, arg3]

        _update_timestrings_for_resume(args, "new")

        assert arg1.timestring == "new"
        assert arg2.timestring == "new"
        assert arg3.timestring == "new"

    def test_skip_objects_without_timestring(self):
        """Test skipping objects that don't have timestring attribute."""
        from deforum.utils.general import _update_timestrings_for_resume
        arg1 = Mock()  # No timestring
        arg2 = Mock(timestring="old")
        args = [arg1, arg2]

        # Should not raise AttributeError
        _update_timestrings_for_resume(args, "new")

        assert arg2.timestring == "new"

    def test_empty_list_no_error(self):
        """Test that empty list doesn't cause errors."""
        from deforum.utils.general import _update_timestrings_for_resume
        # Should not raise any exception
        _update_timestrings_for_resume([], "new")

    def test_preserves_other_attributes(self):
        """Test that updating timestring doesn't affect other attributes."""
        from deforum.utils.general import _update_timestrings_for_resume
        arg1 = Mock(timestring="old", other_attr="unchanged")
        args = [arg1]

        _update_timestrings_for_resume(args, "new")

        assert arg1.timestring == "new"
        assert arg1.other_attr == "unchanged"


class TestBuildValuesDict:
    """Test _build_values_dict helper (extracted from substitute_placeholders)."""

    def test_build_from_single_object(self):
        """Test building dict from single object."""
        from deforum.utils.general import _build_values_dict
        class MockArg:
            name = "test"
            value = 42
            enabled = True

        arg1 = MockArg()
        args = [arg1]

        values = _build_values_dict(args)

        assert values["name"] == "test"
        assert values["value"] == 42
        assert values["enabled"] is True

    def test_build_from_multiple_objects(self):
        from deforum.utils.general import _build_values_dict
        """Test building dict from multiple objects."""
        arg1 = Mock(width=512, height=768)
        arg2 = Mock(fps=30, steps=20)
        args = [arg1, arg2]

        values = _build_values_dict(args)

        assert values["width"] == 512
        assert values["height"] == 768
        assert values["fps"] == 30
        assert values["steps"] == 20

    def test_lowercase_keys(self):
        from deforum.utils.general import _build_values_dict
        """Test that keys are lowercased."""
        arg1 = Mock(FPS=30, Width=512)
        args = [arg1]

        values = _build_values_dict(args)

        assert "fps" in values
        assert "width" in values
        assert "FPS" not in values
        assert "Width" not in values

    def test_exclude_callables(self):
        """Test that callable attributes are excluded."""
        from deforum.utils.general import _build_values_dict
        arg1 = Mock(value=42)
        arg1.method = lambda: "test"
        args = [arg1]

        values = _build_values_dict(args)

        assert "value" in values
        assert "method" not in values

    def test_exclude_dunder_attributes(self):
        """Test that __dunder__ attributes are excluded."""
        from deforum.utils.general import _build_values_dict
        arg1 = Mock(value=42)
        args = [arg1]

        values = _build_values_dict(args)

        assert "value" in values
        assert "__class__" not in values
        assert "__dict__" not in values
        assert "__init__" not in values

    def test_later_values_override_earlier(self):
        """Test that values from later objects override earlier ones."""
        from deforum.utils.general import _build_values_dict
        class MockArg1:
            name = "first"

        class MockArg2:
            name = "second"

        arg1 = MockArg1()
        arg2 = MockArg2()
        args = [arg1, arg2]

        values = _build_values_dict(args)

        # Later value should override
        assert values["name"] == "second"

    def test_empty_list_returns_empty_dict(self):
        """Test that empty list returns empty dictionary."""
        from deforum.utils.general import _build_values_dict
        values = _build_values_dict([])

        assert values == {}

    def test_handles_various_types(self):
        """Test that function handles various attribute types."""
        from deforum.utils.general import _build_values_dict
        arg1 = Mock(
            integer=42,
            floating=3.14,
            string="text",
            boolean=True,
            none_value=None,
            list_val=[1, 2, 3],
            dict_val={"key": "value"}
        )
        args = [arg1]

        values = _build_values_dict(args)

        assert values["integer"] == 42
        assert values["floating"] == 3.14
        assert values["string"] == "text"
        assert values["boolean"] is True
        assert values["none_value"] is None
        assert values["list_val"] == [1, 2, 3]
        assert values["dict_val"] == {"key": "value"}


class TestSubstituteAndCleanTemplate:
    """Test _substitute_and_clean_template helper (extracted from substitute_placeholders)."""

    def test_simple_substitution(self):
        """Test simple placeholder substitution."""
        from deforum.utils.general import _substitute_and_clean_template
        template = "{width}x{height}"
        values = {"width": "512", "height": "768"}

        result = _substitute_and_clean_template(template, values)

        assert result == "512x768"

    def test_remove_invalid_placeholders(self):
        """Test removal of invalid (missing) placeholders."""
        from deforum.utils.general import _substitute_and_clean_template
        template = "{width}x{invalid}"
        values = {"width": "512"}

        result = _substitute_and_clean_template(template, values)

        # Invalid placeholder and its braces should be removed
        assert "{" not in result
        assert "}" not in result
        assert "512x" in result

    def test_clean_invalid_filename_characters(self):
        """Test cleaning of invalid filename characters."""
        from deforum.utils.general import _substitute_and_clean_template
        template = "output/file:name*here"
        values = {}

        result = _substitute_and_clean_template(template, values)

        # Invalid chars should be replaced with underscores
        assert ":" not in result
        assert "*" not in result
        assert "/" not in result
        assert result == "output_file_name_here"

    def test_clean_all_invalid_chars(self):
        """Test cleaning all types of invalid filename characters."""
        from deforum.utils.general import _substitute_and_clean_template
        template = '<>:"/\\|?*test'
        values = {}

        result = _substitute_and_clean_template(template, values)

        # All invalid chars should be gone
        for char in '<>:"/\\|?*':
            assert char not in result

    def test_clean_trailing_underscores(self):
        """Test removal of trailing underscores."""
        from deforum.utils.general import _substitute_and_clean_template
        template = "filename   "  # Spaces become underscores, then get trimmed
        values = {}

        result = _substitute_and_clean_template(template, values)

        assert not result.endswith("_")

    def test_multiple_placeholders(self):
        """Test multiple placeholder substitutions."""
        from deforum.utils.general import _substitute_and_clean_template
        template = "{batch}_{timestring}_{seed}"
        values = {"batch": "test", "timestring": "20231129", "seed": "12345"}

        result = _substitute_and_clean_template(template, values)

        assert result == "test_20231129_12345"

    def test_preserve_underscores_in_valid_content(self):
        """Test that valid underscores are preserved."""
        from deforum.utils.general import _substitute_and_clean_template
        template = "{batch}_output_final"
        values = {"batch": "test"}

        result = _substitute_and_clean_template(template, values)

        assert result == "test_output_final"

    def test_remove_extra_braces(self):
        """Test removal of extra/stray braces."""
        from deforum.utils.general import _substitute_and_clean_template
        template = "test{{double}}braces"
        values = {}

        result = _substitute_and_clean_template(template, values)

        assert "{" not in result
        assert "}" not in result

    def test_empty_template(self):
        """Test empty template string."""
        from deforum.utils.general import _substitute_and_clean_template
        result = _substitute_and_clean_template("", {})

        assert result == ""

    def test_no_placeholders(self):
        """Test template without any placeholders."""
        from deforum.utils.general import _substitute_and_clean_template
        template = "simple_filename"
        values = {"unused": "value"}

        result = _substitute_and_clean_template(template, values)

        assert result == "simple_filename"

    def test_spaces_become_underscores(self):
        """Test that spaces are replaced with underscores."""
        from deforum.utils.general import _substitute_and_clean_template
        template = "file name with spaces"
        values = {}

        result = _substitute_and_clean_template(template, values)

        assert " " not in result
        assert "file_name_with_spaces" == result

    def test_commas_become_underscores(self):
        """Test that commas are replaced with underscores."""
        from deforum.utils.general import _substitute_and_clean_template
        template = "file,with,commas"
        values = {}

        result = _substitute_and_clean_template(template, values)

        assert "," not in result
        assert "file_with_commas" == result

    def test_mixed_valid_and_invalid_placeholders(self):
        """Test combination of valid and invalid placeholders."""
        from deforum.utils.general import _substitute_and_clean_template
        template = "{valid1}_{invalid}_{valid2}"
        values = {"valid1": "a", "valid2": "b"}

        result = _substitute_and_clean_template(template, values)

        assert "a_" in result
        assert "_b" in result
        assert "{" not in result
        assert "}" not in result


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
