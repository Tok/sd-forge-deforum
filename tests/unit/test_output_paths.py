"""
Tests for deforum/utils/output_paths.py
Centralized output path management utilities.
"""

import pytest
from pathlib import Path
from deforum.utils.output_paths import OutputPaths


class TestOutputPaths:
    """Test OutputPaths class constants and utilities."""

    def test_base_constants(self):
        """Test base directory constants."""
        assert OutputPaths.BASE == "output"
        assert OutputPaths.DEFORUM == "output/deforum"
        assert OutputPaths.DEFORUM_TUNING == "output/deforum-tuning"
        assert OutputPaths.DEFORUM_TESTS == "output/deforum-tests"

    def test_get_deforum_output_no_batch(self):
        """Test get_deforum_output without batch name."""
        result = OutputPaths.get_deforum_output()
        assert result == Path("output/deforum")

    def test_get_deforum_output_with_batch(self):
        """Test get_deforum_output with batch name."""
        result = OutputPaths.get_deforum_output("test_batch")
        assert result == Path("output/deforum/test_batch")

    def test_get_deforum_output_with_timestring_batch(self):
        """Test get_deforum_output with timestring-based batch."""
        result = OutputPaths.get_deforum_output("Deforum_20231129")
        assert result == Path("output/deforum/Deforum_20231129")

    def test_find_settings_file_not_found(self):
        """Test find_settings_file returns None when not found."""
        result = OutputPaths.find_settings_file("nonexistent_20231129")
        assert result is None

    def test_find_settings_file_searches_multiple_paths(self, tmp_path):
        """Test find_settings_file searches all possible paths."""
        # This test documents the search behavior without requiring actual files
        timestring = "test_20231129"
        result = OutputPaths.find_settings_file(timestring)

        # Should return None when file doesn't exist
        assert result is None

    def test_find_settings_file_with_custom_outdir(self):
        """Test find_settings_file with custom output directory."""
        timestring = "test_20231129"
        custom_dir = "/custom/output"

        result = OutputPaths.find_settings_file(timestring, custom_dir)

        # Should return None when file doesn't exist
        assert result is None

    def test_find_settings_file_in_deforum_subdir(self, tmp_path):
        """Test finding settings file in Deforum/Deforum_timestring/ structure."""
        # Create test structure
        timestring = "20231129_120000"
        settings_dir = tmp_path / "output" / "deforum" / f"Deforum_{timestring}"
        settings_dir.mkdir(parents=True)
        settings_file = settings_dir / f"{timestring}_settings.txt"
        settings_file.write_text("test settings")

        # Temporarily change to tmp_path for path resolution
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = OutputPaths.find_settings_file(timestring)
            # Should find the file
            assert result is not None
            assert result.exists()
            assert result.name == f"{timestring}_settings.txt"
        finally:
            os.chdir(original_cwd)

    def test_find_settings_file_in_timestring_subdir(self, tmp_path):
        """Test finding settings file in Deforum/timestring/ structure."""
        # Create test structure
        timestring = "20231129_120000"
        settings_dir = tmp_path / "output" / "deforum" / timestring
        settings_dir.mkdir(parents=True)
        settings_file = settings_dir / f"{timestring}_settings.txt"
        settings_file.write_text("test settings")

        # Temporarily change to tmp_path
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = OutputPaths.find_settings_file(timestring)
            assert result is not None
            assert result.exists()
            assert result.name == f"{timestring}_settings.txt"
        finally:
            os.chdir(original_cwd)

    def test_find_settings_file_in_custom_outdir(self, tmp_path):
        """Test finding settings file in custom output directory."""
        timestring = "20231129_120000"
        custom_dir = tmp_path / "custom_output"
        settings_file = custom_dir / f"{timestring}_settings.txt"
        settings_file.parent.mkdir(parents=True)
        settings_file.write_text("test settings")

        result = OutputPaths.find_settings_file(timestring, str(custom_dir))
        assert result is not None
        assert result.exists()
        assert result.name == f"{timestring}_settings.txt"

    def test_find_settings_file_prioritizes_first_match(self, tmp_path):
        """Test that find_settings_file returns first matching path."""
        timestring = "20231129_120000"

        # Create multiple possible locations
        dir1 = tmp_path / "output" / "deforum" / f"Deforum_{timestring}"
        dir2 = tmp_path / "output" / "deforum" / timestring

        dir1.mkdir(parents=True)
        dir2.mkdir(parents=True)

        file1 = dir1 / f"{timestring}_settings.txt"
        file2 = dir2 / f"{timestring}_settings.txt"

        file1.write_text("first file")
        file2.write_text("second file")

        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = OutputPaths.find_settings_file(timestring)
            assert result is not None
            # Should find first one (Deforum_timestring structure)
            assert "Deforum_" in str(result)
        finally:
            os.chdir(original_cwd)

    def test_path_returns_pathlib_path(self):
        """Test that get_deforum_output returns Path objects."""
        result = OutputPaths.get_deforum_output()
        assert isinstance(result, Path)

        result_with_batch = OutputPaths.get_deforum_output("batch")
        assert isinstance(result_with_batch, Path)

    def test_find_settings_file_returns_path_or_none(self):
        """Test that find_settings_file returns Path or None."""
        result = OutputPaths.find_settings_file("nonexistent")
        assert result is None or isinstance(result, Path)
