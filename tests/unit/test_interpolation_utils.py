"""Unit tests for frame interpolation calculation utilities."""

import pytest

from deforum.utils.math.interpolation import (
    extract_rife_name,
    clean_folder_name,
    set_interp_out_fps,
    calculate_frames_to_add,
)


class TestExtractRifeName:
    """Test extract_rife_name function."""

    def test_standard_version(self):
        result = extract_rife_name("RIFE v4.26")
        assert result == "RIFE426"

    def test_single_digit_version(self):
        result = extract_rife_name("RIFE v2.0")
        assert result == "RIFE20"

    def test_multi_digit_version(self):
        result = extract_rife_name("RIFE v10.15")
        assert result == "RIFE1015"

    def test_three_part_version(self):
        result = extract_rife_name("RIFE v4.26.1")
        assert result == "RIFE4261"

    def test_single_number_version(self):
        result = extract_rife_name("RIFE v5")
        assert result == "RIFE5"

    def test_invalid_missing_v_prefix(self):
        with pytest.raises(ValueError, match="Version should start with 'v'"):
            extract_rife_name("RIFE 4.26")

    def test_invalid_wrong_first_word(self):
        with pytest.raises(ValueError, match="First word should be 'RIFE'"):
            extract_rife_name("FILM v4.26")

    def test_invalid_too_many_words(self):
        with pytest.raises(ValueError, match="exactly 2 words"):
            extract_rife_name("RIFE v4.26 beta")

    def test_invalid_too_few_words(self):
        with pytest.raises(ValueError, match="exactly 2 words"):
            extract_rife_name("RIFE")

    def test_invalid_non_numeric_version(self):
        with pytest.raises(ValueError, match="only digits and dots"):
            extract_rife_name("RIFE v4.26a")

    def test_invalid_letters_in_version(self):
        with pytest.raises(ValueError, match="only digits and dots"):
            extract_rife_name("RIFE vABC")


class TestCleanFolderName:
    """Test clean_folder_name function."""

    def test_already_clean(self):
        result = clean_folder_name("my_video")
        assert result == "my_video"

    def test_extension(self):
        result = clean_folder_name("my_video.mp4")
        assert result == "my_video_mp4"

    def test_forward_slash(self):
        result = clean_folder_name("path/to/file")
        assert result == "path_to_file"

    def test_backslash(self):
        result = clean_folder_name("path\\to\\file")
        assert result == "path_to_file"

    def test_colon(self):
        result = clean_folder_name("time:stamp")
        assert result == "time_stamp"

    def test_angle_brackets(self):
        result = clean_folder_name("anime<girl>")
        assert result == "anime_girl_"

    def test_quotes(self):
        result = clean_folder_name('video"name"')
        assert result == "video_name_"

    def test_asterisk_and_question(self):
        result = clean_folder_name("file*name?txt")
        assert result == "file_name_txt"

    def test_pipe(self):
        result = clean_folder_name("name|version")
        assert result == "name_version"

    def test_spaces(self):
        result = clean_folder_name("my video file")
        assert result == "my_video_file"

    def test_comma(self):
        result = clean_folder_name("item1,item2")
        assert result == "item1_item2"

    def test_period(self):
        result = clean_folder_name("file.name.txt")
        assert result == "file_name_txt"

    def test_all_illegal_chars(self):
        # Input has 15 illegal chars: /\\<>:"|?*.,\ (space)(apostrophe)(space)
        # Note: String '/\\<>:"|?*.,\' ' has some escaped chars
        result = clean_folder_name('/\\<>:"|?*., ')
        # Each illegal char → underscore
        assert result == "____________"  # 12 underscores

    def test_empty_string(self):
        result = clean_folder_name("")
        assert result == ""

    def test_complex_filename(self):
        result = clean_folder_name("my/video<2023>: final|version.mp4")
        assert result == "my_video_2023___final_version_mp4"


class TestSetInterpOutFps:
    """Test set_interp_out_fps function."""

    def test_disabled_interpolation(self):
        result = set_interp_out_fps("Disabled", False, "1", 30.0)
        assert result == "---"

    def test_none_fps(self):
        result = set_interp_out_fps("2", False, "1", None)
        assert result == "---"

    def test_empty_string_fps(self):
        result = set_interp_out_fps("2", False, "1", "")
        assert result == "---"

    def test_none_string_fps(self):
        result = set_interp_out_fps("2", False, "1", "None")
        assert result == "---"

    def test_dashes_fps(self):
        result = set_interp_out_fps("2", False, "1", "---")
        assert result == "---"

    def test_2x_interpolation_no_slowmo(self):
        result = set_interp_out_fps("2", False, "1", 30.0)
        assert result == 60.0

    def test_3x_interpolation_no_slowmo(self):
        result = set_interp_out_fps("3", False, "1", 24.0)
        assert result == 72.0

    def test_4x_interpolation_no_slowmo(self):
        result = set_interp_out_fps("4", False, "1", 30.0)
        assert result == 120.0

    def test_2x_interpolation_2x_slowmo(self):
        result = set_interp_out_fps("2", True, "2", 30.0)
        assert result == 30.0

    def test_4x_interpolation_2x_slowmo(self):
        result = set_interp_out_fps("4", True, "2", 30.0)
        assert result == 60.0

    def test_3x_interpolation_3x_slowmo(self):
        result = set_interp_out_fps("3", True, "3", 24.0)
        assert result == 24.0

    def test_returns_int_for_whole_number(self):
        result = set_interp_out_fps("2", False, "1", 30.0)
        assert isinstance(result, int)
        assert result == 60

    def test_returns_float_for_fractional(self):
        # 30 * 3 / 2 = 45.0 which is a whole number, so returns int
        result = set_interp_out_fps("3", True, "2", 30.0)
        assert result == 45
        # Test with truly fractional result: 25 * 3 / 2 = 37.5
        result2 = set_interp_out_fps("3", True, "2", 25.0)
        assert isinstance(result2, float)
        assert result2 == 37.5

    def test_high_framerate(self):
        result = set_interp_out_fps("10", False, "1", 60.0)
        assert result == 600

    def test_slowmo_disabled_but_value_set(self):
        # Slow-mo factor should be ignored if slow_x_enabled is False
        result = set_interp_out_fps("2", False, "5", 30.0)
        assert result == 60.0

    def test_string_fps_input(self):
        result = set_interp_out_fps("2", False, "1", "30.0")
        assert result == 60.0

    def test_integer_interp_x(self):
        result = set_interp_out_fps(2, False, 1, 30.0)
        assert result == 60.0

    def test_integer_slow_x(self):
        result = set_interp_out_fps("2", True, 2, 30.0)
        assert result == 30.0


class TestCalculateFramesToAdd:
    """Test calculate_frames_to_add function."""

    def test_10_frames_2x(self):
        result = calculate_frames_to_add(10, 2)
        assert result == 1

    def test_10_frames_3x(self):
        result = calculate_frames_to_add(10, 3)
        assert result == 2

    def test_5_frames_4x(self):
        result = calculate_frames_to_add(5, 4)
        assert result == 4

    def test_2_frames_2x(self):
        # Edge case: only 2 frames
        result = calculate_frames_to_add(2, 2)
        assert result == 2

    def test_100_frames_2x(self):
        result = calculate_frames_to_add(100, 2)
        assert result == 1

    def test_100_frames_5x(self):
        result = calculate_frames_to_add(100, 5)
        assert result == 4

    def test_50_frames_3x(self):
        result = calculate_frames_to_add(50, 3)
        assert result == 2

    def test_rounding_up(self):
        # Test that rounding works correctly
        # 20 frames, 3x: (20*3 - 20) / (20-1) = 40/19 ≈ 2.105 → rounds to 2
        result = calculate_frames_to_add(20, 3)
        assert result == 2

    def test_rounding_down(self):
        # Test that rounding works correctly
        # 15 frames, 4x: (15*4 - 15) / (15-1) = 45/14 ≈ 3.214 → rounds to 3
        result = calculate_frames_to_add(15, 4)
        assert result == 3

    def test_large_frame_count(self):
        result = calculate_frames_to_add(1000, 2)
        assert result == 1


class TestIntegration:
    """Integration tests combining interpolation utilities."""

    def test_rife_workflow(self):
        """Test complete RIFE interpolation setup."""
        model_name = "RIFE v4.26"
        total_frames = 100
        interp_x = 2
        in_fps = 30.0

        # Extract model folder name
        model_folder = extract_rife_name(model_name)
        assert model_folder == "RIFE426"

        # Calculate output FPS
        out_fps = set_interp_out_fps(str(interp_x), False, "1", in_fps)
        assert out_fps == 60.0

        # Calculate frames to add (not used by RIFE but for comparison)
        frames_to_add = calculate_frames_to_add(total_frames, interp_x)
        assert frames_to_add == 1

    def test_film_workflow(self):
        """Test complete FILM interpolation setup."""
        filename = "my video (2023): final cut.mp4"
        total_frames = 50
        interp_x = 3
        in_fps = 24.0

        # Clean filename for folder
        folder_name = clean_folder_name(filename)
        assert "/" not in folder_name
        assert ":" not in folder_name

        # Calculate output FPS
        out_fps = set_interp_out_fps(str(interp_x), False, "1", in_fps)
        assert out_fps == 72.0

        # Calculate frames FILM needs to add
        frames_to_add = calculate_frames_to_add(total_frames, interp_x)
        assert frames_to_add == 2

    def test_slowmo_workflow(self):
        """Test interpolation with slow-motion."""
        total_frames = 200
        interp_x = 4
        slow_x = 2
        in_fps = 60.0

        # Calculate FPS with slow-mo: 60 * 4 / 2 = 120
        out_fps = set_interp_out_fps(str(interp_x), True, str(slow_x), in_fps)
        assert out_fps == 120.0

        # Frames to add per pair
        frames_to_add = calculate_frames_to_add(total_frames, interp_x)
        assert frames_to_add == 3

    def test_complex_filename_cleaning(self):
        """Test cleaning various problematic filenames."""
        filenames = [
            "test/path\\file.mp4",
            "video: part<2>.avi",
            "file|version*3?.mov",
            "my, video. name .webm",
        ]

        for filename in filenames:
            cleaned = clean_folder_name(filename)
            # Verify no illegal chars remain
            illegal_chars = '/\\<>:"|?*.,\' '
            for char in illegal_chars:
                assert char not in cleaned


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
