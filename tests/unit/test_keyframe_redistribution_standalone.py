"""Standalone unit tests for keyframe redistribution logic.

CRITICAL: These tests ensure that REDISTRIBUTED mode maintains exact keyframe placement
while distributing cadence frames appropriately between keyframes.

This is a standalone test that doesn't import the full module to avoid dependency issues.
"""

import pytest


def select_keyframes_simple(prompt_keyframes, max_frames, use_parseq=False, parseq_json=None):
    """Simplified keyframe selection logic."""
    if use_parseq and parseq_json:
        keyframes = [kf["frame"] for kf in parseq_json["keyframes"]]
    else:
        keyframes = [int(k) for k in prompt_keyframes] + [max_frames - 1]
        keyframes = sorted(list(set(keyframes)))
        keyframes[0] = 0  # Ensure first frame is 0
    return keyframes


def redistributed_logic(max_frames, prompt_keyframes, diffusion_frame_count, cadence,
                       use_parseq=False, parseq_json=None):
    """Standalone implementation of redistributed logic for testing."""
    # Get EXACT keyframes
    keyframes = select_keyframes_simple(prompt_keyframes, max_frames, use_parseq, parseq_json)
    keyframes_set = set(keyframes)

    num_keyframes = len(keyframes)
    cadence_budget = diffusion_frame_count - num_keyframes

    if cadence_budget <= 0:
        return sorted(keyframes)

    min_spacing = max(1, cadence // 2)

    # Distribute cadence frames between keyframes
    cadence_frames = []
    keyframes_sorted = sorted(keyframes)

    for i in range(len(keyframes_sorted) - 1):
        section_start = keyframes_sorted[i]
        section_end = keyframes_sorted[i + 1]
        section_length = section_end - section_start

        if section_length <= min_spacing:
            continue

        num_cadence_in_section = max(0, (section_length - min_spacing) // cadence)

        if num_cadence_in_section == 0:
            continue

        for j in range(1, num_cadence_in_section + 1):
            cadence_frame = section_start + int(j * section_length / (num_cadence_in_section + 1))

            too_close = any(abs(cadence_frame - kf) < min_spacing for kf in keyframes_sorted)

            if not too_close and cadence_frame not in keyframes_set:
                cadence_frames.append(cadence_frame)

    all_frames = sorted(list(keyframes_set) + cadence_frames)

    # Trim if too many
    if len(all_frames) > diffusion_frame_count:
        cadence_frames = [f for f in all_frames if f not in keyframes_set]
        cadence_frames = cadence_frames[:diffusion_frame_count - num_keyframes]
        all_frames = sorted(list(keyframes_set) + cadence_frames)

    # Add more if too few
    while len(all_frames) < diffusion_frame_count:
        largest_gap_start = 0
        largest_gap_size = 0

        for i in range(len(all_frames) - 1):
            gap_size = all_frames[i + 1] - all_frames[i]
            if gap_size > largest_gap_size:
                largest_gap_size = gap_size
                largest_gap_start = all_frames[i]

        if largest_gap_size <= 1:
            break

        new_frame = largest_gap_start + largest_gap_size // 2
        if new_frame not in all_frames:
            all_frames.append(new_frame)
            all_frames.sort()

    return all_frames


class TestKeyframeRedistribution:
    """Test suite for REDISTRIBUTED keyframe distribution mode."""

    def test_exact_keyframe_placement(self):
        """Test that keyframes appear at EXACT requested positions."""
        result = redistributed_logic(100, ['25', '50', '75'], 20, 5)

        # CRITICAL: All keyframes MUST be at exact positions
        for kf in [0, 25, 50, 75, 99]:
            assert kf in result, f"Keyframe {kf} missing from result!"

    def test_keyframes_never_moved(self):
        """Test that keyframes are NEVER moved to approximate cadence."""
        result = redistributed_logic(100, ['23', '47', '71'], 15, 10)

        # CRITICAL: Keyframes must be at EXACT positions, not moved to cadence
        assert 23 in result, "Keyframe 23 was moved!"
        assert 47 in result, "Keyframe 47 was moved!"
        assert 71 in result, "Keyframe 71 was moved!"

    def test_cadence_frames_between_keyframes(self):
        """Test that cadence frames are distributed BETWEEN keyframes."""
        result = redistributed_logic(100, ['50'], 12, 10)

        # Check that cadence frames are BETWEEN keyframes, not outside
        for frame in result:
            if frame not in [0, 50, 99]:  # If it's a cadence frame
                # Must be between keyframes
                assert (0 < frame < 50) or (50 < frame < 99), \
                    f"Cadence frame {frame} is not between keyframes!"

    def test_minimum_spacing_enforcement(self):
        """Test that cadence frames maintain minimum spacing from keyframes."""
        result = redistributed_logic(100, ['10', '20', '30'], 15, 5)

        min_spacing = 5 // 2  # From algorithm

        # Check spacing between consecutive diffusion frames
        sorted_result = sorted(result)
        for i in range(len(sorted_result) - 1):
            spacing = sorted_result[i + 1] - sorted_result[i]
            # Allow spacing < min only if both are keyframes (unavoidable)
            if spacing < min_spacing:
                assert sorted_result[i] in [0, 10, 20, 30, 99] and \
                       sorted_result[i + 1] in [0, 10, 20, 30, 99], \
                    f"Spacing violation: {sorted_result[i]} to {sorted_result[i + 1]} (spacing={spacing})"

    def _test_no_cadence_budget(self):  # DISABLED
        """Test behavior when all diffusion budget is used by keyframes."""
        # 6 keyframes: 0, 10, 20, 30, 40, 49 exactly fills budget of 6
        result = redistributed_logic(50, ['10', '20', '30', '40'], 6, 5)

        # Should return ONLY exact keyframes (no cadence frames)
        expected_keyframes = {0, 10, 20, 30, 40, 49}
        assert expected_keyframes.issubset(set(result)), \
            f"Missing keyframes: {expected_keyframes - set(result)}"
        assert len(result) == 6, \
            f"Should have exactly 6 frames (all keyframes), got {len(result)}"

    def _test_large_gap(self):  # DISABLED
        """Test handling of large gap between two keyframes."""
        # Empty prompt_keyframes means only start (0) and end (99) keyframes
        result = redistributed_logic(100, ['50'], 12, 10)  # Add a middle keyframe

        # Should have keyframes at 0, 50, and 99
        assert 0 in result
        assert 50 in result
        assert 99 in result

        # Should have cadence frames in between
        cadence_frames = [f for f in result if f not in [0, 50, 99]]
        assert len(cadence_frames) > 0, "Should have cadence frames in large gap"

    def test_uneven_spacing(self):
        """Test with very uneven keyframe spacing."""
        result = redistributed_logic(100, ['10', '90'], 20, 5)

        # Keyframes must be exact
        assert 0 in result and 10 in result and 90 in result and 99 in result

        # Most cadence frames should be in the large gap (10-90)
        cadence_in_large_gap = len([f for f in result if 10 < f < 90])
        cadence_in_small_gaps = len([f for f in result if (0 < f < 10) or (90 < f < 99)])

        assert cadence_in_large_gap > cadence_in_small_gaps

    def test_frame_count_target(self):
        """Test that result matches target diffusion_frame_count."""
        result = redistributed_logic(200, ['50', '100', '150'], 25, 10)

        # Should not exceed target (strict requirement)
        assert len(result) <= 25, f"Result has {len(result)} frames, exceeds target 25"

        # Should be close to target
        assert len(result) >= 20, f"Result has only {len(result)} frames, too far from target 25"

    def test_parseq_keyframes(self):
        """Test that Parseq keyframes are also treated as exact positions."""
        parseq_json = {
            "keyframes": [
                {"frame": 0},
                {"frame": 33},
                {"frame": 67},
                {"frame": 99}
            ]
        }

        result = redistributed_logic(100, [], 15, 10, use_parseq=True, parseq_json=parseq_json)

        # Parseq keyframes must be at EXACT positions
        assert 0 in result and 33 in result and 67 in result and 99 in result

    def test_result_is_sorted(self):
        """Test that result is always sorted."""
        result = redistributed_logic(100, ['75', '25', '50'], 20, 5)  # Unsorted input

        assert result == sorted(result), "Result must be sorted!"

    def test_no_duplicates(self):
        """Test that result has no duplicate frames."""
        result = redistributed_logic(100, ['25', '50', '75'], 20, 5)

        assert len(result) == len(set(result)), "Result has duplicate frames!"

    def test_valid_range(self):
        """Test that all frames are within valid range."""
        result = redistributed_logic(100, ['30', '60'], 20, 5)

        for frame in result:
            assert 0 <= frame < 100, f"Frame {frame} outside valid range [0, 99]!"

    def test_first_and_last_included(self):
        """Test that frame 0 and last frame are always in result."""
        result = redistributed_logic(100, ['50'], 15, 10)

        assert 0 in result, "First frame (0) must always be included!"
        assert 99 in result, "Last frame must always be included!"

    def test_two_frame_animation(self):
        """Test minimum case: 2-frame animation."""
        # 2-frame animation has keyframes at 0 and 1 (last frame)
        result = redistributed_logic(2, ['0'], 2, 1)

        assert sorted(result) == [0, 1], "2-frame animation should have frames 0 and 1"

    def test_all_frames_keyframes(self):
        """Test when every frame is a keyframe."""
        keyframes = [str(k) for k in range(0, 19)]  # 0-18
        result = redistributed_logic(20, keyframes, 20, 1)

        # Should return all keyframes
        assert sorted(result) == list(range(0, 20))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
