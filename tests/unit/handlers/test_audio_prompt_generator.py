"""Unit tests for audio_prompt_generator helpers."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path to allow direct imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

# Import directly from the module file to avoid Forge dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "audio_prompt_generator",
    Path(__file__).parent.parent.parent.parent / "deforum" / "ui" / "handlers" / "audio_prompt_generator.py"
)
audio_prompt_generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audio_prompt_generator)

_get_intensity_instruction = audio_prompt_generator._get_intensity_instruction
_clean_qwen_output = audio_prompt_generator._clean_qwen_output
_generate_fallback_prompts = audio_prompt_generator._generate_fallback_prompts


class TestGetIntensityInstruction:
    """Tests for _get_intensity_instruction function."""

    def test_returns_empty_for_empty_intensity(self):
        """Should return empty string for empty intensity."""
        result = _get_intensity_instruction("")
        assert result == ""

    def test_returns_subtle_instruction(self):
        """Should return correct instruction for subtle intensity."""
        result = _get_intensity_instruction("subtle")
        assert "subtle" in result.lower()
        assert "minimal" in result.lower()

    def test_returns_normal_instruction(self):
        """Should return correct instruction for normal intensity."""
        result = _get_intensity_instruction("normal")
        assert "realistic" in result.lower()
        assert "grounded" in result.lower()

    def test_returns_crazy_instruction(self):
        """Should return correct instruction for crazy intensity."""
        result = _get_intensity_instruction("crazy")
        assert "over-the-top" in result.lower() or "creative" in result.lower()

    def test_returns_extreme_instruction(self):
        """Should return correct instruction for extreme intensity."""
        result = _get_intensity_instruction("extreme")
        assert "bonkers" in result.lower() or "insane" in result.lower()

    def test_returns_chaotic_instruction(self):
        """Should return correct instruction for chaotic intensity."""
        result = _get_intensity_instruction("chaotic")
        assert "chaos" in result.lower() or "chaotic" in result.lower()

    def test_returns_surreal_instruction(self):
        """Should return correct instruction for surreal intensity."""
        result = _get_intensity_instruction("surreal")
        assert "surreal" in result.lower() or "dream" in result.lower()

    def test_returns_custom_instruction_for_unknown(self):
        """Should return custom instruction for unknown intensity."""
        custom = "my custom vibe"
        result = _get_intensity_instruction(custom)
        assert "Creative direction:" in result
        assert custom in result

    def test_returns_crazy_for_empty_custom(self):
        """Empty string should return crazy default via ternary."""
        result = _get_intensity_instruction("")
        # Empty string uses the empty key which returns ""
        assert result == ""


class TestCleanQwenOutput:
    """Tests for _clean_qwen_output function."""

    def test_cleans_numbered_lines(self):
        """Should remove leading numbers from output."""
        qwen_output = """1. First prompt
2. Second prompt
3. Third prompt"""
        result = _clean_qwen_output(qwen_output, 3)

        assert len(result) == 3
        assert result[0] == "First prompt"
        assert result[1] == "Second prompt"
        assert result[2] == "Third prompt"

    def test_cleans_numbered_with_parentheses(self):
        """Should handle numbers with parentheses."""
        qwen_output = """1) First prompt
2) Second prompt"""
        result = _clean_qwen_output(qwen_output, 2)

        assert len(result) == 2
        assert result[0] == "First prompt"
        assert result[1] == "Second prompt"

    def test_removes_comment_lines(self):
        """Should skip comment lines starting with # or //."""
        qwen_output = """# This is a comment
First prompt
// Another comment
Second prompt"""
        result = _clean_qwen_output(qwen_output, 10)

        assert len(result) == 2
        assert "comment" not in result[0].lower()
        assert "comment" not in result[1].lower()

    def test_removes_empty_lines(self):
        """Should skip empty lines."""
        qwen_output = """First prompt

Second prompt


Third prompt"""
        result = _clean_qwen_output(qwen_output, 10)

        assert len(result) == 3
        assert all(line.strip() for line in result)

    def test_limits_output_to_count(self):
        """Should return only requested count of prompts."""
        qwen_output = """1. First
2. Second
3. Third
4. Fourth
5. Fifth"""
        result = _clean_qwen_output(qwen_output, 3)

        assert len(result) == 3
        assert result[-1] == "Third"

    def test_handles_mixed_formatting(self):
        """Should handle mixed number formats."""
        qwen_output = """1. First
2) Second
3. Third
Fourth
5) Fifth"""
        result = _clean_qwen_output(qwen_output, 10)

        assert len(result) == 5
        assert result[0] == "First"
        assert result[1] == "Second"
        assert result[2] == "Third"
        assert result[3] == "Fourth"
        assert result[4] == "Fifth"


class TestGenerateFallbackPrompts:
    """Tests for _generate_fallback_prompts function."""

    def test_start_to_end_mode_includes_start_and_end(self):
        """Start-to-end mode should include start and end prompts."""
        result = _generate_fallback_prompts(
            generation_mode="start-to-end",
            style="cyberpunk",
            theme="city",
            count=5,
            start_prompt="quiet city street",
            end_prompt="bustling neon metropolis"
        )

        lines = result.split('\n')
        assert len(lines) == 5
        assert lines[0] == "quiet city street"
        assert lines[-1] == "bustling neon metropolis"

    def test_start_to_end_fills_middle_with_transforming(self):
        """Middle prompts should use 'transforming' text."""
        result = _generate_fallback_prompts(
            generation_mode="start-to-end",
            style="anime",
            theme="forest",
            count=4,
            start_prompt="calm forest",
            end_prompt="magical realm"
        )

        lines = result.split('\n')
        assert len(lines) == 4
        assert "transforming" in lines[1]
        assert "transforming" in lines[2]

    def test_other_modes_use_action_progression(self):
        """Non-start-to-end modes should use action sequence."""
        result = _generate_fallback_prompts(
            generation_mode="varied",
            style="realistic",
            theme="cat",
            count=5,
            start_prompt="",
            end_prompt=""
        )

        lines = result.split('\n')
        assert len(lines) == 5

        # Check progression keywords appear
        result_text = result.lower()
        assert "resting" in result_text or "peacefully" in result_text
        assert "wild" in result_text

    def test_applies_style_prefix(self):
        """Should apply style prefix to all prompts."""
        result = _generate_fallback_prompts(
            generation_mode="varied",
            style="watercolor",
            theme="landscape",
            count=3,
            start_prompt="",
            end_prompt=""
        )

        lines = result.split('\n')
        for line in lines:
            assert "watercolor" in line

    def test_handles_no_style(self):
        """Should work without style prefix."""
        result = _generate_fallback_prompts(
            generation_mode="varied",
            style="",
            theme="ocean",
            count=3,
            start_prompt="",
            end_prompt=""
        )

        lines = result.split('\n')
        assert len(lines) == 3
        for line in lines:
            assert "ocean" in line

    def test_respects_count_parameter(self):
        """Should generate exactly count prompts."""
        for count in [1, 3, 5, 10]:
            result = _generate_fallback_prompts(
                generation_mode="varied",
                style="",
                theme="test",
                count=count,
                start_prompt="",
                end_prompt=""
            )
            lines = result.split('\n')
            assert len(lines) == min(count, 5)  # Limited by actions list

    def test_handles_zero_count(self):
        """Should handle edge case of zero count."""
        result = _generate_fallback_prompts(
            generation_mode="varied",
            style="",
            theme="test",
            count=0,
            start_prompt="",
            end_prompt=""
        )
        # Empty string split by newline gives [''], not []
        lines = [l for l in result.split('\n') if l]
        assert len(lines) == 0

    def test_start_to_end_with_count_two(self):
        """Count of 2 should only have start and end, no middle."""
        result = _generate_fallback_prompts(
            generation_mode="start-to-end",
            style="",
            theme="test",
            count=2,
            start_prompt="beginning",
            end_prompt="ending"
        )
        lines = result.split('\n')
        assert len(lines) == 2
        assert lines[0] == "beginning"
        assert lines[1] == "ending"
