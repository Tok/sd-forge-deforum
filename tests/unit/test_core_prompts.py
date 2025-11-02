"""Unit tests for deforum.core.prompts module."""

import pytest

from deforum.core.prompts import PromptScheduler, prepare_prompt


class TestPromptSchedulerInit:
    """Test PromptScheduler initialization."""

    def test_init_with_string(self):
        """Initialize with prompt string."""
        scheduler = PromptScheduler(
            prompts="0: a cat, 30: a dog",
            max_frames=100
        )
        assert scheduler.max_frames == 100
        assert len(scheduler.prompt_dict) == 2
        assert scheduler.prompt_dict[0] == "a cat"
        assert scheduler.prompt_dict[30] == "a dog"

    def test_init_with_dict(self):
        """Initialize with prompt dictionary."""
        scheduler = PromptScheduler(
            prompts={0: "a cat", 30: "a dog"},
            max_frames=100
        )
        assert scheduler.max_frames == 100
        assert len(scheduler.prompt_dict) == 2
        assert scheduler.prompt_dict[0] == "a cat"
        assert scheduler.prompt_dict[30] == "a dog"

    def test_init_creates_series(self):
        """Initialization creates interpolated series."""
        scheduler = PromptScheduler(
            prompts="0: cat, 50: dog",
            max_frames=100
        )
        assert len(scheduler.prompt_series) == 100

    def test_init_single_prompt(self):
        """Initialize with single prompt."""
        scheduler = PromptScheduler(
            prompts="0: constant prompt",
            max_frames=100
        )
        assert scheduler.prompt_dict[0] == "constant prompt"


class TestPromptSchedulerGetPrompt:
    """Test PromptScheduler.get_prompt()."""

    def test_get_prompt_at_keyframe(self):
        """Get prompt at exact keyframe."""
        scheduler = PromptScheduler(
            prompts="0: a cat, 30: a dog, 60: a bird",
            max_frames=100
        )
        assert scheduler.get_prompt(0) == "a cat"
        assert scheduler.get_prompt(30) == "a dog"
        assert scheduler.get_prompt(60) == "a bird"

    def test_get_prompt_interpolated(self):
        """Get interpolated prompt between keyframes."""
        scheduler = PromptScheduler(
            prompts="0: cat, 30: dog",
            max_frames=100
        )
        # Middle frame should have both
        prompt = scheduler.get_prompt(15)
        assert "cat" in prompt.lower()
        assert "dog" in prompt.lower()

    def test_get_prompt_with_expressions(self):
        """Get prompt with expression evaluation."""
        scheduler = PromptScheduler(
            prompts="0: value is `t`",
            max_frames=100
        )
        prompt = scheduler.get_prompt(25, evaluate_expressions=True)
        assert "25" in prompt

    def test_get_prompt_without_expression_eval(self):
        """Get prompt without evaluating expressions."""
        scheduler = PromptScheduler(
            prompts="0: value is `t`",
            max_frames=100
        )
        prompt = scheduler.get_prompt(25, evaluate_expressions=False)
        assert "`t`" in prompt

    def test_get_prompt_out_of_range_low(self):
        """Getting prompt with negative index raises error."""
        scheduler = PromptScheduler(
            prompts="0: cat",
            max_frames=100
        )
        with pytest.raises(ValueError, match="out of range"):
            scheduler.get_prompt(-1)

    def test_get_prompt_out_of_range_high(self):
        """Getting prompt beyond max_frames raises error."""
        scheduler = PromptScheduler(
            prompts="0: cat",
            max_frames=100
        )
        with pytest.raises(ValueError, match="out of range"):
            scheduler.get_prompt(100)

    def test_get_prompt_last_frame(self):
        """Get prompt at last valid frame."""
        scheduler = PromptScheduler(
            prompts="0: cat, 99: dog",
            max_frames=100
        )
        # Frame 99 is valid (0-indexed, 100 frames)
        prompt = scheduler.get_prompt(99)
        assert "dog" in prompt.lower()


class TestPromptSchedulerGetPromptSplit:
    """Test PromptScheduler.get_prompt_split()."""

    def test_split_with_negative(self):
        """Split prompt with negative part."""
        scheduler = PromptScheduler(
            prompts="0: a cat --neg blurry",
            max_frames=100
        )
        pos, neg = scheduler.get_prompt_split(0)
        assert pos == "a cat"
        assert neg == "blurry"

    def test_split_without_negative(self):
        """Split prompt without negative part."""
        scheduler = PromptScheduler(
            prompts="0: a cat",
            max_frames=100
        )
        pos, neg = scheduler.get_prompt_split(0)
        assert pos == "a cat"
        assert neg == ""

    def test_split_interpolated_prompts(self):
        """Split interpolated prompts with negatives."""
        scheduler = PromptScheduler(
            prompts="0: cat --neg blurry, 50: dog --neg low quality",
            max_frames=100
        )
        pos, neg = scheduler.get_prompt_split(25)
        # Should contain parts of both
        assert "cat" in pos.lower() or "dog" in pos.lower()
        assert len(neg) > 0  # Should have negative content

    def test_split_with_expression_eval(self):
        """Split with expression evaluation enabled."""
        scheduler = PromptScheduler(
            prompts="0: frame `t` --neg bad",
            max_frames=100
        )
        pos, neg = scheduler.get_prompt_split(42, evaluate_expressions=True)
        assert "42" in pos
        assert neg == "bad"


class TestPromptSchedulerUpdatePrompts:
    """Test PromptScheduler.update_prompts()."""

    def test_update_with_string(self):
        """Update prompts with new string."""
        scheduler = PromptScheduler(
            prompts="0: cat",
            max_frames=100
        )
        scheduler.update_prompts("0: dog, 50: bird")
        assert scheduler.get_prompt(0) == "dog"
        assert scheduler.get_prompt(50) == "bird"

    def test_update_with_dict(self):
        """Update prompts with new dictionary."""
        scheduler = PromptScheduler(
            prompts="0: cat",
            max_frames=100
        )
        scheduler.update_prompts({0: "dog", 50: "bird"})
        assert scheduler.get_prompt(0) == "dog"
        assert scheduler.get_prompt(50) == "bird"

    def test_update_recreates_series(self):
        """Update recreates interpolated series."""
        scheduler = PromptScheduler(
            prompts="0: cat, 50: dog",
            max_frames=100
        )
        old_series_id = id(scheduler.prompt_series)
        scheduler.update_prompts("0: bird, 50: fish")
        # Series should be recreated
        assert scheduler.get_prompt(0) == "bird"
        assert scheduler.get_prompt(50) == "fish"


class TestPromptSchedulerGetKeyframes:
    """Test PromptScheduler.get_keyframes()."""

    def test_get_keyframes_returns_copy(self):
        """get_keyframes returns a copy, not reference."""
        scheduler = PromptScheduler(
            prompts="0: cat, 30: dog",
            max_frames=100
        )
        keyframes = scheduler.get_keyframes()
        keyframes[50] = "bird"
        # Original should be unchanged
        assert 50 not in scheduler.prompt_dict

    def test_get_keyframes_content(self):
        """get_keyframes returns correct content."""
        scheduler = PromptScheduler(
            prompts="0: cat, 30: dog, 60: bird",
            max_frames=100
        )
        keyframes = scheduler.get_keyframes()
        assert len(keyframes) == 3
        assert keyframes[0] == "cat"
        assert keyframes[30] == "dog"
        assert keyframes[60] == "bird"


class TestPromptSchedulerHasNegativePrompts:
    """Test PromptScheduler.has_negative_prompts()."""

    def test_has_negative_prompts_true(self):
        """Detects negative prompts."""
        scheduler = PromptScheduler(
            prompts="0: cat --neg blurry",
            max_frames=100
        )
        assert scheduler.has_negative_prompts() is True

    def test_has_negative_prompts_false(self):
        """Detects absence of negative prompts."""
        scheduler = PromptScheduler(
            prompts="0: cat, 30: dog",
            max_frames=100
        )
        assert scheduler.has_negative_prompts() is False

    def test_has_negative_prompts_multiple_keyframes(self):
        """Detects negative in any keyframe."""
        scheduler = PromptScheduler(
            prompts="0: cat, 30: dog --neg blurry, 60: bird",
            max_frames=100
        )
        assert scheduler.has_negative_prompts() is True


class TestPromptSchedulerRepr:
    """Test PromptScheduler string representation."""

    def test_repr_format(self):
        """Repr contains key information."""
        scheduler = PromptScheduler(
            prompts="0: cat, 30: dog",
            max_frames=100
        )
        repr_str = repr(scheduler)
        assert "max_frames=100" in repr_str
        assert "keyframes=2" in repr_str


class TestPreparePromptLegacyFunction:
    """Test legacy prepare_prompt function."""

    def test_prepare_prompt_simple(self):
        """Test simple prompt preparation (no console output)."""
        result = prepare_prompt("a cat", max_frames=100, seed=42, frame_idx=0)
        assert result == "a cat"

    def test_prepare_prompt_with_negative(self):
        """Test prompt with negative part (no console output)."""
        result = prepare_prompt(
            "a cat --neg blurry",
            max_frames=100,
            seed=42,
            frame_idx=0
        )
        assert result == "a cat --neg blurry"

    def test_prepare_prompt_with_expression(self):
        """Test prompt with expression evaluation (no console output)."""
        result = prepare_prompt(
            "frame `t`",
            max_frames=100,
            seed=42,
            frame_idx=25
        )
        assert "25" in result


class TestPromptSchedulerIntegration:
    """Integration tests for PromptScheduler."""

    def test_typical_animation_workflow(self):
        """Simulate typical animation prompt workflow."""
        # 100 frame animation with 3 prompt changes
        scheduler = PromptScheduler(
            prompts="0: a cat sitting, 33: a cat walking, 66: a cat jumping",
            max_frames=100
        )

        # Check keyframes
        assert "sitting" in scheduler.get_prompt(0).lower()
        assert "walking" in scheduler.get_prompt(33).lower()
        assert "jumping" in scheduler.get_prompt(66).lower()

        # Check interpolation
        mid_prompt = scheduler.get_prompt(16)
        assert "cat" in mid_prompt.lower()

    def test_complex_prompt_with_weights_and_negatives(self):
        """Test complex prompt with all features."""
        scheduler = PromptScheduler(
            prompts={
                0: "(beautiful cat:1.2), (sitting:0.8) --neg (blurry:1.5)",
                50: "(beautiful dog:1.2), (running:0.8) --neg (blurry:1.5)"
            },
            max_frames=100
        )

        # Get split prompts
        pos, neg = scheduler.get_prompt_split(0)
        assert "cat" in pos.lower()
        assert "blurry" in neg.lower()

        pos, neg = scheduler.get_prompt_split(50)
        assert "dog" in pos.lower()
        assert "blurry" in neg.lower()

    def test_dynamic_prompt_changes(self):
        """Test updating prompts mid-animation."""
        scheduler = PromptScheduler(
            prompts="0: static scene",
            max_frames=100
        )

        # Start with static
        assert scheduler.get_prompt(50) == "static scene"

        # Update to animated
        scheduler.update_prompts("0: scene A, 50: scene B, 100: scene C")
        assert "A" in scheduler.get_prompt(0)
        assert "B" in scheduler.get_prompt(50)

    def test_expression_evaluation_throughout_animation(self):
        """Test math expressions across frames."""
        scheduler = PromptScheduler(
            prompts="0: strength `sin(t/10)`",
            max_frames=100
        )

        # Get prompts at different frames
        prompts = [scheduler.get_prompt(i) for i in range(0, 100, 10)]

        # Should have different evaluated values
        unique_values = len(set(prompts))
        assert unique_values > 1  # Not all the same

    def test_long_animation_keyframe_spacing(self):
        """Test long animation with widely spaced keyframes."""
        scheduler = PromptScheduler(
            prompts="0: start, 500: middle, 1000: end",
            max_frames=1000
        )

        # Check interpolation works at various points
        assert "start" in scheduler.get_prompt(0).lower()
        assert "end" in scheduler.get_prompt(999).lower()

        # Middle frames should have interpolated content
        mid = scheduler.get_prompt(250)
        assert len(mid) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
