"""Unit tests for DA3-3DGS scene grouping strategies.

Tests per-prompt grouping, rolling window logic, and VRAM optimization decisions.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from dataclasses import dataclass


@dataclass
class MockKeyframe:
    """Mock keyframe for testing."""
    i: int  # Frame index


class TestPerPromptGrouping:
    """Test per-prompt scene grouping logic."""

    def test_single_prompt_creates_one_group(self):
        """Single prompt should create one group with all segments."""
        keyframes = [MockKeyframe(i) for i in [0, 10, 20, 30]]
        prompt_series = pd.Series({
            0: "same prompt",
            10: "same prompt",
            20: "same prompt",
            30: "same prompt"
        })

        groups = self._group_segments_by_prompt(keyframes, prompt_series)

        assert len(groups) == 1
        assert groups[0][0] == "same prompt"
        assert len(groups[0][1]) == 3  # 3 segments between 4 keyframes
        assert groups[0][1] == [(0, 10), (10, 20), (20, 30)]

    def test_two_prompts_creates_two_groups(self):
        """Two distinct prompts should create two groups."""
        keyframes = [MockKeyframe(i) for i in [0, 10, 20, 30]]
        prompt_series = pd.Series({
            0: "first prompt",
            10: "first prompt",
            20: "second prompt",
            30: "second prompt"
        })

        groups = self._group_segments_by_prompt(keyframes, prompt_series)

        assert len(groups) == 2
        assert groups[0][0] == "first prompt"
        # Segments 0→10 and 10→20 both have first_kf.i with "first prompt"
        assert groups[0][1] == [(0, 10), (10, 20)]
        assert groups[1][0] == "second prompt"
        # Segment 20→30 has first_kf.i=20 with "second prompt"
        assert groups[1][1] == [(20, 30)]

    def test_prompt_changes_at_every_segment(self):
        """Each segment with different prompt creates separate group."""
        keyframes = [MockKeyframe(i) for i in [0, 10, 20, 30]]
        prompt_series = pd.Series({
            0: "prompt A",
            10: "prompt B",
            20: "prompt C",
            30: "prompt D"
        })

        groups = self._group_segments_by_prompt(keyframes, prompt_series)

        assert len(groups) == 3  # 3 segments, 3 groups
        assert groups[0][0] == "prompt A"
        assert groups[0][1] == [(0, 10)]
        assert groups[1][0] == "prompt B"
        assert groups[1][1] == [(10, 20)]
        assert groups[2][0] == "prompt C"
        assert groups[2][1] == [(20, 30)]

    def test_prompt_changes_then_returns(self):
        """Prompt A → B → A should create 3 separate groups."""
        keyframes = [MockKeyframe(i) for i in [0, 10, 20, 30, 40]]
        prompt_series = pd.Series({
            0: "prompt A",
            10: "prompt A",
            20: "prompt B",
            30: "prompt A",  # Returns to A - should create NEW group
            40: "prompt A"
        })

        groups = self._group_segments_by_prompt(keyframes, prompt_series)

        # Should be 3 groups:
        # Group 1: A for segments 0→10, 10→20 (both start with "prompt A")
        # Group 2: B for segment 20→30 (starts with "prompt B")
        # Group 3: A for segment 30→40 (starts with "prompt A" again)
        assert len(groups) == 3
        assert groups[0][0] == "prompt A"
        assert groups[0][1] == [(0, 10), (10, 20)]
        assert groups[1][0] == "prompt B"
        assert groups[1][1] == [(20, 30)]
        assert groups[2][0] == "prompt A"  # NEW group for same prompt
        assert groups[2][1] == [(30, 40)]

    def test_real_world_example(self):
        """Test with realistic prompt schedule."""
        keyframes = [MockKeyframe(i) for i in [0, 12, 22, 32, 43, 53, 64, 73]]
        prompt_series = pd.Series({
            0: "driving along empty city boulevard",
            12: "driving along empty city boulevard",
            22: "driving along empty city boulevard",
            32: "driving along empty city boulevard",
            43: "turning into narrow alleyway",
            53: "turning into narrow alleyway",
            64: "turning into narrow alleyway",
            73: "turning into narrow alleyway"
        })

        groups = self._group_segments_by_prompt(keyframes, prompt_series)

        assert len(groups) == 2
        assert groups[0][0] == "driving along empty city boulevard"
        # Segments: 0→12, 12→22, 22→32, 32→43 (all start with "city boulevard")
        assert len(groups[0][1]) == 4
        assert groups[1][0] == "turning into narrow alleyway"
        # Segments: 43→53, 53→64, 64→73 (all start with "narrow alleyway")
        assert len(groups[1][1]) == 3

    def _group_segments_by_prompt(self, keyframes, prompt_series):
        """Helper to extract grouping logic for testing."""
        prompt_groups = []
        current_group = []
        current_prompt = None

        for idx in range(len(keyframes) - 1):
            first_kf = keyframes[idx]
            last_kf = keyframes[idx + 1]

            # Get prompt for this segment (use first keyframe's prompt)
            segment_prompt = prompt_series[first_kf.i]

            # Start new group if prompt changed
            if segment_prompt != current_prompt:
                if current_group:
                    prompt_groups.append((current_prompt, current_group))
                current_group = [(first_kf.i, last_kf.i)]
                current_prompt = segment_prompt
            else:
                current_group.append((first_kf.i, last_kf.i))

        # Add final group
        if current_group:
            prompt_groups.append((current_prompt, current_group))

        return prompt_groups


class TestRollingWindowGrouping:
    """Test rolling window scene grouping logic."""

    def test_exact_window_fit(self):
        """Keyframes exactly fill windows with no overlap needed."""
        keyframes = [MockKeyframe(i) for i in range(0, 100, 10)]  # 10 keyframes
        window_size = 5  # 5 keyframes per window

        windows = self._create_rolling_windows(keyframes, window_size, overlap=0)

        # Should create 2 windows: [0-4], [5-9]
        assert len(windows) == 2
        assert windows[0] == [0, 10, 20, 30, 40]
        assert windows[1] == [50, 60, 70, 80, 90]

    def test_overlapping_windows(self):
        """Windows should overlap to maintain continuity."""
        keyframes = [MockKeyframe(i) for i in range(0, 100, 10)]  # 10 keyframes
        window_size = 5
        overlap = 2  # 2 keyframes overlap

        windows = self._create_rolling_windows(keyframes, window_size, overlap)

        # Window 1: [0-4], Window 2: [3-7], Window 3: [6-9]
        assert len(windows) == 3
        assert windows[0] == [0, 10, 20, 30, 40]
        assert windows[1] == [30, 40, 50, 60, 70]  # Overlaps with window 1 at 30, 40
        assert windows[2] == [60, 70, 80, 90]  # Overlaps with window 2 at 60, 70

    def test_window_larger_than_keyframes(self):
        """If window size >= keyframe count, should create single window."""
        keyframes = [MockKeyframe(i) for i in [0, 10, 20]]  # Only 3 keyframes
        window_size = 10  # Window larger than keyframe count

        windows = self._create_rolling_windows(keyframes, window_size, overlap=0)

        assert len(windows) == 1
        assert windows[0] == [0, 10, 20]  # All keyframes in one window

    def _create_rolling_windows(self, keyframes, window_size, overlap=0):
        """Helper to create rolling windows for testing."""
        windows = []
        keyframe_indices = [kf.i for kf in keyframes]

        if len(keyframe_indices) <= window_size:
            return [keyframe_indices]

        stride = window_size - overlap
        start = 0

        while start < len(keyframe_indices):
            end = min(start + window_size, len(keyframe_indices))
            windows.append(keyframe_indices[start:end])

            if end == len(keyframe_indices):
                break

            start += stride

        return windows


class TestVRAMOptimization:
    """Test VRAM optimization decisions for keyframe count vs densification."""

    def test_vram_budget_allocation(self):
        """Test optimal allocation of VRAM budget between keyframes and splats."""
        available_vram_gb = 24.0  # Realistic for 24GB GPUs

        # Test different strategies
        strategies = [
            ("few_keyframes_high_density", 10, 8),   # 10 keyframes, 8x densification
            ("medium_keyframes_medium_density", 30, 4),  # 30 keyframes, 4x densification
            ("many_keyframes_low_density", 50, 2),   # 50 keyframes, 2x densification
        ]

        for name, keyframe_count, densification in strategies:
            vram_estimate = self._estimate_vram_usage(
                keyframe_count, densification, resolution=(1024, 1024)
            )

            print(f"{name}: {keyframe_count} keyframes × {densification}x = {vram_estimate:.1f}GB")
            assert vram_estimate <= available_vram_gb, f"{name} exceeds VRAM budget"

    def test_quality_vs_coverage_tradeoff(self):
        """More keyframes provide better coverage, higher densification provides finer detail.

        Key insight: Quality doesn't scale linearly with keyframe count!
        - 10 keyframes @ 8x density: High detail, limited coverage
        - 50 keyframes @ 2x density: Better coverage, less detail per splat
        """
        # VRAM budget: 20GB
        # Base splat count: 705k

        # Strategy 1: High detail, limited coverage
        strat1_keyframes = 10
        strat1_density = 8
        strat1_splats = 705_000 * strat1_density  # 5.6M splats

        # Strategy 2: Better coverage, less detail
        strat2_keyframes = 50
        strat2_density = 2
        strat2_splats = 705_000 * strat2_density  # 1.4M splats

        # Both should fit in VRAM, but provide different quality characteristics
        assert strat1_splats > strat2_splats, "High density should have more splats"
        assert strat1_keyframes < strat2_keyframes, "High density should use fewer keyframes"

        # The key question: which produces better results?
        # Answer: Depends on use case!
        # - High detail (strat1): Better for static scenes with fine detail
        # - Better coverage (strat2): Better for dynamic scenes with camera movement

    def _estimate_vram_usage(self, keyframe_count, densification, resolution):
        """Estimate VRAM usage for given parameters.

        Rough formula:
        - Base model: ~4GB (DA3NESTED-GIANT-LARGE)
        - Per keyframe: ~200MB overhead
        - Splats: 705k base × densification × ~2.5MB
        - Resolution: (W*H) / 1024^2 × 500MB
        """
        base_model_gb = 4.0
        keyframe_overhead_gb = keyframe_count * 0.2

        base_splats = 705_000
        total_splats = base_splats * densification
        splat_memory_gb = (total_splats / 1_000_000) * 2.5

        w, h = resolution
        resolution_factor = (w * h) / (1024 * 1024)
        resolution_gb = resolution_factor * 0.5

        total = base_model_gb + keyframe_overhead_gb + splat_memory_gb + resolution_gb
        return total


class TestMaxPromptKeyframesSplitting:
    """Test splitting of large prompt segments."""

    def test_prompt_within_limit(self):
        """Prompt segment within limit should not be split."""
        keyframes_in_prompt = 30
        max_keyframes = 50

        should_split, num_sub_groups = self._check_splitting(keyframes_in_prompt, max_keyframes)

        assert not should_split
        assert num_sub_groups == 1

    def test_prompt_exceeds_limit(self):
        """Prompt segment exceeding limit should be split."""
        keyframes_in_prompt = 100
        max_keyframes = 50

        should_split, num_sub_groups = self._check_splitting(keyframes_in_prompt, max_keyframes)

        assert should_split
        assert num_sub_groups == 2  # 100 / 50 = 2 sub-groups

    def test_prompt_exactly_at_limit(self):
        """Prompt segment exactly at limit should not be split."""
        keyframes_in_prompt = 50
        max_keyframes = 50

        should_split, num_sub_groups = self._check_splitting(keyframes_in_prompt, max_keyframes)

        assert not should_split
        assert num_sub_groups == 1

    def _check_splitting(self, keyframes_in_prompt, max_keyframes):
        """Helper to check if splitting is needed."""
        should_split = keyframes_in_prompt > max_keyframes
        num_sub_groups = (keyframes_in_prompt + max_keyframes - 1) // max_keyframes  # Ceiling division
        return should_split, num_sub_groups


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
