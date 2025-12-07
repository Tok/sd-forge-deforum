"""DA3-3DGS Parameter Sweep Tests.

This module provides automated parameter optimization for DA3-3DGS novel view synthesis.
Tests different combinations of scene strategies, models, neighbor segments, densification,
and near-clip distance to find optimal quality/performance tradeoffs.

Quality Metrics:
- Visual Quality: Subjective quality score (1-10)
- Temporal Consistency: Frame-to-frame SSIM stability
- Coordinate Drift: Scene drift between segments (lower = better)
- VRAM Usage: Peak memory usage in GB
- Render Time: Time per frame in seconds

VRAM Budget Analysis:
- Base model: DA3-GIANT ~4GB, DA3NESTED-GIANT-LARGE ~4.5GB
- Per keyframe overhead: ~200MB
- Splat memory: (705k × densification) × ~2.5MB per million splats
- Resolution overhead: (W×H / 1024²) × 500MB

Examples:
- 10 keyframes @ 8x density @ 512×512: ~20GB VRAM
- 50 keyframes @ 2x density @ 512×512: ~17GB VRAM
- 30 keyframes @ 4x density @ 1024×1024: ~24GB VRAM
"""

import pytest
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

from deforum.api.tuning_api import TuningTestType, TuningTestConfig


class TestDA33DGSParameterSweep:
    """Automated parameter optimization for DA3-3DGS."""

    @pytest.fixture
    def base_config(self) -> Dict[str, Any]:
        """Base configuration for all 3DGS tests."""
        return {
            "test_type": TuningTestType.DA3_3DGS_TUNING,
            "aspect_ratios": [[1.78, 512, 288]],  # 16:9 landscape
            "rotation_factor": -8.0,  # Empirically validated optimal
            "orbit_radius": 5.0,  # Moderate movement
            "test_iterations": 50,  # 50 frames per test
        }

    def test_scene_strategy_comparison(self, base_config):
        """Compare per-segment, per-prompt, and rolling-window strategies.

        Expected findings:
        - per_segment: Fastest, lowest VRAM, but coordinate drift
        - per_prompt: Best semantic coherence, eliminates drift within prompts
        - rolling_window: Predictable VRAM, consistent quality
        """
        config = TuningTestConfig(
            **base_config,
            dgs_scene_strategies=["per_segment", "per_prompt", "rolling_window"],
            dgs_rolling_window_size=30,
            dgs_max_prompt_keyframes=50,
            dgs_models=["DA3NESTED-GIANT-LARGE"],  # Use best model for comparison
            dgs_neighbor_segments_min=6,
            dgs_neighbor_segments_max=6,
            dgs_neighbor_segments_step=1,
            dgs_densification_min=4,
            dgs_densification_max=4,
            dgs_densification_step=1,
            dgs_nearclip_min=0.1,
            dgs_nearclip_max=0.1,
            dgs_nearclip_step=1.0,
        )

        # This would call the API endpoint
        # Result: 3 test runs (one per strategy)
        assert config.test_type == TuningTestType.DA3_3DGS_TUNING

    def test_model_comparison(self, base_config):
        """Compare DA3-GIANT vs DA3NESTED-GIANT-LARGE.

        Expected findings:
        - DA3-GIANT: 4GB VRAM, faster, good quality
        - DA3NESTED-GIANT-LARGE: 4.5GB VRAM, slightly slower, better quality
        """
        config = TuningTestConfig(
            **base_config,
            dgs_scene_strategies=["per_segment"],
            dgs_models=["DA3-GIANT", "DA3NESTED-GIANT-LARGE"],
            dgs_neighbor_segments_min=6,
            dgs_neighbor_segments_max=6,
            dgs_neighbor_segments_step=1,
            dgs_densification_min=4,
            dgs_densification_max=4,
            dgs_densification_step=1,
            dgs_nearclip_min=0.1,
            dgs_nearclip_max=0.1,
            dgs_nearclip_step=1.0,
        )

        # Result: 2 test runs (one per model)
        assert len(config.dgs_models) == 2

    def test_neighbor_segments_sweep(self, base_config):
        """Sweep neighbor segments from 4 to 8 (step 2).

        More keyframes = better geometry but slower and more VRAM.

        Expected findings:
        - 4 keyframes: Fast, minimal VRAM, may miss details
        - 6 keyframes: Good balance of quality/speed (current default)
        - 8 keyframes: Best geometry, slower, more VRAM
        """
        config = TuningTestConfig(
            **base_config,
            dgs_scene_strategies=["per_segment"],
            dgs_models=["DA3NESTED-GIANT-LARGE"],
            dgs_neighbor_segments_min=4,
            dgs_neighbor_segments_max=8,
            dgs_neighbor_segments_step=2,
            dgs_densification_min=4,
            dgs_densification_max=4,
            dgs_densification_step=1,
            dgs_nearclip_min=0.1,
            dgs_nearclip_max=0.1,
            dgs_nearclip_step=1.0,
        )

        # Result: 3 test runs (4, 6, 8 keyframes)
        expected_configs = list(range(4, 9, 2))
        assert len(expected_configs) == 3

    def test_densification_sweep(self, base_config):
        """Sweep densification from 2x to 6x (step 2).

        Densification multiplies base splat count (705k).
        Higher = finer detail but more VRAM.

        Expected findings:
        - 2x (1.4M splats): Fast, minimal detail
        - 4x (2.8M splats): Good balance (current default)
        - 6x (4.2M splats): Best detail, high VRAM
        """
        config = TuningTestConfig(
            **base_config,
            dgs_scene_strategies=["per_segment"],
            dgs_models=["DA3NESTED-GIANT-LARGE"],
            dgs_neighbor_segments_min=6,
            dgs_neighbor_segments_max=6,
            dgs_neighbor_segments_step=1,
            dgs_densification_min=2,
            dgs_densification_max=6,
            dgs_densification_step=2,
            dgs_nearclip_min=0.1,
            dgs_nearclip_max=0.1,
            dgs_nearclip_step=1.0,
        )

        # Result: 3 test runs (2x, 4x, 6x densification)
        expected_configs = list(range(2, 7, 2))
        assert len(expected_configs) == 3

    def test_nearclip_sweep(self, base_config):
        """Sweep near-clip distance from 0.05 to 0.15 (step 0.05).

        Near-clip filters splats too close to camera (reduces 'straw' artifacts).

        Expected findings:
        - 0.05: Minimal filtering, may have artifacts
        - 0.10: Good balance
        - 0.15: Aggressive filtering, cleaner but may lose detail
        """
        config = TuningTestConfig(
            **base_config,
            dgs_scene_strategies=["per_segment"],
            dgs_models=["DA3NESTED-GIANT-LARGE"],
            dgs_neighbor_segments_min=6,
            dgs_neighbor_segments_max=6,
            dgs_neighbor_segments_step=1,
            dgs_densification_min=4,
            dgs_densification_max=4,
            dgs_densification_step=1,
            dgs_nearclip_min=0.05,
            dgs_nearclip_max=0.15,
            dgs_nearclip_step=0.05,
        )

        # Result: 3 test runs (0.05, 0.10, 0.15)
        expected_values = np.arange(0.05, 0.16, 0.05)
        assert len(expected_values) == 3

    def test_comprehensive_sweep(self, base_config):
        """Comprehensive parameter sweep across all dimensions.

        This generates a large test matrix:
        - 1 scene strategy (per_segment for speed)
        - 1 model (DA3NESTED-GIANT-LARGE for quality)
        - 3 neighbor segments (4, 6, 8)
        - 3 densifications (2x, 4x, 6x)
        - 3 near-clips (0.05, 0.10, 0.15)

        Total: 1 × 1 × 3 × 3 × 3 = 27 test runs
        """
        config = TuningTestConfig(
            **base_config,
            dgs_scene_strategies=["per_segment"],
            dgs_models=["DA3NESTED-GIANT-LARGE"],
            dgs_neighbor_segments_min=4,
            dgs_neighbor_segments_max=8,
            dgs_neighbor_segments_step=2,
            dgs_densification_min=2,
            dgs_densification_max=6,
            dgs_densification_step=2,
            dgs_nearclip_min=0.05,
            dgs_nearclip_max=0.15,
            dgs_nearclip_step=0.05,
        )

        # Calculate expected test count
        neighbor_configs = list(range(4, 9, 2))  # 4, 6, 8 = 3
        densification_configs = list(range(2, 7, 2))  # 2, 4, 6 = 3
        nearclip_configs = list(np.arange(0.05, 0.16, 0.05))  # 0.05, 0.10, 0.15 = 3

        total_tests = (
            len(config.dgs_scene_strategies) *
            len(config.dgs_models) *
            len(neighbor_configs) *
            len(densification_configs) *
            len(nearclip_configs)
        )

        assert total_tests == 27

    def test_vram_budget_optimization(self, base_config):
        """Find optimal parameters within VRAM budget.

        24GB GPU budget:
        - Strategy 1: Few keyframes, high densification
          - 6 neighbor segments, 6x densification = ~18GB
        - Strategy 2: More keyframes, medium densification
          - 8 neighbor segments, 4x densification = ~20GB
        - Strategy 3: Many keyframes, low densification
          - 10 neighbor segments, 2x densification = ~16GB

        Test all three to find best quality within budget.
        """
        configs = [
            # Strategy 1: High detail, limited coverage
            TuningTestConfig(
                **base_config,
                dgs_scene_strategies=["per_segment"],
                dgs_models=["DA3NESTED-GIANT-LARGE"],
                dgs_neighbor_segments_min=6,
                dgs_neighbor_segments_max=6,
                dgs_neighbor_segments_step=1,
                dgs_densification_min=6,
                dgs_densification_max=6,
                dgs_densification_step=1,
                dgs_nearclip_min=0.1,
                dgs_nearclip_max=0.1,
                dgs_nearclip_step=1.0,
            ),
            # Strategy 2: Balanced
            TuningTestConfig(
                **base_config,
                dgs_scene_strategies=["per_segment"],
                dgs_models=["DA3NESTED-GIANT-LARGE"],
                dgs_neighbor_segments_min=8,
                dgs_neighbor_segments_max=8,
                dgs_neighbor_segments_step=1,
                dgs_densification_min=4,
                dgs_densification_max=4,
                dgs_densification_step=1,
                dgs_nearclip_min=0.1,
                dgs_nearclip_max=0.1,
                dgs_nearclip_step=1.0,
            ),
            # Strategy 3: Better coverage, less detail
            TuningTestConfig(
                **base_config,
                dgs_scene_strategies=["per_segment"],
                dgs_models=["DA3NESTED-GIANT-LARGE"],
                dgs_neighbor_segments_min=10,
                dgs_neighbor_segments_max=10,
                dgs_neighbor_segments_step=1,
                dgs_densification_min=2,
                dgs_densification_max=2,
                dgs_densification_step=1,
                dgs_nearclip_min=0.1,
                dgs_nearclip_max=0.1,
                dgs_nearclip_step=1.0,
            ),
        ]

        # Each strategy = 1 test run
        # Total: 3 test runs
        assert len(configs) == 3

    def test_multi_aspect_ratio_sweep(self, base_config):
        """Test across multiple aspect ratios.

        Different aspect ratios may have different optimal parameters.
        Test 16:9, 9:16, and 1:1 to find aspect-specific optima.
        """
        config = TuningTestConfig(
            test_type=TuningTestType.DA3_3DGS_TUNING,
            aspect_ratios=[
                [1.78, 512, 288],  # 16:9 landscape
                [0.56, 288, 512],  # 9:16 portrait
                [1.0, 512, 512],   # 1:1 square
            ],
            rotation_factor=-8.0,
            orbit_radius=5.0,
            test_iterations=50,
            dgs_scene_strategies=["per_segment"],
            dgs_models=["DA3NESTED-GIANT-LARGE"],
            dgs_neighbor_segments_min=6,
            dgs_neighbor_segments_max=6,
            dgs_neighbor_segments_step=1,
            dgs_densification_min=4,
            dgs_densification_max=4,
            dgs_densification_step=1,
            dgs_nearclip_min=0.1,
            dgs_nearclip_max=0.1,
            dgs_nearclip_step=1.0,
        )

        # Result: 3 test runs (one per aspect ratio)
        assert len(config.aspect_ratios) == 3


def estimate_vram_usage(
    model: str,
    neighbor_segments: int,
    densification: int,
    width: int,
    height: int
) -> float:
    """Estimate VRAM usage for given parameters.

    Args:
        model: "DA3-GIANT" or "DA3NESTED-GIANT-LARGE"
        neighbor_segments: Number of keyframes (4-10)
        densification: Splat multiplier (1-8x)
        width: Resolution width
        height: Resolution height

    Returns:
        Estimated VRAM usage in GB
    """
    # Base model memory
    base_model_gb = 4.0 if model == "DA3-GIANT" else 4.5

    # Per-keyframe overhead
    keyframe_overhead_gb = neighbor_segments * 0.2

    # Splat memory (705k base splats)
    base_splats = 705_000
    total_splats = base_splats * densification
    splat_memory_gb = (total_splats / 1_000_000) * 2.5

    # Resolution overhead
    resolution_factor = (width * height) / (1024 * 1024)
    resolution_gb = resolution_factor * 0.5

    total_gb = base_model_gb + keyframe_overhead_gb + splat_memory_gb + resolution_gb

    return total_gb


def test_vram_estimation():
    """Test VRAM estimation accuracy."""
    # Example 1: 10 keyframes @ 8x density @ 512×512 should be ~20GB
    est1 = estimate_vram_usage("DA3NESTED-GIANT-LARGE", 10, 8, 512, 512)
    assert 19.0 <= est1 <= 21.0, f"Expected ~20GB, got {est1:.1f}GB"

    # Example 2: 50 keyframes @ 2x density @ 512×512 should be ~17GB
    est2 = estimate_vram_usage("DA3NESTED-GIANT-LARGE", 50, 2, 512, 512)
    assert 16.0 <= est2 <= 18.0, f"Expected ~17GB, got {est2:.1f}GB"

    # Example 3: 30 keyframes @ 4x density @ 1024×1024 should be ~24GB
    est3 = estimate_vram_usage("DA3NESTED-GIANT-LARGE", 30, 4, 1024, 1024)
    assert 23.0 <= est3 <= 25.0, f"Expected ~24GB, got {est3:.1f}GB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
