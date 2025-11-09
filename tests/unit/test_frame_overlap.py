"""Unit tests for frame overlap simulator.

Tests preservation/novelty calculations for various camera movements to validate
the frame overlap simulator and help tune proper camera settings.
"""

import pytest
import numpy as np
from deforum.utils.frame_overlap_simulator import (
    Rectangle,
    apply_translation,
    apply_rotation,
    apply_zoom,
    calculate_rectangle_intersection_area,
    calculate_frame_metrics,
    simulate_camera_path,
)


class TestRectangleGeometry:
    """Test basic rectangle geometry operations."""

    def test_rectangle_no_rotation(self):
        """Test that unrotated rectangle has correct corners."""
        rect = Rectangle(center_x=0, center_y=0, width=100, height=60, rotation=0)
        corners = rect.get_corners()

        assert corners.shape == (4, 2)
        # Top-left, top-right, bottom-right, bottom-left
        np.testing.assert_array_almost_equal(corners[0], [-50, -30])
        np.testing.assert_array_almost_equal(corners[1], [50, -30])
        np.testing.assert_array_almost_equal(corners[2], [50, 30])
        np.testing.assert_array_almost_equal(corners[3], [-50, 30])

    def test_rectangle_90_degree_rotation(self):
        """Test 90-degree rotation (width becomes height)."""
        rect = Rectangle(center_x=0, center_y=0, width=100, height=60, rotation=90)
        corners = rect.get_corners()

        # After 90° rotation: [-50,-30] becomes [30,-50] (swap + negate)
        np.testing.assert_array_almost_equal(corners[0], [30, -50])
        np.testing.assert_array_almost_equal(corners[1], [30, 50])
        np.testing.assert_array_almost_equal(corners[2], [-30, 50])
        np.testing.assert_array_almost_equal(corners[3], [-30, -50])


class TestTransformations:
    """Test transformation operations."""

    def test_apply_translation(self):
        """Test translation moves center without changing size/rotation."""
        rect = Rectangle(center_x=0, center_y=0, width=100, height=60, rotation=0)
        translated = apply_translation(rect, translation_x=50, translation_y=30)

        assert translated.center_x == 50
        assert translated.center_y == 30
        assert translated.width == 100
        assert translated.height == 60
        assert translated.rotation == 0

    def test_apply_rotation_accumulates(self):
        """Test that rotation accumulates (adds to existing rotation)."""
        rect = Rectangle(center_x=0, center_y=0, width=100, height=60, rotation=45)
        rotated = apply_rotation(rect, rotation_degrees=15)

        assert rotated.rotation == 60  # 45 + 15

    def test_apply_zoom_in(self):
        """Test zoom in makes rectangle larger."""
        rect = Rectangle(center_x=0, center_y=0, width=100, height=60, rotation=0)
        zoomed = apply_zoom(rect, zoom_factor=1.5)

        assert zoomed.width == 150  # 100 * 1.5
        assert zoomed.height == 90  # 60 * 1.5
        assert zoomed.center_x == 0
        assert zoomed.center_y == 0

    def test_apply_zoom_out(self):
        """Test zoom out makes rectangle smaller."""
        rect = Rectangle(center_x=0, center_y=0, width=100, height=60, rotation=0)
        zoomed = apply_zoom(rect, zoom_factor=0.5)

        assert zoomed.width == 50
        assert zoomed.height == 30


class TestIntersectionCalculations:
    """Test rectangle intersection area calculations."""

    def test_perfect_overlap(self):
        """Test identical rectangles have 100% overlap."""
        rect1 = Rectangle(center_x=0, center_y=0, width=100, height=60, rotation=0)
        rect2 = Rectangle(center_x=0, center_y=0, width=100, height=60, rotation=0)

        area = calculate_rectangle_intersection_area(rect1, rect2)
        expected = 100 * 60

        assert abs(area - expected) < 1  # Allow 1px tolerance

    def test_no_overlap(self):
        """Test separated rectangles have 0% overlap."""
        rect1 = Rectangle(center_x=0, center_y=0, width=100, height=60, rotation=0)
        rect2 = Rectangle(center_x=200, center_y=0, width=100, height=60, rotation=0)

        area = calculate_rectangle_intersection_area(rect1, rect2)

        assert area == 0

    def test_50_percent_horizontal_overlap(self):
        """Test 50% horizontal overlap."""
        rect1 = Rectangle(center_x=0, center_y=0, width=100, height=60, rotation=0)
        rect2 = Rectangle(center_x=50, center_y=0, width=100, height=60, rotation=0)

        area = calculate_rectangle_intersection_area(rect1, rect2)
        expected = 50 * 60  # Half overlap

        assert abs(area - expected) < 1


class TestFrameMetrics:
    """Test frame-to-frame preservation/novelty calculations."""

    def test_no_movement_perfect_preservation(self):
        """Test that no movement = 100% preservation, 0% novelty."""
        prev_rect = Rectangle(center_x=0, center_y=0, width=1920, height=1080, rotation=0)
        metrics = calculate_frame_metrics(
            prev_frame_rect=prev_rect,
            translation_x=0,
            translation_y=0,
            rotation_3d_y=0,
            zoom=1.0,
            viewport_width=1920,
            viewport_height=1080,
            frame_index=1
        )

        assert abs(metrics.preservation - 1.0) < 0.01  # 100% preservation
        assert abs(metrics.novelty - 0.0) < 0.01  # 0% novelty

    def test_small_translation_high_preservation(self):
        """Test that small movement still has high preservation."""
        prev_rect = Rectangle(center_x=0, center_y=0, width=1920, height=1080, rotation=0)
        metrics = calculate_frame_metrics(
            prev_frame_rect=prev_rect,
            translation_x=10,  # Small 10px shift
            translation_y=0,
            rotation_3d_y=0,
            zoom=1.0,
            viewport_width=1920,
            viewport_height=1080,
            frame_index=1
        )

        # With 10px shift on 1920px width, should have ~99.5% preservation
        assert metrics.preservation > 0.99
        assert metrics.novelty < 0.01

    def test_large_translation_low_preservation(self):
        """Test that large movement has low preservation."""
        prev_rect = Rectangle(center_x=0, center_y=0, width=1920, height=1080, rotation=0)
        metrics = calculate_frame_metrics(
            prev_frame_rect=prev_rect,
            translation_x=960,  # Half viewport width
            translation_y=0,
            rotation_3d_y=0,
            zoom=1.0,
            viewport_width=1920,
            viewport_height=1080,
            frame_index=1
        )

        # With 50% viewport shift, should have ~50% preservation
        assert 0.45 < metrics.preservation < 0.55
        assert 0.45 < metrics.novelty < 0.55

    def test_preservation_plus_novelty_equals_one(self):
        """Test that preservation + novelty always equals 1.0."""
        prev_rect = Rectangle(center_x=0, center_y=0, width=1920, height=1080, rotation=0)

        for tx in [0, 10, 50, 100, 500]:
            metrics = calculate_frame_metrics(
                prev_frame_rect=prev_rect,
                translation_x=tx,
                translation_y=0,
                rotation_3d_y=0,
                zoom=1.0,
                viewport_width=1920,
                viewport_height=1080,
                frame_index=1
            )

            total = metrics.preservation + metrics.novelty
            assert abs(total - 1.0) < 0.01  # Should sum to 1.0


class TestCameraPathSimulation:
    """Test full camera path simulation."""

    def test_static_camera_perfect_preservation(self):
        """Test that static camera maintains 100% preservation."""
        num_frames = 10
        tx_schedule = [0.0] * num_frames
        ty_schedule = [0.0] * num_frames
        ry_schedule = [0.0] * num_frames
        zoom_schedule = [1.0] * num_frames

        metrics_list = simulate_camera_path(
            translation_x_schedule=tx_schedule,
            translation_y_schedule=ty_schedule,
            rotation_3d_y_schedule=ry_schedule,
            zoom_schedule=zoom_schedule,
            viewport_width=1920,
            viewport_height=1080
        )

        assert len(metrics_list) == num_frames

        for metrics in metrics_list:
            assert abs(metrics.preservation - 1.0) < 0.01
            assert abs(metrics.novelty - 0.0) < 0.01

    def test_constant_translation_accumulation(self):
        """Test that constant translation deltas accumulate correctly."""
        num_frames = 5
        tx_per_frame = 10.0
        tx_schedule = [tx_per_frame] * num_frames
        ty_schedule = [0.0] * num_frames
        ry_schedule = [0.0] * num_frames
        zoom_schedule = [1.0] * num_frames

        metrics_list = simulate_camera_path(
            translation_x_schedule=tx_schedule,
            translation_y_schedule=ty_schedule,
            rotation_3d_y_schedule=ry_schedule,
            zoom_schedule=zoom_schedule,
            viewport_width=1920,
            viewport_height=1080
        )

        # Check that center_x accumulates: 10, 20, 30, 40, 50
        # (Frame 0 already has first delta applied)
        for i, metrics in enumerate(metrics_list):
            expected_x = tx_per_frame * (i + 1)  # Frame i has (i+1) deltas applied
            assert abs(metrics.prev_frame_rect.center_x - expected_x) < 0.1

    def test_rotation_accumulation(self):
        """Test that rotation deltas accumulate correctly."""
        num_frames = 5
        rotation_per_frame = 5.0  # 5 degrees per frame
        tx_schedule = [0.0] * num_frames
        ty_schedule = [0.0] * num_frames
        ry_schedule = [rotation_per_frame] * num_frames
        zoom_schedule = [1.0] * num_frames

        metrics_list = simulate_camera_path(
            translation_x_schedule=tx_schedule,
            translation_y_schedule=ty_schedule,
            rotation_3d_y_schedule=ry_schedule,
            zoom_schedule=zoom_schedule,
            viewport_width=1920,
            viewport_height=1080
        )

        # Check that rotation accumulates: 5, 10, 15, 20, 25
        # (Frame 0 already has first delta applied)
        for i, metrics in enumerate(metrics_list):
            expected_rotation = rotation_per_frame * (i + 1)  # Frame i has (i+1) deltas applied
            assert abs(metrics.prev_frame_rect.rotation - expected_rotation) < 0.1


class TestNoveltynMetricsForTuning:
    """Tests to help tune camera movement for optimal preservation/novelty."""

    @pytest.mark.parametrize("translation_x,expected_preservation_range", [
        (0, (0.99, 1.0)),      # No movement: 100% preservation
        (10, (0.99, 1.0)),     # Tiny movement: ~99.5% preservation
        (50, (0.96, 0.99)),    # Small movement: ~97% preservation
        (100, (0.93, 0.97)),   # Medium movement: ~95% preservation
        (500, (0.70, 0.80)),   # Large movement: ~75% preservation
        (960, (0.45, 0.55)),   # Half viewport: ~50% preservation
    ])
    def test_preservation_by_translation(self, translation_x, expected_preservation_range):
        """Test preservation metrics for various translation amounts (1920x1080 viewport)."""
        prev_rect = Rectangle(center_x=0, center_y=0, width=1920, height=1080, rotation=0)
        metrics = calculate_frame_metrics(
            prev_frame_rect=prev_rect,
            translation_x=translation_x,
            translation_y=0,
            rotation_3d_y=0,
            zoom=1.0,
            viewport_width=1920,
            viewport_height=1080,
            frame_index=1
        )

        min_pres, max_pres = expected_preservation_range
        assert min_pres <= metrics.preservation <= max_pres, (
            f"Translation {translation_x}px: preservation {metrics.preservation:.3f} "
            f"not in expected range [{min_pres}, {max_pres}]"
        )

    @pytest.mark.parametrize("rotation_degrees,min_preservation", [
        (0, 0.99),    # No rotation: 100% preservation
        (1, 0.95),    # 1 degree: >95% preservation
        (5, 0.80),    # 5 degrees: >80% preservation
        (10, 0.60),   # 10 degrees: >60% preservation
        (30, 0.20),   # 30 degrees: >20% preservation
    ])
    def test_preservation_by_rotation(self, rotation_degrees, min_preservation):
        """Test preservation metrics for various rotation amounts."""
        prev_rect = Rectangle(center_x=0, center_y=0, width=1920, height=1080, rotation=0)
        metrics = calculate_frame_metrics(
            prev_frame_rect=prev_rect,
            translation_x=0,
            translation_y=0,
            rotation_3d_y=rotation_degrees,
            zoom=1.0,
            viewport_width=1920,
            viewport_height=1080,
            frame_index=1
        )

        assert metrics.preservation >= min_preservation, (
            f"Rotation {rotation_degrees}°: preservation {metrics.preservation:.3f} "
            f"below minimum {min_preservation}"
        )


class TestTranslationRotationRatios:
    """Test optimal translation/rotation ratios for depth warping.

    For best depth warping results, rotation should counter-act translation
    to maintain high preservation. Optimal ratio is around -5.0:
    - Moving left (+tx) → Rotate right (-ry) to look inward
    - Moving right (-tx) → Rotate left (+ry) to look inward

    Rule: rotation_delta_y ≈ -translation_delta_x / 5.0
    """

    def test_counter_rotation_maintains_preservation(self):
        """Test that counter-rotation improves preservation vs no rotation."""
        num_frames = 10
        translation_per_frame = 50.0  # 50px per frame

        # Without counter-rotation (camera drifts out of view)
        tx_schedule_no_rotation = [translation_per_frame] * num_frames
        ty_schedule = [0.0] * num_frames
        ry_schedule_no_rotation = [0.0] * num_frames  # No rotation
        zoom_schedule = [1.0] * num_frames

        metrics_no_rotation = simulate_camera_path(
            translation_x_schedule=tx_schedule_no_rotation,
            translation_y_schedule=ty_schedule,
            rotation_3d_y_schedule=ry_schedule_no_rotation,
            zoom_schedule=zoom_schedule,
            viewport_width=1920,
            viewport_height=1080
        )

        # With optimal counter-rotation (camera looks inward)
        # Rule: rotation_y = -translation_x / 5.0
        rotation_per_frame = -translation_per_frame / 5.0  # -10 degrees
        ry_schedule_with_rotation = [rotation_per_frame] * num_frames

        metrics_with_rotation = simulate_camera_path(
            translation_x_schedule=tx_schedule_no_rotation,
            translation_y_schedule=ty_schedule,
            rotation_3d_y_schedule=ry_schedule_with_rotation,
            zoom_schedule=zoom_schedule,
            viewport_width=1920,
            viewport_height=1080
        )

        # Counter-rotation should maintain MUCH better preservation
        avg_preservation_no_rotation = sum(m.preservation for m in metrics_no_rotation) / len(metrics_no_rotation)
        avg_preservation_with_rotation = sum(m.preservation for m in metrics_with_rotation) / len(metrics_with_rotation)

        assert avg_preservation_with_rotation > avg_preservation_no_rotation, (
            f"Counter-rotation should improve preservation: "
            f"{avg_preservation_with_rotation:.3f} > {avg_preservation_no_rotation:.3f}"
        )

    @pytest.mark.parametrize("ratio", [3.0, 5.0, 7.0, 10.0])
    def test_translation_rotation_ratios(self, ratio):
        """Test various translation/rotation ratios for preservation.

        Ideal ratio is around 5.0 (5 pixels per degree).
        """
        num_frames = 20
        translation_per_frame = 50.0
        rotation_per_frame = -translation_per_frame / ratio

        tx_schedule = [translation_per_frame] * num_frames
        ty_schedule = [0.0] * num_frames
        ry_schedule = [rotation_per_frame] * num_frames
        zoom_schedule = [1.0] * num_frames

        metrics = simulate_camera_path(
            translation_x_schedule=tx_schedule,
            translation_y_schedule=ty_schedule,
            rotation_3d_y_schedule=ry_schedule,
            zoom_schedule=zoom_schedule,
            viewport_width=1920,
            viewport_height=1080
        )

        avg_preservation = sum(m.preservation for m in metrics) / len(metrics)

        # All ratios should maintain reasonable preservation (>70%)
        # but ratio=5.0 should be optimal
        assert avg_preservation > 0.70, (
            f"Ratio {ratio}: preservation {avg_preservation:.3f} too low (<70%)"
        )

        # Print for analysis
        print(f"Ratio {ratio:.1f} (tx={translation_per_frame}, ry={rotation_per_frame:.1f}°): "
              f"Preservation={avg_preservation:.3f}")

    def test_optimal_ratio_is_around_5(self):
        """Test that ratio=5.0 provides near-optimal preservation."""
        num_frames = 20
        translation_per_frame = 50.0

        results = {}
        for ratio in [3.0, 4.0, 5.0, 6.0, 7.0]:
            rotation_per_frame = -translation_per_frame / ratio

            metrics = simulate_camera_path(
                translation_x_schedule=[translation_per_frame] * num_frames,
                translation_y_schedule=[0.0] * num_frames,
                rotation_3d_y_schedule=[rotation_per_frame] * num_frames,
                zoom_schedule=[1.0] * num_frames,
                viewport_width=1920,
                viewport_height=1080
            )

            avg_preservation = sum(m.preservation for m in metrics) / len(metrics)
            results[ratio] = avg_preservation

        # Ratio 5.0 should be among the best (top 2)
        sorted_by_preservation = sorted(results.items(), key=lambda x: x[1], reverse=True)
        best_ratios = [r for r, _ in sorted_by_preservation[:2]]

        assert 5.0 in best_ratios, (
            f"Ratio 5.0 should be among top 2, but results are: {sorted_by_preservation}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to show print statements
