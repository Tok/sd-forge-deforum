"""Tests for frame overlap simulator."""

import pytest
import numpy as np
from deforum.utils.frame_overlap_simulator import (
    Rectangle,
    apply_translation,
    apply_rotation,
    apply_zoom,
    calculate_polygon_area,
    calculate_rectangle_intersection_area,
    calculate_frame_metrics,
    simulate_camera_path,
    analyze_metrics,
    MIN_PRESERVATION_THRESHOLD,
    MAX_NOVELTY_THRESHOLD,
)


class TestRectangle:
    """Test Rectangle dataclass and methods."""

    def test_rectangle_creation(self):
        """Test creating a rectangle."""
        rect = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=50.0)
        assert rect.center_x == 0.0
        assert rect.center_y == 0.0
        assert rect.width == 100.0
        assert rect.height == 50.0
        assert rect.rotation == 0.0

    def test_rectangle_with_rotation(self):
        """Test creating a rotated rectangle."""
        rect = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=50.0, rotation=45.0)
        assert rect.rotation == 45.0

    def test_get_corners_unrotated(self):
        """Test getting corners of an unrotated rectangle."""
        rect = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=50.0)
        corners = rect.get_corners()

        assert corners.shape == (4, 2)
        # Top-left, top-right, bottom-right, bottom-left
        expected = np.array([
            [-50.0, -25.0],  # Top-left
            [50.0, -25.0],   # Top-right
            [50.0, 25.0],    # Bottom-right
            [-50.0, 25.0],   # Bottom-left
        ])
        np.testing.assert_array_almost_equal(corners, expected)

    def test_get_corners_translated(self):
        """Test getting corners of a translated rectangle."""
        rect = Rectangle(center_x=10.0, center_y=20.0, width=100.0, height=50.0)
        corners = rect.get_corners()

        expected = np.array([
            [-40.0, -5.0],   # Top-left
            [60.0, -5.0],    # Top-right
            [60.0, 45.0],    # Bottom-right
            [-40.0, 45.0],   # Bottom-left
        ])
        np.testing.assert_array_almost_equal(corners, expected)

    def test_get_corners_rotated_90deg(self):
        """Test getting corners of a 90-degree rotated rectangle."""
        rect = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=50.0, rotation=90.0)
        corners = rect.get_corners()

        # After 90° rotation, width and height swap
        expected = np.array([
            [25.0, -50.0],   # Top-left rotated
            [25.0, 50.0],    # Top-right rotated
            [-25.0, 50.0],   # Bottom-right rotated
            [-25.0, -50.0],  # Bottom-left rotated
        ])
        np.testing.assert_array_almost_equal(corners, expected, decimal=1)


class TestTransforms:
    """Test transform operations."""

    def test_apply_translation(self):
        """Test applying translation to a rectangle."""
        rect = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=50.0)
        translated = apply_translation(rect, translation_x=10.0, translation_y=20.0)

        assert translated.center_x == 10.0
        assert translated.center_y == 20.0
        assert translated.width == 100.0
        assert translated.height == 50.0
        assert translated.rotation == 0.0

    def test_apply_rotation(self):
        """Test applying rotation to a rectangle."""
        rect = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=50.0)
        rotated = apply_rotation(rect, rotation_degrees=45.0)

        assert rotated.center_x == 0.0
        assert rotated.center_y == 0.0
        assert rotated.width == 100.0
        assert rotated.height == 50.0
        assert rotated.rotation == 45.0

    def test_apply_rotation_accumulation(self):
        """Test that rotation accumulates."""
        rect = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=50.0, rotation=30.0)
        rotated = apply_rotation(rect, rotation_degrees=15.0)

        assert rotated.rotation == 45.0

    def test_apply_zoom_in(self):
        """Test zooming in (previous frame appears larger)."""
        rect = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=50.0)
        zoomed = apply_zoom(rect, zoom_factor=2.0)

        # Zoom in = previous frame appears larger
        assert zoomed.center_x == 0.0
        assert zoomed.center_y == 0.0
        assert zoomed.width == 200.0  # 100 * 2
        assert zoomed.height == 100.0  # 50 * 2

    def test_apply_zoom_out(self):
        """Test zooming out (previous frame appears smaller)."""
        rect = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=50.0)
        zoomed = apply_zoom(rect, zoom_factor=0.5)

        # Zoom out = previous frame appears smaller
        assert zoomed.center_x == 0.0
        assert zoomed.center_y == 0.0
        assert zoomed.width == 50.0  # 100 * 0.5
        assert zoomed.height == 25.0  # 50 * 0.5


class TestAreaCalculations:
    """Test area calculation functions."""

    def test_calculate_polygon_area_square(self):
        """Test calculating area of a square."""
        corners = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ])
        area = calculate_polygon_area(corners)
        assert area == 100.0

    def test_calculate_polygon_area_rectangle(self):
        """Test calculating area of a rectangle."""
        corners = np.array([
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 10.0],
            [0.0, 10.0],
        ])
        area = calculate_polygon_area(corners)
        assert area == 200.0

    def test_calculate_polygon_area_rotated(self):
        """Test that rotation doesn't affect area calculation."""
        # 10x10 square rotated 45 degrees
        half_diag = 10.0 / np.sqrt(2)
        corners = np.array([
            [0.0, -half_diag],
            [half_diag, 0.0],
            [0.0, half_diag],
            [-half_diag, 0.0],
        ])
        area = calculate_polygon_area(corners)
        np.testing.assert_almost_equal(area, 100.0, decimal=1)


class TestIntersection:
    """Test rectangle intersection calculations."""

    def test_aabb_intersection_full_overlap(self):
        """Test AABB intersection with full overlap."""
        rect1 = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=100.0)
        rect2 = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=100.0)

        area = calculate_rectangle_intersection_area(rect1, rect2)
        assert area == 10000.0  # 100 * 100

    def test_aabb_intersection_partial_overlap(self):
        """Test AABB intersection with partial overlap."""
        rect1 = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=100.0)
        rect2 = Rectangle(center_x=50.0, center_y=0.0, width=100.0, height=100.0)

        area = calculate_rectangle_intersection_area(rect1, rect2)
        assert area == 5000.0  # 50 * 100 overlap

    def test_aabb_intersection_no_overlap(self):
        """Test AABB intersection with no overlap."""
        rect1 = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=100.0)
        rect2 = Rectangle(center_x=200.0, center_y=0.0, width=100.0, height=100.0)

        area = calculate_rectangle_intersection_area(rect1, rect2)
        assert area == 0.0

    def test_aabb_intersection_touching_edges(self):
        """Test AABB intersection with touching edges (no overlap)."""
        rect1 = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=100.0)
        rect2 = Rectangle(center_x=100.0, center_y=0.0, width=100.0, height=100.0)

        area = calculate_rectangle_intersection_area(rect1, rect2)
        assert area == 0.0

    def test_rotated_intersection_45deg(self):
        """Test intersection with 45-degree rotated rectangle."""
        rect1 = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=100.0)
        rect2 = Rectangle(center_x=0.0, center_y=0.0, width=100.0, height=100.0, rotation=45.0)

        area = calculate_rectangle_intersection_area(rect1, rect2)

        # 45-degree rotated square creates octagon intersection
        # Area should be approximately 82.84% of original area
        assert 0.0 < area < 10000.0
        np.testing.assert_almost_equal(area, 8284.0, decimal=0)


class TestFrameMetrics:
    """Test frame metrics calculation."""

    def test_no_movement_full_preservation(self):
        """Test that no movement results in 100% preservation."""
        prev_rect = Rectangle(center_x=0.0, center_y=0.0, width=1920.0, height=1080.0)

        metrics = calculate_frame_metrics(
            prev_frame_rect=prev_rect,
            translation_x=0.0,
            translation_y=0.0,
            rotation_3d_y=0.0,
            zoom=1.0,
            viewport_width=1920.0,
            viewport_height=1080.0,
            frame_index=1
        )

        assert metrics.preservation == 1.0
        assert metrics.novelty == 0.0
        assert metrics.frame_index == 1

    def test_zoom_out_creates_novelty(self):
        """Test that zoom out creates new space (novelty)."""
        prev_rect = Rectangle(center_x=0.0, center_y=0.0, width=1920.0, height=1080.0)

        metrics = calculate_frame_metrics(
            prev_frame_rect=prev_rect,
            translation_x=0.0,
            translation_y=0.0,
            rotation_3d_y=0.0,
            zoom=0.8,  # Zoom out to 80%
            viewport_width=1920.0,
            viewport_height=1080.0,
            frame_index=1
        )

        # After zoom out, previous frame covers 80% of viewport
        # So we have 20% novelty
        np.testing.assert_almost_equal(metrics.preservation, 0.64, decimal=2)  # 0.8 * 0.8
        np.testing.assert_almost_equal(metrics.novelty, 0.36, decimal=2)

    def test_zoom_in_reduces_preservation(self):
        """Test that zoom in causes previous frame to be cropped (still 100% coverage)."""
        prev_rect = Rectangle(center_x=0.0, center_y=0.0, width=1920.0, height=1080.0)

        metrics = calculate_frame_metrics(
            prev_frame_rect=prev_rect,
            translation_x=0.0,
            translation_y=0.0,
            rotation_3d_y=0.0,
            zoom=1.2,  # Zoom in to 120%
            viewport_width=1920.0,
            viewport_height=1080.0,
            frame_index=1
        )

        # After zoom in, previous frame appears larger and covers entire viewport
        # Preservation stays at 100% (no black borders)
        assert metrics.preservation == 1.0
        assert metrics.novelty == 0.0

    def test_translation_reduces_preservation(self):
        """Test that translation reduces preservation."""
        prev_rect = Rectangle(center_x=0.0, center_y=0.0, width=1920.0, height=1080.0)

        # Translate 25% of width to the right
        metrics = calculate_frame_metrics(
            prev_frame_rect=prev_rect,
            translation_x=480.0,  # 1920 * 0.25
            translation_y=0.0,
            rotation_3d_y=0.0,
            zoom=1.0,
            viewport_width=1920.0,
            viewport_height=1080.0,
            frame_index=1
        )

        # 25% translation should give ~75% preservation
        np.testing.assert_almost_equal(metrics.preservation, 0.75, decimal=2)
        np.testing.assert_almost_equal(metrics.novelty, 0.25, decimal=2)


class TestCameraPathSimulation:
    """Test full camera path simulation."""

    def test_simulate_static_camera(self):
        """Test simulating a static camera (no movement)."""
        num_frames = 10
        translation_x = [0.0] * num_frames
        translation_y = [0.0] * num_frames
        rotation_y = [0.0] * num_frames
        zoom = [1.0] * num_frames

        metrics = simulate_camera_path(
            translation_x_schedule=translation_x,
            translation_y_schedule=translation_y,
            rotation_3d_y_schedule=rotation_y,
            zoom_schedule=zoom,
            viewport_width=1920.0,
            viewport_height=1080.0
        )

        assert len(metrics) == num_frames
        for m in metrics:
            assert m.preservation == 1.0
            assert m.novelty == 0.0

    def test_simulate_zoom_out_sequence(self):
        """Test simulating a zoom out sequence."""
        num_frames = 10
        translation_x = [0.0] * num_frames
        translation_y = [0.0] * num_frames
        rotation_y = [0.0] * num_frames
        zoom = [0.98] * num_frames  # 2% zoom out per frame

        metrics = simulate_camera_path(
            translation_x_schedule=translation_x,
            translation_y_schedule=translation_y,
            rotation_3d_y_schedule=rotation_y,
            zoom_schedule=zoom,
            viewport_width=1920.0,
            viewport_height=1080.0
        )

        assert len(metrics) == num_frames

        # Cumulative zoom out means preservation decreases over time
        # Frame 0: 0.98^2 ≈ 0.96, Frame 9: 0.98^20 ≈ 0.67
        # Each frame should have decreasing preservation
        for i in range(1, len(metrics)):
            assert metrics[i].preservation < metrics[i - 1].preservation

        # First frame should be around 96%, last around 67%
        assert 0.95 < metrics[0].preservation < 0.97
        assert 0.66 < metrics[-1].preservation < 0.68

    def test_simulate_pan_sequence(self):
        """Test simulating a horizontal pan."""
        num_frames = 10
        translation_x = [50.0] * num_frames  # Pan right 50px per frame
        translation_y = [0.0] * num_frames
        rotation_y = [0.0] * num_frames
        zoom = [1.0] * num_frames

        metrics = simulate_camera_path(
            translation_x_schedule=translation_x,
            translation_y_schedule=translation_y,
            rotation_3d_y_schedule=rotation_y,
            zoom_schedule=zoom,
            viewport_width=1920.0,
            viewport_height=1080.0
        )

        assert len(metrics) == num_frames

        # Cumulative panning means preservation decreases over time
        # Frame 0: 50px shift = 97.4%, Frame 9: 500px shift = 73.9%
        # Each frame should have decreasing preservation
        for i in range(1, len(metrics)):
            assert metrics[i].preservation < metrics[i - 1].preservation

        # First frame around 97%, last around 74%
        assert 0.97 < metrics[0].preservation < 0.98
        assert 0.73 < metrics[-1].preservation < 0.75


class TestMetricsAnalysis:
    """Test metrics analysis functions."""

    def test_analyze_metrics_empty(self):
        """Test analyzing empty metrics list."""
        analysis = analyze_metrics([])

        assert analysis['avg_preservation'] == 0.0
        assert analysis['min_preservation'] == 0.0
        assert analysis['avg_novelty'] == 0.0
        assert analysis['max_novelty'] == 0.0

    def test_analyze_metrics_perfect_preservation(self):
        """Test analyzing metrics with perfect preservation."""
        # Create mock metrics
        metrics = [
            calculate_frame_metrics(
                prev_frame_rect=Rectangle(0, 0, 1920, 1080),
                translation_x=0.0,
                translation_y=0.0,
                rotation_3d_y=0.0,
                zoom=1.0,
                viewport_width=1920.0,
                viewport_height=1080.0,
                frame_index=i
            )
            for i in range(10)
        ]

        analysis = analyze_metrics(metrics)

        assert analysis['avg_preservation'] == 1.0
        assert analysis['min_preservation'] == 1.0
        assert analysis['avg_novelty'] == 0.0
        assert analysis['max_novelty'] == 0.0
        assert analysis['preservation_below_threshold'] == 0.0
        assert analysis['novelty_above_threshold'] == 0.0

    def test_analyze_metrics_with_warnings(self):
        """Test analyzing metrics that trigger warnings."""
        # Create metrics with low preservation (high movement)
        # Need >70% translation to get <30% preservation
        metrics = [
            calculate_frame_metrics(
                prev_frame_rect=Rectangle(0, 0, 1920, 1080),
                translation_x=1400.0,  # 73% of width = 27% preservation
                translation_y=0.0,
                rotation_3d_y=0.0,
                zoom=1.0,
                viewport_width=1920.0,
                viewport_height=1080.0,
                frame_index=i
            )
            for i in range(10)
        ]

        analysis = analyze_metrics(metrics)

        # Should trigger warnings (preservation < 30%, novelty > 70%)
        assert analysis['min_preservation'] < MIN_PRESERVATION_THRESHOLD
        assert analysis['max_novelty'] > MAX_NOVELTY_THRESHOLD
        assert analysis['preservation_below_threshold'] > 0.0
        assert analysis['novelty_above_threshold'] > 0.0
