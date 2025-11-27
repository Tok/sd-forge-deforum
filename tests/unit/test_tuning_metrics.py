"""Unit tests for tuning quality metrics.

These are fast unit tests that don't require GPU or API.
They verify the metric calculations work correctly.
"""

import numpy as np
import pytest
from ..integration.metrics import (
    measure_color_preservation,
    measure_temporal_consistency,
    measure_perceptual_hash_distance,
    calculate_comprehensive_quality_score,
)


def create_test_image(color_saturation: float = 1.0) -> np.ndarray:
    """Create a test image with specified color saturation.

    Args:
        color_saturation: 0.0 = grayscale, 1.0 = full color

    Returns:
        RGB numpy array (256, 256, 3)
    """
    import cv2

    # Create HSV image
    hsv = np.zeros((256, 256, 3), dtype=np.uint8)

    # Rainbow gradient with specified saturation
    for y in range(256):
        hue = int((y / 256) * 180)  # 0-180 for OpenCV
        saturation = int(255 * color_saturation)
        value = 255

        hsv[y, :] = [hue, saturation, value]

    # Convert to RGB
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def test_color_preservation_full_color():
    """Test that fully colorful image scores close to 100."""
    colorful = create_test_image(color_saturation=1.0)
    score = measure_color_preservation(colorful)

    assert 90 <= score <= 100, f"Colorful image should score 90-100, got {score:.1f}"


def test_color_preservation_grayscale():
    """Test that grayscale image scores close to 0."""
    grayscale = create_test_image(color_saturation=0.0)
    score = measure_color_preservation(grayscale)

    assert 0 <= score <= 10, f"Grayscale image should score 0-10, got {score:.1f}"


def test_color_preservation_partial():
    """Test that partially desaturated image scores in between."""
    partial = create_test_image(color_saturation=0.5)
    score = measure_color_preservation(partial)

    assert 40 <= score <= 60, f"50% saturated image should score 40-60, got {score:.1f}"


def test_temporal_consistency_identical():
    """Test that identical frames score 100."""
    frame = create_test_image()
    score = measure_temporal_consistency(frame, frame)

    assert score == 100, f"Identical frames should score 100, got {score:.1f}"


def test_temporal_consistency_different():
    """Test that different frames score lower."""
    frame1 = create_test_image(color_saturation=1.0)
    frame2 = create_test_image(color_saturation=0.5)

    score = measure_temporal_consistency(frame1, frame2)

    # They're still similar structure (same gradient pattern)
    # but different saturation, so score should be high but not 100
    assert 70 <= score < 100, f"Similar frames should score 70-99, got {score:.1f}"


def test_temporal_consistency_random():
    """Test that random noise frames score low."""
    frame1 = create_test_image()
    frame2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    score = measure_temporal_consistency(frame1, frame2)

    # Random noise should have very low structural similarity
    assert score < 50, f"Random frames should score <50, got {score:.1f}"


def test_perceptual_hash_identical():
    """Test that identical frames have 0 distance."""
    frame = create_test_image()
    distance = measure_perceptual_hash_distance(frame, frame)

    assert distance == 0, f"Identical frames should have 0 distance, got {distance:.1f}"


def test_perceptual_hash_different():
    """Test that different frames have non-zero distance."""
    frame1 = create_test_image(color_saturation=1.0)
    frame2 = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    distance = measure_perceptual_hash_distance(frame1, frame2)

    assert distance > 0, f"Different frames should have >0 distance, got {distance:.1f}"


def test_comprehensive_quality_score():
    """Test comprehensive quality calculation."""
    # Create sequence with degrading color
    frames = [
        create_test_image(color_saturation=1.0),
        create_test_image(color_saturation=0.8),
        create_test_image(color_saturation=0.6),
        create_test_image(color_saturation=0.4),
        create_test_image(color_saturation=0.2),
    ]

    metrics = calculate_comprehensive_quality_score(frames)

    # Verify structure
    assert 'color_scores' in metrics
    assert 'temporal_scores' in metrics
    assert 'overall_score' in metrics
    assert 'degradation_rate' in metrics

    # Verify color scores are decreasing
    assert len(metrics['color_scores']) == 5
    assert metrics['color_scores'][0] > metrics['color_scores'][-1], "Color should degrade"

    # Verify temporal scores exist (4 transitions between 5 frames)
    assert len(metrics['temporal_scores']) == 4

    # Degradation rate should be positive (color is decreasing)
    assert metrics['degradation_rate'] > 0, "Should detect color degradation"


def test_metric_input_validation():
    """Test that metrics handle edge cases properly."""
    frame = create_test_image()

    # Test with different shapes should raise error
    frame_wrong_size = np.zeros((128, 128, 3), dtype=np.uint8)

    with pytest.raises(ValueError):
        measure_temporal_consistency(frame, frame_wrong_size)

    # Test with empty sequence
    metrics = calculate_comprehensive_quality_score([])
    assert metrics['num_frames'] == 0
    assert metrics['overall_score'] == 0

    # Test with single frame
    metrics = calculate_comprehensive_quality_score([frame])
    assert metrics['num_frames'] == 1
    assert len(metrics['temporal_scores']) == 0  # No pairs for consistency
