"""
Integration test for DA3 3DGS rendering with near_clip_distance parameter.

This test creates synthetic test frames and validates:
1. Gaussian splat scene construction
2. Keyframe rendering quality
3. Tween frame rendering quality
4. Impact of near_clip_distance on rendering output
"""

import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deforum.rendering.da3_3dgs_novel_view import generate_da3_3dgs_interpolation


def create_synthetic_gradient_image(width: int = 512, height: int = 512, hue_shift: float = 0.0) -> Image.Image:
    """Create synthetic test image with horizontal gradient and simple geometry.

    Args:
        width: Image width
        height: Image height
        hue_shift: Shift hue by this amount (0.0-1.0) for frame variation

    Returns:
        PIL Image with gradient background and simple shapes
    """
    from colorsys import hsv_to_rgb

    # Create numpy array
    img_array = np.zeros((height, width, 3), dtype=np.uint8)

    # Add horizontal gradient background
    for x in range(width):
        hue = ((x / width) + hue_shift) % 1.0
        r, g, b = hsv_to_rgb(hue, 0.8, 0.9)
        img_array[:, x, 0] = int(r * 255)
        img_array[:, x, 1] = int(g * 255)
        img_array[:, x, 2] = int(b * 255)

    # Add simple geometric shapes for depth cues
    # Circle in center
    center_x, center_y = width // 2, height // 2
    radius = min(width, height) // 4
    y_coords, x_coords = np.ogrid[:height, :width]
    mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
    img_array[mask] = [255, 255, 255]  # White circle

    # Add corner squares
    square_size = 50
    img_array[10:10+square_size, 10:10+square_size] = [255, 0, 0]  # Red top-left
    img_array[10:10+square_size, -10-square_size:-10] = [0, 255, 0]  # Green top-right
    img_array[-10-square_size:-10, 10:10+square_size] = [0, 0, 255]  # Blue bottom-left
    img_array[-10-square_size:-10, -10-square_size:-10] = [255, 255, 0]  # Yellow bottom-right

    return Image.fromarray(img_array)


def validate_image_quality(img_path: str, min_mean: float = 10.0) -> dict:
    """Validate rendered image is not black/corrupted.

    Args:
        img_path: Path to image file
        min_mean: Minimum acceptable mean pixel value (0-255)

    Returns:
        Dict with validation metrics: {
            'is_valid': bool,
            'mean': float,
            'std': float,
            'min': int,
            'max': int,
            'file_size': int,
            'reason': str
        }
    """
    result = {
        'is_valid': False,
        'mean': 0.0,
        'std': 0.0,
        'min': 0,
        'max': 0,
        'file_size': 0,
        'reason': 'Not checked'
    }

    # Check file exists
    if not os.path.exists(img_path):
        result['reason'] = 'File does not exist'
        return result

    # Check file size
    file_size = os.path.getsize(img_path)
    result['file_size'] = file_size

    if file_size < 1000:  # Less than 1KB is suspicious
        result['reason'] = f'File too small ({file_size} bytes)'
        return result

    # Load image and compute stats
    try:
        img = Image.open(img_path)
        img_array = np.array(img)

        result['mean'] = float(np.mean(img_array))
        result['std'] = float(np.std(img_array))
        result['min'] = int(np.min(img_array))
        result['max'] = int(np.max(img_array))

        # Validate
        if result['mean'] < min_mean:
            result['reason'] = f"Mean pixel value too low ({result['mean']:.2f} < {min_mean})"
            return result

        if result['std'] < 1.0:
            result['reason'] = f"Standard deviation too low ({result['std']:.2f}) - likely solid color"
            return result

        if result['max'] == result['min']:
            result['reason'] = "All pixels same value - solid color image"
            return result

        # All checks passed
        result['is_valid'] = True
        result['reason'] = 'Valid image'

    except Exception as e:
        result['reason'] = f'Error loading image: {e}'
        return result

    return result


@pytest.mark.integration
@pytest.mark.parametrize("near_clip", [0.0, 0.05, 0.1, 0.2])
def test_da3_3dgs_nearclip_impact(near_clip, tmp_path):
    """Test DA3 3DGS rendering with different near_clip_distance values.

    This test:
    1. Creates 4 synthetic keyframe images with gradual hue shift
    2. Runs 3DGS interpolation with specified near_clip_distance
    3. Validates output frames are not black/corrupted
    4. Reports detailed statistics
    """
    # Skip if CUDA not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    # Create 4 synthetic keyframe images
    keyframe_images = []
    keyframe_indices = [0, 10, 20, 30]  # Frame indices

    print(f"\n{'='*80}")
    print(f"Testing near_clip_distance={near_clip}")
    print(f"{'='*80}")

    for i, idx in enumerate(keyframe_indices):
        hue_shift = i * 0.25  # Shift hue for each frame
        img = create_synthetic_gradient_image(width=512, height=512, hue_shift=hue_shift)
        keyframe_images.append(img)

        # Save keyframe for inspection
        save_path = tmp_path / f"keyframe_{idx:04d}.png"
        img.save(save_path)
        print(f"Created keyframe {idx:04d} (hue_shift={hue_shift:.2f}) -> {save_path}")

    # Target frames to interpolate (between keyframes)
    target_frame_indices = [5, 15, 25]  # Middle points between keyframes

    print(f"\nKeyframe indices: {keyframe_indices}")
    print(f"Target tween indices: {target_frame_indices}")
    print(f"Near clip distance: {near_clip}")

    # Run 3DGS interpolation
    output_dir = str(tmp_path)

    try:
        output_paths = generate_da3_3dgs_interpolation(
            keyframe_images=keyframe_images,
            keyframe_indices=keyframe_indices,
            target_frame_indices=target_frame_indices,
            model_selection="DepthAnythingV2-Small",  # Lightweight for testing
            output_dir=output_dir,
            device=device,
            render_keyframes=True,  # Also render boundary keyframes
            segment_first_idx=keyframe_indices[0],
            segment_last_idx=keyframe_indices[-1],
            densification_factor=1,  # Low for faster testing
            near_clip_distance=near_clip,
            dashboard=None
        )

        print(f"\nGenerated {len(output_paths)} output frames")

        # Validate ALL output frames
        results = {}
        all_valid = True

        for frame_path in output_paths:
            if not os.path.exists(frame_path):
                print(f"  ❌ Missing: {frame_path}")
                all_valid = False
                continue

            validation = validate_image_quality(frame_path, min_mean=10.0)
            results[frame_path] = validation

            status = "✅" if validation['is_valid'] else "❌"
            print(f"  {status} {os.path.basename(frame_path)}")
            print(f"      Mean: {validation['mean']:.2f}, Std: {validation['std']:.2f}, "
                  f"Range: [{validation['min']}, {validation['max']}], "
                  f"Size: {validation['file_size']} bytes")
            if not validation['is_valid']:
                print(f"      Reason: {validation['reason']}")
                all_valid = False

        # Statistics
        valid_count = sum(1 for r in results.values() if r['is_valid'])
        total_count = len(results)

        print(f"\n{'='*80}")
        print(f"RESULTS for near_clip={near_clip}")
        print(f"{'='*80}")
        print(f"Valid frames: {valid_count}/{total_count} ({valid_count/total_count*100:.1f}%)")

        if all_valid:
            print(f"✅ ALL FRAMES VALID")
        else:
            print(f"❌ SOME FRAMES INVALID")

        # Pytest assertion
        assert all_valid, (
            f"Some frames are black/corrupted with near_clip={near_clip}. "
            f"Valid: {valid_count}/{total_count}"
        )

    except Exception as e:
        print(f"\n❌ ERROR during 3DGS generation:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise


@pytest.mark.integration
def test_da3_3dgs_nearclip_comparison(tmp_path):
    """Compare output quality across different near_clip values.

    This test runs the same synthetic scene with multiple near_clip values
    and generates a comparison report.
    """
    # Skip if CUDA not available
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda")

    # Create 4 synthetic keyframe images (same as above)
    keyframe_images = []
    keyframe_indices = [0, 10, 20, 30]

    for i, idx in enumerate(keyframe_indices):
        hue_shift = i * 0.25
        img = create_synthetic_gradient_image(width=512, height=512, hue_shift=hue_shift)
        keyframe_images.append(img)

    target_frame_indices = [5, 15, 25]

    # Test multiple near_clip values
    near_clip_values = [0.0, 0.05, 0.1, 0.15, 0.2]

    comparison_results = {}

    for near_clip in near_clip_values:
        print(f"\n{'='*80}")
        print(f"Testing near_clip={near_clip}")
        print(f"{'='*80}")

        # Create subdirectory for this run
        run_dir = tmp_path / f"nearclip_{near_clip:.2f}"
        run_dir.mkdir(exist_ok=True)

        try:
            output_paths = generate_da3_3dgs_interpolation(
                keyframe_images=keyframe_images,
                keyframe_indices=keyframe_indices,
                target_frame_indices=target_frame_indices,
                model_selection="DepthAnythingV2-Small",
                output_dir=str(run_dir),
                device=device,
                render_keyframes=True,
                segment_first_idx=keyframe_indices[0],
                segment_last_idx=keyframe_indices[-1],
                densification_factor=1,
                near_clip_distance=near_clip,
                dashboard=None
            )

            # Validate frames
            valid_count = 0
            total_mean = 0.0

            for frame_path in output_paths:
                validation = validate_image_quality(frame_path, min_mean=10.0)
                if validation['is_valid']:
                    valid_count += 1
                    total_mean += validation['mean']

            avg_mean = total_mean / len(output_paths) if output_paths else 0.0

            comparison_results[near_clip] = {
                'valid_count': valid_count,
                'total_count': len(output_paths),
                'avg_mean': avg_mean,
                'success': valid_count == len(output_paths)
            }

            print(f"  Valid: {valid_count}/{len(output_paths)}, Avg mean: {avg_mean:.2f}")

        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            comparison_results[near_clip] = {
                'valid_count': 0,
                'total_count': 0,
                'avg_mean': 0.0,
                'success': False,
                'error': str(e)
            }

    # Print comparison report
    print(f"\n{'='*80}")
    print(f"COMPARISON REPORT")
    print(f"{'='*80}")
    print(f"{'near_clip':<12} {'Valid Frames':<15} {'Avg Mean':<12} {'Status':<10}")
    print(f"{'-'*80}")

    for near_clip, result in comparison_results.items():
        valid_str = f"{result['valid_count']}/{result['total_count']}"
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{near_clip:<12.2f} {valid_str:<15} {result['avg_mean']:<12.2f} {status:<10}")

    # Save report to file
    report_path = tmp_path / "nearclip_comparison_report.txt"
    with open(report_path, 'w') as f:
        f.write("DA3 3DGS Near Clip Distance Comparison Report\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'near_clip':<12} {'Valid Frames':<15} {'Avg Mean':<12} {'Status':<10}\n")
        f.write("-"*80 + "\n")
        for near_clip, result in comparison_results.items():
            valid_str = f"{result['valid_count']}/{result['total_count']}"
            status = "PASS" if result['success'] else "FAIL"
            f.write(f"{near_clip:<12.2f} {valid_str:<15} {result['avg_mean']:<12.2f} {status:<10}\n")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    # Allow running standalone for debugging
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        print("Running single near_clip test (0.1)...")
        test_da3_3dgs_nearclip_impact(near_clip=0.1, tmp_path=tmp_path)

        print("\n" + "="*80)
        print("Running comparison test...")
        test_da3_3dgs_nearclip_comparison(tmp_path=tmp_path)
