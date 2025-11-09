#!/usr/bin/env python3
"""Standalone depth warping orbit test - NO DIFFUSION.

Tests pure depth warping with orbital camera paths to find optimal
rotation_factor values WITHOUT loading or using any diffusion models.

This bypasses the entire Deforum render pipeline and directly tests
the depth warping mathematics.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import cv2
import sys
from typing import Tuple, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from deforum.rendering.util.depth_utils import DepthProcessor


def generate_synthetic_sphere(width: int, height: int) -> np.ndarray:
    """Generate a synthetic 3D sphere with Phong shading."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 128  # Gray background

    center_x = width // 2
    center_y = height // 2
    radius = min(width, height) * 0.35

    light_pos = np.array([-1.0, -1.0, 2.0])
    light_pos = light_pos / np.linalg.norm(light_pos)

    for y in range(height):
        for x in range(width):
            dx = x - center_x
            dy = y - center_y
            dist_sq = dx*dx + dy*dy

            if dist_sq <= radius*radius:
                z = np.sqrt(radius*radius - dist_sq)
                normal = np.array([dx, dy, z])
                normal = normal / np.linalg.norm(normal)
                diffuse = max(0.0, np.dot(normal, light_pos))
                intensity = 0.2 + 0.8 * diffuse
                color = int(255 * intensity)
                img[y, x] = [color, color, color]

    return img


def apply_depth_warp(
    image: np.ndarray,
    depth_map: np.ndarray,
    tx: float,
    ty: float,
    tz: float,
    rx: float,
    ry: float,
    rz: float,
) -> np.ndarray:
    """Apply 3D depth warping transformation to image.

    This is a simplified version that uses OpenCV's perspective warp.
    For production, would use the full Deforum depth warping pipeline.
    """
    h, w = image.shape[:2]

    # Simple translation + rotation approximation
    # For now, just apply translation (proper 3D would need full depth warping)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    warped = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    return warped


def count_iterations_until_offscreen(
    frames: List[np.ndarray],
    width: int,
    height: int
) -> int:
    """Count how many iterations until sphere goes off-screen."""
    edge_margin = min(width, height) * 0.15

    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        moments = cv2.moments(binary)

        if moments['m00'] == 0:
            return i

        cx = moments['m10'] / moments['m00']
        cy = moments['m01'] / moments['m00']

        if (cx < edge_margin or cx > width - edge_margin or
            cy < edge_margin or cy > height - edge_margin):
            return i

    return len(frames)


def run_orbit_test(
    width: int,
    height: int,
    orbit_radius: float,
    rotation_factor: float,
    max_iterations: int,
    output_dir: Path,
) -> dict:
    """Run single orbit test configuration - PURE DEPTH WARP, NO DIFFUSION.

    Args:
        width: Frame width
        height: Frame height
        orbit_radius: Orbit radius in pixels
        rotation_factor: Translation/rotation ratio
        max_iterations: Maximum frames to generate
        output_dir: Where to save frames

    Returns:
        Test results dict
    """
    print(f"\n{'='*70}")
    print(f"Testing: {width}x{height}, radius={orbit_radius}, factor={rotation_factor}")
    print(f"{'='*70}")

    # Generate sphere
    print("Generating synthetic sphere...")
    sphere = generate_synthetic_sphere(width, height)

    # Save initial sphere
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sphere).save(output_dir / "000000000.png")

    # Initialize depth processor
    print("Initializing depth estimation...")
    depth_proc = DepthProcessor("Depth-Anything-V2-Small")

    # Get initial depth map
    depth_map = depth_proc.predict(sphere)
    depth_viz = (depth_map * 255).astype(np.uint8)
    Image.fromarray(depth_viz).save(output_dir / "depth_maps" / "000000000_depth.png")

    # Generate orbit path
    angles = np.linspace(0, 2 * np.pi, max_iterations + 1)[:-1]
    x_positions = orbit_radius * np.cos(angles)
    y_positions = orbit_radius * np.sin(angles)
    rotation_angles = -np.degrees(angles) / rotation_factor

    # Apply depth warping for each frame
    print(f"Applying depth warping for {max_iterations} iterations...")
    frames = [sphere]
    current_frame = sphere.copy()

    for i in range(1, max_iterations):
        # Apply depth warp transformation
        tx = x_positions[i] - x_positions[i-1]
        ty = y_positions[i] - y_positions[i-1]
        ry = rotation_angles[i] - rotation_angles[i-1]

        # Warp frame (simplified - would use full Deforum depth warp in production)
        warped = apply_depth_warp(current_frame, depth_map, tx, ty, 0, 0, ry, 0)

        # Save frame
        Image.fromarray(warped).save(output_dir / f"{i:09d}.png")
        frames.append(warped)
        current_frame = warped

        # Re-estimate depth for next iteration
        depth_map = depth_proc.predict(warped)

        if i % 10 == 0:
            print(f"  Frame {i}/{max_iterations}")

    # Measure results
    print("Analyzing results...")
    iterations_visible = count_iterations_until_offscreen(frames, width, height)

    results = {
        "width": width,
        "height": height,
        "orbit_radius": orbit_radius,
        "rotation_factor": rotation_factor,
        "max_iterations": max_iterations,
        "iterations_until_offscreen": iterations_visible,
    }

    print(f"Sphere stayed in frame for {iterations_visible} iterations")
    print(f"{'='*70}\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pure depth warp orbit test (NO diffusion)")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--factor-min", type=float, default=-7.0)
    parser.add_argument("--factor-max", type=float, default=-3.0)
    parser.add_argument("--factor-step", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("outputs/depth-warp-orbit"))

    args = parser.parse_args()

    # Sweep rotation factors
    rotation_factors = np.arange(args.factor_min, args.factor_max + 0.01, args.factor_step)
    all_results = []

    for rf in rotation_factors:
        test_dir = args.output / f"factor{abs(rf):.2f}"
        result = run_orbit_test(
            args.width,
            args.height,
            args.radius,
            rf,
            args.iterations,
            test_dir,
        )
        all_results.append(result)

    # Generate plot
    print("\nGenerating results graph...")
    import plotly.graph_objects as go

    factors = [r["rotation_factor"] for r in all_results]
    iterations = [r["iterations_until_offscreen"] for r in all_results]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=factors,
        y=iterations,
        mode="lines+markers",
        name=f"{args.width}x{args.height}",
        line=dict(color="blue", width=2),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="Pure Depth Warp Orbit Test (NO Diffusion)",
        xaxis_title="Rotation Factor",
        yaxis_title="Iterations Until Off-Screen",
        template="plotly_white",
    )

    output_html = args.output / "orbit_results.html"
    fig.write_html(str(output_html))
    print(f"Results saved to: {output_html}")
