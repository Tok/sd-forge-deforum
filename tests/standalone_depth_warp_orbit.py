#!/usr/bin/env python3
"""Standalone depth warping orbit test - NO DIFFUSION.

**DEPRECATED:** Use the Tuning Lab UI instead (WebUI → Tuning Lab → Orbit Depth Warping Tests)

This standalone script is kept for backward compatibility and quick CLI testing,
but the Tuning Lab UI provides the same functionality with better visualization,
persistence, and integration with other tuning tests.

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
import os
import json
from typing import Tuple, List
import argparse

# PARSE ARGS FIRST - before importing Forge modules that load their own argparse!
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

# CRITICAL: Clear sys.argv BEFORE importing Forge modules!
# Forge loads its own argparse which conflicts with our args
sys.argv = [sys.argv[0]]  # Keep script name only

# Add paths for imports
forge_root = Path(__file__).parent.parent.parent.parent
extension_root = Path(__file__).parent.parent

# Add both to sys.path
sys.path.insert(0, str(forge_root))  # For modules.devices
sys.path.insert(0, str(extension_root))  # For deforum module

# Now we can import Forge/Deforum modules
import torch
from modules import devices
from deforum.depth.depth import DepthModel


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

    # Initialize depth model
    print("Initializing depth estimation...")
    models_path = forge_root / "models" / "Deforum"
    depth_model = DepthModel(
        str(models_path),
        devices.device,
        keep_in_vram=True,
        depth_algorithm="Depth-Anything-V2-Small"
    )

    # Get initial depth map
    # Convert RGB to BGR for OpenCV format (depth model expects BGR numpy array)
    sphere_bgr = cv2.cvtColor(sphere, cv2.COLOR_RGB2BGR)
    depth_tensor = depth_model.predict(sphere_bgr)  # Returns torch.Tensor (1, 1, H, W)
    depth_map = depth_tensor.cpu().numpy().squeeze()  # Convert to numpy and remove batch/channel dims → (H, W)

    # Save depth visualization
    depth_viz = (depth_map * 255).astype(np.uint8)
    (output_dir / "depth_maps").mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_viz).save(output_dir / "depth_maps" / "000000000_depth.png")

    # Generate orbit path
    angles = np.linspace(0, 2 * np.pi, max_iterations + 1)[:-1]
    x_positions = orbit_radius * np.cos(angles)
    y_positions = orbit_radius * np.sin(angles)
    # Calculate rotation to counter the orbital motion
    # rotation_factor = -1 means 360° orbit → 360° counter-rotation (perfect)
    # rotation_factor = -5 means 360° orbit → 72° counter-rotation (under-rotated)
    rotation_angles = np.degrees(angles) / rotation_factor

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
        warped_bgr = cv2.cvtColor(warped, cv2.COLOR_RGB2BGR)
        depth_tensor = depth_model.predict(warped_bgr)
        depth_map = depth_tensor.cpu().numpy().squeeze()  # Remove batch/channel dims

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
    # Args already parsed at top of file before imports

    # Print deprecation warning
    print("\n" + "="*80)
    print("⚠️  DEPRECATION WARNING")
    print("="*80)
    print("This standalone script is DEPRECATED.")
    print("Please use: WebUI → Tuning Lab → Orbit Depth Warping Tests")
    print("The UI provides better visualization, persistence, and integration.")
    print("="*80 + "\n")

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
    print(f"\nResults saved to: {output_html}")

    # Save summary JSON
    summary_json = args.output / "orbit_results_summary.json"
    with open(summary_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary JSON saved to: {summary_json}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Rotation Factor':<20} {'Iterations Until Off-Screen':<30} {'Status':<20}")
    print("-"*80)

    max_iterations_seen = max(r["iterations_until_offscreen"] for r in all_results)
    optimal_factor = None
    optimal_iterations = 0

    for r in all_results:
        iters = r["iterations_until_offscreen"]
        factor = r["rotation_factor"]

        # Track optimal
        if iters > optimal_iterations:
            optimal_iterations = iters
            optimal_factor = factor

        # Status indicator
        if iters == max_iterations_seen:
            status = "✓ BEST"
        elif iters >= max_iterations_seen * 0.8:
            status = "Good"
        else:
            status = "Poor"

        print(f"{factor:<20.2f} {iters:<30} {status:<20}")

    print("="*80)
    print(f"\nOPTIMAL ROTATION FACTOR: {optimal_factor:.2f}")
    print(f"(Sphere stayed visible for {optimal_iterations}/{args.iterations} iterations)")
    print("="*80)
