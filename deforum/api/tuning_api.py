"""Tuning API endpoints for automated parameter optimization.

This module provides REST API endpoints for running parameter sweeps
and retrieving quality metrics.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
from fastapi import FastAPI, Response, status, HTTPException
from pydantic import BaseModel, Field
import gradio as gr

from deforum.utils.system.logging import get_logger

logger = get_logger()


class TuningTestType(str, Enum):
    """Types of tuning tests available."""
    COLOR_PRESERVATION = "color_preservation"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    FLUX_PARAMETER_SWEEP = "flux_parameter_sweep"
    DEPTH_WARPING_ORBIT = "depth_warping_orbit"
    RAFT_TUNING = "raft_tuning"


class TuningTestConfig(BaseModel):
    """Configuration for a tuning test run.

    Now supports fractional strength precision (0.01 resolution) via Forge monkey patches.
    Default step size changed from 0.05 to 0.01 to leverage fractional interpolation.
    """
    test_type: TuningTestType = Field(..., description="Type of test to run")
    steps: List[int] = Field([20], description="List of step counts to test")
    strength_min: float = Field(0.80, ge=0.0, le=1.0)
    strength_max: float = Field(0.95, ge=0.0, le=1.0)
    strength_step: float = Field(0.01, ge=0.001, le=0.1, description="Step size for strength sweep (0.01 = 1% precision)")
    kf_strength_min: float = Field(0.10, ge=0.0, le=0.5)
    kf_strength_max: float = Field(0.25, ge=0.0, le=0.5)
    kf_strength_step: float = Field(0.01, ge=0.001, le=0.1, description="Step size for keyframe strength sweep (0.01 = 1% precision)")
    max_iterations: int = Field(20, ge=1, le=50)
    grayscale_threshold: float = Field(20.0, ge=0.0, le=50.0)

    # Orbit-specific parameters (for depth_warping_orbit test type)
    aspect_ratios: Optional[List[List[float]]] = Field(
        None, description="List of [ratio, width, height] arrays for orbit tests"
    )
    rotation_factor_min: Optional[float] = Field(None, ge=-10.0, le=-1.0)
    rotation_factor_max: Optional[float] = Field(None, ge=-10.0, le=-1.0)
    rotation_factor_step: Optional[float] = Field(None, ge=0.01, le=2.0, description="Step size for rotation factor sweep (0.01 = fine, 0.5 = coarse)")
    orbit_radius: Optional[float] = Field(None, ge=1.0, le=100.0, description="Orbit radius in pixels (1.0 = very slow, 100.0 = fast)")
    orbit_iterations: Optional[int] = Field(None, ge=10, le=200, description="Number of orbit iterations (max 200 for slow orbits)")

    # RAFT-specific parameters (for raft_tuning test type)
    raft_rotation_factor: Optional[float] = Field(None, ge=-10.0, le=-1.0, description="Fixed rotation factor for RAFT tests")
    raft_model_sizes: Optional[List[str]] = Field(None, description="List of RAFT model sizes to test: ['Small', 'Large']")
    raft_flow_iterations_min: Optional[int] = Field(None, ge=6, le=50, description="Min RAFT flow refinement iterations")
    raft_flow_iterations_max: Optional[int] = Field(None, ge=6, le=50, description="Max RAFT flow refinement iterations")
    raft_flow_iterations_step: Optional[int] = Field(None, ge=2, le=10, description="Step size for flow iterations sweep")
    raft_flow_factor_min: Optional[float] = Field(None, ge=0.0, le=2.0, description="Min flow factor (0=depth-only, 1=normal, 2=strong RAFT)")
    raft_flow_factor_max: Optional[float] = Field(None, ge=0.0, le=2.0, description="Max flow factor")
    raft_flow_factor_step: Optional[float] = Field(None, ge=0.05, le=0.5, description="Step size for flow factor sweep")


class TuningTestStatus(BaseModel):
    """Status of a running tuning test."""
    test_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = Field(0.0, ge=0.0, le=1.0)
    current_config: Optional[Dict[str, Any]] = None
    results: List[Dict[str, Any]] = []
    error: Optional[str] = None


class TuningTestResult(BaseModel):
    """Result of a single test configuration."""
    steps: int
    normal_strength: float
    keyframe_strength: float
    iterations_completed: int
    final_color_score: float
    avg_temporal_consistency: float
    overall_score: float
    degradation_rate: float


class TuningTestManager:
    """Manages running tuning tests."""

    def __init__(self):
        self.active_tests: Dict[str, TuningTestStatus] = {}
        self.test_lock = threading.Lock()

    def start_test(self, test_id: str, config: TuningTestConfig) -> TuningTestStatus:
        """Start a new tuning test.

        Args:
            test_id: Unique identifier for this test
            config: Test configuration

        Returns:
            Initial test status
        """
        with self.test_lock:
            status = TuningTestStatus(
                test_id=test_id,
                status="pending",
                progress=0.0,
                current_config=config.dict(),
                results=[],
            )
            self.active_tests[test_id] = status
            logger.info(f"Started tuning test {test_id}: {config.test_type}")

        # Start test in background thread
        thread = threading.Thread(
            target=self._run_test,
            args=(test_id, config),
            daemon=True,
        )
        thread.start()

        return status

    def get_status(self, test_id: str) -> Optional[TuningTestStatus]:
        """Get status of a running test.

        Args:
            test_id: Test identifier

        Returns:
            Test status or None if not found
        """
        with self.test_lock:
            return self.active_tests.get(test_id)

    def cancel_test(self, test_id: str) -> bool:
        """Cancel a running test.

        Args:
            test_id: Test identifier

        Returns:
            True if test was cancelled, False if not found
        """
        with self.test_lock:
            if test_id in self.active_tests:
                status = self.active_tests[test_id]
                if status.status == "running":
                    status.status = "cancelled"
                    logger.info(f"Cancelled tuning test {test_id}")
                    return True
        return False

    def _run_test(self, test_id: str, config: TuningTestConfig):
        """Run the tuning test in background thread.

        Args:
            test_id: Test identifier
            config: Test configuration
        """
        try:
            with self.test_lock:
                self.active_tests[test_id].status = "running"

            # Route to appropriate test type
            if config.test_type == TuningTestType.DEPTH_WARPING_ORBIT:
                self._run_orbit_tests(test_id, config)
            elif config.test_type == TuningTestType.RAFT_TUNING:
                self._run_raft_tests(test_id, config)
            else:
                # Run standard I2V chaining tests (color preservation, temporal, flux)
                self._run_i2v_chaining_tests(test_id, config)

            # Mark as completed
            with self.test_lock:
                self.active_tests[test_id].status = "completed"
                logger.info(f"Test {test_id}: Completed")

        except Exception as e:
            import traceback
            logger.error(f"Test {test_id} failed: {e}")
            logger.error(traceback.format_exc())
            with self.test_lock:
                self.active_tests[test_id].status = "failed"
                self.active_tests[test_id].error = str(e)

    def _run_i2v_chaining_tests(self, test_id: str, config: TuningTestConfig):
        """Run I2V chaining tests (color preservation, temporal, flux).

        Args:
            test_id: Test identifier
            config: Test configuration
        """
        # Generate parameter combinations to test
        from numpy import arange

        strength_values = list(arange(
            config.strength_min,
            config.strength_max + 0.001,  # Add small epsilon to include max
            config.strength_step
        ))
        kf_strength_values = list(arange(
            config.kf_strength_min,
            config.kf_strength_max + 0.001,
            config.kf_strength_step
        ))

        total_tests = len(config.steps) * len(strength_values) * len(kf_strength_values)
        completed_tests = 0

        logger.info(f"Test {test_id}: {total_tests} parameter combinations to test")

        # Run each parameter combination
        for steps in config.steps:
            for normal_strength in strength_values:
                for kf_strength in kf_strength_values:
                    # Check if cancelled
                    with self.test_lock:
                        if self.active_tests[test_id].status == "cancelled":
                            return

                    # Run test with these parameters
                    logger.info(
                        f"Testing: steps={steps}, "
                        f"strength={normal_strength:.2f}, "
                        f"kf_strength={kf_strength:.2f}"
                    )

                    # TODO: Actually run the test using pytest/test infrastructure
                    # For now, just simulate with dummy results
                    result = self._run_single_test(
                        test_id,
                        steps,
                        normal_strength,
                        kf_strength,
                        config.max_iterations,
                        config.grayscale_threshold,
                    )

                    # Update progress
                    completed_tests += 1
                    with self.test_lock:
                        status = self.active_tests[test_id]
                        status.progress = completed_tests / total_tests
                        status.results.append(result)

    def _run_orbit_tests(self, test_id: str, config: TuningTestConfig):
        """Run depth warping orbit tests with rotation factor sweep.

        Args:
            test_id: Test identifier
            config: Test configuration with orbit parameters
        """
        from numpy import arange

        # DEBUG: Log what values we received from UI
        logger.info(f"DEBUG orbit_tests: config.orbit_radius={config.orbit_radius}, config.orbit_iterations={config.orbit_iterations}")

        # Validate orbit parameters
        if not config.aspect_ratios:
            raise ValueError("aspect_ratios required for orbit tests")
        if config.rotation_factor_min is None or config.rotation_factor_max is None:
            raise ValueError("rotation_factor_min/max required for orbit tests")
        if config.rotation_factor_step is None:
            raise ValueError("rotation_factor_step required for orbit tests")

        # Generate rotation factor values to test
        rotation_factors = list(arange(
            config.rotation_factor_min,
            config.rotation_factor_max + 0.01,  # Small epsilon to include max
            config.rotation_factor_step
        ))

        total_tests = len(config.aspect_ratios) * len(rotation_factors)
        completed_tests = 0

        logger.info(
            f"Test {test_id}: {total_tests} orbit configurations to test "
            f"({len(config.aspect_ratios)} aspect ratios × {len(rotation_factors)} factors)"
        )

        # Run each aspect ratio × rotation factor combination
        for aspect_config in config.aspect_ratios:
            aspect_ratio = aspect_config[0]
            width = int(aspect_config[1])
            height = int(aspect_config[2])
            for rotation_factor in rotation_factors:
                # Check if cancelled
                with self.test_lock:
                    if self.active_tests[test_id].status == "cancelled":
                        return

                # Run orbit test with these parameters
                actual_orbit_radius = config.orbit_radius or 2.0
                actual_orbit_iterations = config.orbit_iterations or 50

                logger.info(
                    f"Testing: aspect {aspect_ratio:.2f} ({width}×{height}), "
                    f"rotation_factor={rotation_factor:.1f}, "
                    f"orbit_radius={actual_orbit_radius}, orbit_iterations={actual_orbit_iterations}"
                )

                result = self._run_orbit_single_test(
                    test_id=test_id,
                    aspect_ratio=aspect_ratio,
                    width=width,
                    height=height,
                    rotation_factor=rotation_factor,
                    orbit_radius=actual_orbit_radius,  # Very slow orbit to see differences (was 50.0, then 10.0)
                    orbit_iterations=actual_orbit_iterations,  # Reduced to 50 since slower orbit takes longer
                )

                # Update progress
                completed_tests += 1
                with self.test_lock:
                    status = self.active_tests[test_id]
                    status.progress = completed_tests / total_tests
                    status.results.append(result)

        # Generate visualization graph after all tests complete
        self._generate_orbit_tuning_graph(test_id)

    def _run_raft_tests(self, test_id: str, config: TuningTestConfig):
        """Run RAFT optical flow parameter tuning tests.

        Tests different RAFT configurations on fixed orbit paths to find optimal
        flow settings that improve depth warping quality.

        Args:
            test_id: Test identifier
            config: Test configuration with RAFT parameters
        """
        from numpy import arange

        # Validate RAFT parameters
        if not config.aspect_ratios:
            raise ValueError("aspect_ratios required for RAFT tests")
        if config.raft_rotation_factor is None:
            raise ValueError("raft_rotation_factor required (use known-good from orbit tests)")
        if not config.raft_model_sizes:
            raise ValueError("raft_model_sizes required (e.g., ['Small', 'Large'])")
        if config.raft_flow_iterations_min is None or config.raft_flow_iterations_max is None:
            raise ValueError("raft_flow_iterations_min/max required")
        if config.raft_flow_factor_min is None or config.raft_flow_factor_max is None:
            raise ValueError("raft_flow_factor_min/max required")

        # Generate parameter sweep values
        flow_iterations_values = list(range(
            config.raft_flow_iterations_min,
            config.raft_flow_iterations_max + 1,
            config.raft_flow_iterations_step or 4
        ))
        flow_factor_values = list(arange(
            config.raft_flow_factor_min,
            config.raft_flow_factor_max + 0.01,
            config.raft_flow_factor_step or 0.2
        ))

        total_tests = (
            len(config.aspect_ratios) *
            len(config.raft_model_sizes) *
            len(flow_iterations_values) *
            len(flow_factor_values)
        )
        completed_tests = 0

        logger.info(
            f"Test {test_id}: {total_tests} RAFT configurations to test "
            f"({len(config.aspect_ratios)} aspects × {len(config.raft_model_sizes)} models × "
            f"{len(flow_iterations_values)} iterations × {len(flow_factor_values)} factors)"
        )

        # Use fixed rotation factor from orbit tests
        rotation_factor = config.raft_rotation_factor
        orbit_radius = config.orbit_radius or 2.0
        orbit_iterations = config.orbit_iterations or 50

        # Run depth-only baseline first (flow_factor=0) for comparison
        baseline_results = {}

        # Test each combination
        for aspect_config in config.aspect_ratios:
            aspect_ratio = aspect_config[0]
            width = int(aspect_config[1])
            height = int(aspect_config[2])

            # Run depth-only baseline
            logger.info(f"Running depth-only baseline for aspect {aspect_ratio:.2f}...")
            baseline_result = self._run_raft_single_test(
                test_id=test_id,
                aspect_ratio=aspect_ratio,
                width=width,
                height=height,
                rotation_factor=rotation_factor,
                orbit_radius=orbit_radius,
                orbit_iterations=orbit_iterations,
                model_size="Small",  # Doesn't matter for depth-only
                flow_iterations=12,  # Doesn't matter for depth-only
                flow_factor=0.0,  # CRITICAL: 0=depth-only baseline
            )
            baseline_results[aspect_ratio] = baseline_result

            # Now test RAFT configurations
            for model_size in config.raft_model_sizes:
                for flow_iterations in flow_iterations_values:
                    for flow_factor in flow_factor_values:
                        # Check if cancelled
                        with self.test_lock:
                            if self.active_tests[test_id].status == "cancelled":
                                return

                        # Skip flow_factor=0 (already did baseline)
                        if flow_factor == 0.0:
                            continue

                        logger.info(
                            f"Testing: aspect {aspect_ratio:.2f}, model={model_size}, "
                            f"iterations={flow_iterations}, flow_factor={flow_factor:.2f}"
                        )

                        result = self._run_raft_single_test(
                            test_id=test_id,
                            aspect_ratio=aspect_ratio,
                            width=width,
                            height=height,
                            rotation_factor=rotation_factor,
                            orbit_radius=orbit_radius,
                            orbit_iterations=orbit_iterations,
                            model_size=model_size,
                            flow_iterations=flow_iterations,
                            flow_factor=flow_factor,
                        )

                        # Calculate improvement vs baseline
                        baseline_iters = baseline_results[aspect_ratio]['iterations_until_offscreen']
                        raft_iters = result['iterations_until_offscreen']
                        if baseline_iters > 0:
                            improvement = ((raft_iters - baseline_iters) / baseline_iters) * 100.0
                        else:
                            improvement = 0.0
                        result['improvement_vs_depth_only'] = round(improvement, 1)

                        # Update progress
                        completed_tests += 1
                        with self.test_lock:
                            status = self.active_tests[test_id]
                            status.progress = completed_tests / total_tests
                            status.results.append(result)

        # Generate visualization graph after all tests complete
        self._generate_raft_tuning_graph(test_id)

    def _generate_orbit_tuning_graph(self, test_id: str):
        """Generate plotly visualization of orbit tuning results.

        Creates interactive graph showing rotation_factor vs iterations_until_offscreen
        for each aspect ratio tested.

        Args:
            test_id: Test identifier to get results from
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from pathlib import Path
        import os

        # Get test results
        with self.test_lock:
            if test_id not in self.active_tests:
                logger.warning(f"Test {test_id} not found, skipping graph generation")
                return
            results = self.active_tests[test_id].results

        if not results:
            logger.warning("No results to plot")
            return

        # Group results by aspect ratio
        aspect_groups = {}
        for result in results:
            aspect = result.get("aspect_ratio", 0)
            if aspect not in aspect_groups:
                aspect_groups[aspect] = {"rotation_factors": [], "iterations": []}
            aspect_groups[aspect]["rotation_factors"].append(result.get("rotation_factor", 0))
            aspect_groups[aspect]["iterations"].append(result.get("iterations_until_offscreen", 0))

        # Create figure
        fig = go.Figure()

        # Add trace for each aspect ratio
        colors = ["blue", "red", "green"]
        for idx, (aspect, data) in enumerate(sorted(aspect_groups.items())):
            fig.add_trace(go.Scatter(
                x=data["rotation_factors"],
                y=data["iterations"],
                mode="lines+markers",
                name=f"Aspect {aspect:.2f}",
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=8)
            ))

        # Update layout
        fig.update_layout(
            title="Orbit Tuning: Rotation Factor vs. Sphere Visibility",
            xaxis_title="Rotation Factor (translation/rotation ratio)",
            yaxis_title="Iterations Until Sphere Off-Screen",
            hovermode="x unified",
            template="plotly_white",
            width=1200,
            height=600
        )

        # Add annotation explaining metric
        fig.add_annotation(
            text="Higher iterations = more stable orbital movement",
            xref="paper", yref="paper",
            x=0.5, y=1.08, showarrow=False,
            font=dict(size=12, color="gray")
        )

        # Save graph
        forge_root = Path(os.getcwd())
        output_dir = forge_root / "outputs" / "deforum-tuning" / "depth_warping_orbits"
        output_path = output_dir / f"orbit_tuning_results_{test_id}.html"

        fig.write_html(str(output_path))
        logger.info(f"Orbit tuning graph saved to: {output_path}")

    def _count_iterations_until_offscreen(self, frames: list, width: int, height: int) -> int:
        """Count how many iterations until the sphere goes off-screen.

        Uses absolute brightness thresholds and area tracking to robustly detect
        when the sphere leaves the frame or becomes too faint.

        Args:
            frames: List of RGB frames as numpy arrays
            width: Frame width
            height: Frame height

        Returns:
            Number of iterations before sphere goes off-screen (or total if never off-screen)
        """
        import cv2
        import numpy as np

        # FIXED thresholds (not adaptive like Otsu)
        BRIGHTNESS_THRESHOLD = 180  # Sphere is bright white (200-255), background is gray (120-140)
        MIN_BRIGHT_PIXELS = 100     # Minimum pixels above threshold to count as "sphere visible"

        # Edge margin: sphere centroid must be this far from edges
        edge_margin = min(width, height) * 0.08  # 8% margin

        # Get reference sphere from first frame using FIXED threshold
        first_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        _, first_binary = cv2.threshold(first_gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
        first_contours, _ = cv2.findContours(first_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not first_contours:
            logger.warning(f"No bright sphere detected in first frame (threshold={BRIGHTNESS_THRESHOLD})")
            return 0

        # Get largest bright contour (should be sphere)
        first_sphere = max(first_contours, key=cv2.contourArea)
        ref_area = cv2.contourArea(first_sphere)
        ref_M = cv2.moments(first_sphere)
        ref_cx = ref_M['m10'] / ref_M['m00']
        ref_cy = ref_M['m01'] / ref_M['m00']

        logger.debug(f"Frame 0 reference: area={ref_area:.0f}px, centroid=({ref_cx:.0f}, {ref_cy:.0f})")

        # Track sphere across frames
        for i, frame in enumerate(frames):
            if i == 0:
                continue  # Skip first frame (reference)

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Count bright pixels (absolute threshold)
            bright_pixels = np.sum(gray > BRIGHTNESS_THRESHOLD)

            if bright_pixels < MIN_BRIGHT_PIXELS:
                # Too few bright pixels - sphere is gone or too faint
                logger.debug(f"Frame {i}: Only {bright_pixels} bright pixels (< {MIN_BRIGHT_PIXELS}), sphere off-screen")
                return i

            # Find bright contours using FIXED threshold
            _, binary = cv2.threshold(gray, BRIGHTNESS_THRESHOLD, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.debug(f"Frame {i}: No bright contours found, sphere off-screen")
                return i

            # Find largest bright contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            area_ratio = area / ref_area

            # Get centroid
            M = cv2.moments(largest)
            if M['m00'] == 0:
                logger.debug(f"Frame {i}: Zero moment, sphere off-screen")
                return i

            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            # Check if area dropped significantly (sphere mostly off-screen)
            if area_ratio < 0.25:  # 25% threshold
                logger.debug(f"Frame {i}: Area={area:.0f} ({area_ratio:.1%} of ref), sphere off-screen")
                return i

            # Check if centroid is too close to edges (sphere leaving frame)
            if (cx < edge_margin or cx > width - edge_margin or
                cy < edge_margin or cy > height - edge_margin):
                logger.debug(f"Frame {i}: Centroid ({cx:.0f}, {cy:.0f}) near edge (margin={edge_margin:.0f}), sphere off-screen")
                return i

        # Sphere stayed in frame for all iterations
        logger.debug(f"Sphere stayed in frame for all {len(frames)} iterations")
        return len(frames)

    def _generate_raft_tuning_graph(self, test_id: str):
        """Generate plotly visualization of RAFT tuning results.

        Creates interactive graph showing flow_factor vs iterations_until_offscreen
        for each RAFT configuration tested.

        Args:
            test_id: Test identifier to get results from
        """
        import plotly.graph_objects as go
        from pathlib import Path
        import os

        # Get test results
        with self.test_lock:
            if test_id not in self.active_tests:
                logger.warning(f"Test {test_id} not found, skipping graph generation")
                return
            results = self.active_tests[test_id].results

        if not results:
            logger.warning("No results to plot")
            return

        # Group results by aspect ratio and model size
        # Format: results[aspect][model_size] = {flow_factors: [...], iterations: [...]}
        grouped_results = {}
        for result in results:
            aspect = result.get("aspect_ratio", 0)
            model = result.get("model_size", "Unknown")

            key = f"{aspect}_{model}"
            if key not in grouped_results:
                grouped_results[key] = {"flow_factors": [], "iterations": []}

            grouped_results[key]["flow_factors"].append(result.get("flow_factor", 0))
            grouped_results[key]["iterations"].append(result.get("iterations_until_offscreen", 0))

        # Create figure
        fig = go.Figure()

        # Add trace for each aspect+model combination
        colors = ["blue", "red", "green", "orange", "purple", "brown"]
        for idx, (key, data) in enumerate(sorted(grouped_results.items())):
            aspect, model = key.split("_")
            fig.add_trace(go.Scatter(
                x=data["flow_factors"],
                y=data["iterations"],
                mode="lines+markers",
                name=f"Aspect {aspect} ({model})",
                line=dict(color=colors[idx % len(colors)], width=2),
                marker=dict(size=8)
            ))

        # Update layout
        fig.update_layout(
            title="RAFT Tuning: Flow Factor vs. Depth Warping Stability",
            xaxis_title="Flow Factor (0=depth-only, 1=normal RAFT, 2=strong RAFT)",
            yaxis_title="Iterations Until Sphere Off-Screen",
            hovermode="x unified",
            showlegend=True,
            height=600,
        )

        # Save graph
        forge_root = Path(os.getcwd())
        output_dir = forge_root / "outputs" / "deforum-tuning" / "raft_tests"
        output_path = output_dir / f"raft_tuning_results_{test_id}.html"
        output_dir.mkdir(parents=True, exist_ok=True)

        fig.write_html(str(output_path))
        logger.info(f"RAFT tuning graph saved to: {output_path}")

    def _generate_sphere_init_image(self, output_path: Path, width: int, height: int):
        """Generate a synthetic 3D sphere image with proper depth gradients.

        Creates a procedurally rendered sphere with Phong shading that provides
        clear depth cues for Depth-Anything estimation.

        Args:
            output_path: Where to save the generated sphere image
            width: Image width
            height: Image height
        """
        import numpy as np
        from PIL import Image

        logger.info(f"Generating synthetic 3D sphere ({width}x{height})...")

        # Create image array (RGB)
        img = np.ones((height, width, 3), dtype=np.uint8) * 128  # Gray background

        # Sphere parameters
        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) * 0.35  # 35% of shortest dimension

        # Light source position (top-left-front for clear depth gradient)
        light_pos = np.array([-1.0, -1.0, 2.0])
        light_pos = light_pos / np.linalg.norm(light_pos)

        # Generate sphere with Phong shading
        for y in range(height):
            for x in range(width):
                # Distance from center
                dx = x - center_x
                dy = y - center_y
                dist_sq = dx*dx + dy*dy

                if dist_sq <= radius*radius:
                    # Point is inside sphere
                    # Calculate z coordinate (sphere surface)
                    z = np.sqrt(radius*radius - dist_sq)

                    # Surface normal (pointing outward)
                    normal = np.array([dx, dy, z])
                    normal = normal / np.linalg.norm(normal)

                    # Diffuse lighting (Lambertian)
                    diffuse = max(0.0, np.dot(normal, light_pos))

                    # Ambient + diffuse
                    ambient = 0.2
                    intensity = ambient + (1.0 - ambient) * diffuse

                    # Map to color (white sphere)
                    color = int(255 * intensity)
                    img[y, x] = [color, color, color]

        # Save image
        pil_img = Image.fromarray(img, 'RGB')
        pil_img.save(output_path)
        logger.info(f"Synthetic 3D sphere saved to {output_path}")

    def _analyze_depth_maps(
        self,
        output_dir: Path,
        width: int,
        height: int,
        num_frames: int,
    ) -> Dict[str, Any]:
        """Analyze depth maps to verify depth warping quality.

        Extracts depth statistics to ensure:
        1. Sphere is detected with proper 3D geometry
        2. Depth estimates are stable across frames
        3. Depth gradients are smooth (no artifacts)
        4. Warping is applied correctly based on depth

        Args:
            output_dir: Directory containing depth_maps subdirectory
            width: Frame width
            height: Frame height
            num_frames: Number of frames to analyze

        Returns:
            Dictionary with depth analysis metrics
        """
        import numpy as np
        from PIL import Image

        depth_dir = output_dir / "depth-maps"
        if not depth_dir.exists():
            logger.warning(f"Depth maps directory not found: {depth_dir}")
            return self._empty_depth_metrics()

        # Load all depth maps
        depth_maps = []
        for i in range(num_frames):
            depth_file = depth_dir / f"{i:09d}_depth.png"
            if not depth_file.exists():
                logger.warning(f"Missing depth map: {depth_file}")
                continue

            # Load as grayscale (depth is single channel)
            depth_img = Image.open(depth_file).convert('L')
            depth_array = np.array(depth_img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            depth_maps.append(depth_array)

        if not depth_maps:
            logger.warning("No depth maps loaded")
            return self._empty_depth_metrics()

        logger.info(f"  Loaded {len(depth_maps)} depth maps for analysis")

        # Segment sphere from background using first frame
        # Sphere should be lighter (closer) than background
        first_depth = depth_maps[0]
        sphere_mask = self._segment_sphere_from_depth(first_depth, width, height)

        # Calculate sphere depth statistics
        sphere_depth_stats = []
        background_depth_stats = []

        for depth_map in depth_maps:
            # Sphere region
            sphere_pixels = depth_map[sphere_mask]
            if len(sphere_pixels) > 0:
                sphere_depth_stats.append({
                    'mean': float(np.mean(sphere_pixels)),
                    'std': float(np.std(sphere_pixels)),
                    'min': float(np.min(sphere_pixels)),
                    'max': float(np.max(sphere_pixels)),
                })
            else:
                # Sphere went off-screen
                sphere_depth_stats.append({
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                })

            # Background region
            bg_pixels = depth_map[~sphere_mask]
            if len(bg_pixels) > 0:
                background_depth_stats.append({
                    'mean': float(np.mean(bg_pixels)),
                    'std': float(np.std(bg_pixels)),
                })

        # Calculate temporal stability (frame-to-frame depth consistency)
        temporal_stability = self._calculate_depth_temporal_stability(
            [s['mean'] for s in sphere_depth_stats]
        )

        # Calculate depth gradient quality (smoothness of sphere surface)
        gradient_quality = self._calculate_depth_gradient_quality(depth_maps[0], sphere_mask)

        # Overall sphere depth metrics (frame 0)
        sphere_mean_depth = sphere_depth_stats[0]['mean']
        sphere_depth_range = sphere_depth_stats[0]['max'] - sphere_depth_stats[0]['min']
        bg_mean_depth = background_depth_stats[0]['mean']

        # Separation quality (how well sphere is distinguished from background)
        depth_separation = sphere_mean_depth - bg_mean_depth

        return {
            'sphere_mean_depth': round(sphere_mean_depth, 3),
            'sphere_depth_range': round(sphere_depth_range, 3),
            'background_mean_depth': round(bg_mean_depth, 3),
            'depth_separation': round(depth_separation, 3),
            'temporal_stability': round(temporal_stability, 3),
            'gradient_quality': round(gradient_quality, 3),
            'sphere_visible_frames': sum(1 for s in sphere_depth_stats if s['mean'] > 0),
        }

    def _segment_sphere_from_depth(
        self,
        depth_map: np.ndarray,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Segment sphere from background using depth thresholding.

        Args:
            depth_map: Depth map as numpy array [0, 1]
            width: Frame width
            height: Frame height

        Returns:
            Boolean mask (True = sphere, False = background)
        """
        import numpy as np

        # Sphere should be in center region and have higher depth values (closer)
        # Use Otsu's method to find optimal threshold
        from skimage.filters import threshold_otsu

        # Focus on center region where sphere should be
        center_x, center_y = width // 2, height // 2
        radius = min(width, height) // 3  # Conservative estimate
        y_coords, x_coords = np.ogrid[:height, :width]
        center_mask = ((x_coords - center_x)**2 + (y_coords - center_y)**2) <= radius**2

        # Find threshold using center region
        center_pixels = depth_map[center_mask]
        if len(center_pixels) == 0:
            return np.zeros_like(depth_map, dtype=bool)

        threshold = threshold_otsu(center_pixels)

        # Segment entire image using threshold
        sphere_mask = depth_map > threshold

        return sphere_mask

    def _calculate_depth_temporal_stability(self, sphere_mean_depths: list) -> float:
        """Calculate how stable sphere depth is across frames.

        Args:
            sphere_mean_depths: List of mean sphere depth values per frame

        Returns:
            Stability score [0, 1] where 1 = perfectly stable
        """
        import numpy as np

        if len(sphere_mean_depths) < 2:
            return 1.0

        # Filter out zeros (sphere off-screen)
        valid_depths = [d for d in sphere_mean_depths if d > 0]
        if len(valid_depths) < 2:
            return 1.0

        # Calculate coefficient of variation (std / mean)
        mean_depth = np.mean(valid_depths)
        std_depth = np.std(valid_depths)

        if mean_depth == 0:
            return 0.0

        cv = std_depth / mean_depth

        # Convert to stability score (lower CV = higher stability)
        # CV of 0.05 (5% variation) or less = perfect score
        stability = max(0.0, 1.0 - cv / 0.05)

        return stability

    def _calculate_depth_gradient_quality(self, depth_map: np.ndarray, sphere_mask: np.ndarray) -> float:
        """Calculate smoothness of depth gradients on sphere surface.

        Smooth gradients indicate proper 3D geometry detection.
        Noisy gradients indicate depth estimation errors.

        Args:
            depth_map: Depth map as numpy array [0, 1]
            sphere_mask: Boolean mask of sphere region

        Returns:
            Quality score [0, 1] where 1 = perfectly smooth gradients
        """
        import numpy as np
        from scipy.ndimage import sobel

        # Calculate gradients
        grad_x = sobel(depth_map, axis=1)
        grad_y = sobel(depth_map, axis=0)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Analyze gradients only on sphere surface
        sphere_gradients = gradient_magnitude[sphere_mask]

        if len(sphere_gradients) == 0:
            return 0.0

        # Calculate gradient smoothness
        # Lower variance = smoother gradients = better quality
        gradient_variance = np.var(sphere_gradients)

        # Normalize to [0, 1] quality score
        # Variance of 0.01 or less = perfect score
        quality = max(0.0, 1.0 - gradient_variance / 0.01)

        return quality

    def _empty_depth_metrics(self) -> Dict[str, Any]:
        """Return empty depth metrics when analysis fails."""
        return {
            'sphere_mean_depth': 0.0,
            'sphere_depth_range': 0.0,
            'background_mean_depth': 0.0,
            'depth_separation': 0.0,
            'temporal_stability': 0.0,
            'gradient_quality': 0.0,
            'sphere_visible_frames': 0,
        }

    def _run_orbit_single_test(
        self,
        test_id: str,
        aspect_ratio: float,
        width: int,
        height: int,
        rotation_factor: float,
        orbit_radius: float,
        orbit_iterations: int,
    ) -> Dict[str, Any]:
        """Run a single orbit test configuration.

        Args:
            test_id: Test identifier
            aspect_ratio: Width/height ratio
            width: Frame width in pixels
            height: Frame height in pixels
            rotation_factor: Translation/rotation ratio (negative = counter-rotation)
            orbit_radius: Orbit radius in pixels
            orbit_iterations: Number of I2I depth warp iterations

        Returns:
            Test result dictionary with drift metrics
        """
        from pathlib import Path
        import sys
        import numpy as np

        # Add tests directory to path
        tests_dir = Path(__file__).parent.parent.parent / "tests"
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))

        from integration.test_depth_warping_orbit_tuning import (
            generate_orbit_schedules,
            measure_subject_position_drift,
        )
        from integration.metrics import (
            measure_temporal_consistency,
            load_image_as_numpy,
        )
        from integration.utils import (
            API_BASE_URL,
            get_test_options_overrides,
            wait_for_job_to_complete,
            get_test_batch_name,
        )

        # Create test output directory in Forge root outputs (central location)
        # Note: Deforum appends batch_name to outdir_samples, so we use parent dir
        import os
        forge_root = Path(os.getcwd())  # Forge root directory
        output_dir = forge_root / "outputs" / "deforum-tuning" / "depth_warping_orbits"
        aspect_str = f"{int(aspect_ratio*100):03d}"
        # Use 2 decimal places to avoid collisions (-7.0 vs -6.95)
        test_name = f"aspect{aspect_str}_{width}x{height}_factor{abs(rotation_factor):.2f}"
        test_dir = output_dir / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running orbit test: {test_name}")

        # Use shared init image (3D sphere) for all tests to ensure comparable metrics
        # Generate it once on first test, reuse for all 81 tests
        shared_init_image = output_dir / "shared_sphere_init.png"

        try:
            # Generate shared sphere init image if it doesn't exist
            if not shared_init_image.exists():
                logger.info("Generating shared 3D sphere init image (first test only)...")
                self._generate_sphere_init_image(shared_init_image, width, height)
                logger.info(f"Shared sphere image saved to: {shared_init_image}")
            else:
                logger.info(f"Using existing shared sphere image: {shared_init_image}")

            # Generate orbit schedules
            schedules = generate_orbit_schedules(orbit_iterations, orbit_radius, rotation_factor)

            # Configure job - Deforum constructs outdir = outdir_samples + batch_name
            # So we set outdir_samples to parent directory, not test-specific directory
            options_overrides = {
                "outdir_samples": str(output_dir),  # Parent dir (Deforum appends batch_name)
                "deforum_save_gen_info_as_srt": False,
            }

            # Create minimal settings dict from scratch (no template)
            # Avoids 36+ single-keyframe schedules in template that cause KeyError: -1
            base_settings = {
                # Basic settings
                "W": width,
                "H": height,
                "seed": 42,  # Fixed seed for consistency
                "sampler": "euler",
                "steps": 20,
                "cfg_scale": 1.0,
                "distilled_cfg_scale": 3.5,

                # Use static init image (3D sphere) for ALL orbital tests
                # This ensures we measure depth warping quality, not generation randomness
                "use_init": True,
                "strength": 1.0,  # Perfect preservation = zero diffusion = pure depth warping only
                "strength_0_no_init": False,  # Use init on frame 0 too
                "init_image": str(shared_init_image),  # Path to shared sphere image

                # Animation settings
                "animation_mode": "3D",
                "render_mode": "keyframes_only",  # Pure depth warping for all tween frames
                "max_frames": orbit_iterations,
                "fps": 24,
                "save_depth_maps": True,  # Save ALL depth maps to verify depth warping on every frame

                # Camera schedules from orbit generation
                "translation_x": schedules["translation_x"],
                "translation_y": schedules["translation_y"],
                "translation_z": "0:(0)",
                "rotation_3d_x": "0:(0)",
                "rotation_3d_y": schedules["rotation_3d_y"],
                "rotation_3d_z": "0:(0)",

                # Prompt
                "animation_prompts": json.dumps({
                    "0": "a detailed 3D render of a colorful geometric sculpture, studio lighting"
                }),

                # Disable audio for tuning tests (faster, cleaner)
                "audio_mode": "None",
                "audio_sync": False,
                "add_soundtrack": "None",

                # Depth warping enabled (uses Depth-Anything V2)
                "use_depth_warping": True,
                "padding_mode": "border",
                "sampling_mode": "bicubic",

                # Disable other features
                "color_coherence": "None",
                "enable_subseed_scheduling": False,
                "enable_sampler_scheduling": False,
                "enable_clipskip_scheduling": False,
                "enable_checkpoint_scheduling": False,

                # Explicitly set numeric schedule fields to prevent single-keyframe defaults
                # Must use multi-frame format (not "0:(x)") to avoid KeyError: -1
                # OMIT string schedules entirely - parser has unfixable bugs:
                #   - Bug #1: Tries float() on strings → ValueError
                #   - Bug #2: Tries i-1 when i=0 → KeyError: -1
                # Let them use whatever default exists (disabled by enable_*_scheduling=False anyway)
                "clipskip_schedule": "0:(2), 1:(2)",
                "noise_schedule": "0:(0.02), 1:(0.02)",
                "strength_schedule": "0:(0.65), 1:(0.65)",
                "contrast_schedule": "0:(1.0), 1:(1.0)",
                "cfg_scale_schedule": "0:(1.0), 1:(1.0)",
                "distilled_cfg_scale_schedule": "0:(3.5), 1:(3.5)",
                "steps_schedule": "0:(20), 1:(20)",
                "seed_schedule": "0:(42), 1:(42)",
                "fov_schedule": "0:(70), 1:(70)",
                "near_schedule": "0:(200), 1:(200)",
                "far_schedule": "0:(10000), 1:(10000)",
                "aspect_ratio_schedule": "0:(1.0), 1:(1.0)",
                "subseed_schedule": "0:(1), 1:(1)",
                "subseed_strength_schedule": "0:(0), 1:(0)",
                # mask_schedule and noise_mask_schedule omitted - string schedules with parser bugs
                "noise_multiplier_schedule": "0:(1.0), 1:(1.0)",
                "ddim_eta_schedule": "0:(0), 1:(0)",
                "ancestral_eta_schedule": "0:(1), 1:(1)",
                "amount_schedule": "0:(0), 1:(0)",
                "kernel_schedule": "0:(5), 1:(5)",
                "sigma_schedule": "0:(1), 1:(1)",
                "threshold_schedule": "0:(0), 1:(0)",
                "cadence_flow_factor_schedule": "0:(1), 1:(1)",
                "redo_flow_factor_schedule": "0:(1), 1:(1)",
                "image_strength_schedule": "0:(0.85), 1:(0.85)",
                "image_keyframe_strength_schedule": "0:(0.20), 1:(0.20)",
                "blendFactorMax": "0:(0.35), 1:(0.35)",
                "blendFactorSlope": "0:(0.25), 1:(0.25)",
                "tweening_frames_schedule": "0:(20), 1:(20)",
                "color_correction_factor": "0:(0.075), 1:(0.075)",

                # Output - batch_name gets appended to outdir_samples by Deforum
                "batch_name": test_name,  # Just the test name (aspect177_512x288_factor7.0)
                # Don't set "outdir" - Deforum constructs it from outdir_samples + batch_name
            }

            # Submit job
            import requests
            response = requests.post(
                f"{API_BASE_URL}/batches",
                json={
                    "deforum_settings": base_settings,
                    "options_overrides": options_overrides,
                }
            )
            response.raise_for_status()
            job_data = response.json()
            job_ids = job_data["job_ids"]  # Batches endpoint returns array
            job_id = job_ids[0]  # Get first job from batch

            logger.info(f"  Submitted job {job_id}, waiting for completion...")

            # Wait for completion (timeout handled by @retry decorator)
            logger.info(f"  About to call wait_for_job_to_complete for job {job_id}...")
            try:
                job_status = wait_for_job_to_complete(job_id)  # No timeout param - handled by @retry
                logger.info(f"  wait_for_job_to_complete returned successfully!")
                logger.info(f"  Job {job_id} completed with status: {job_status.status}")
            except Exception as e:
                logger.error(f"  wait_for_job_to_complete raised exception: {e}")
                raise

            # Load generated frames from job output directory
            # Frames are at root of outdir, not in timestamped subdirectory
            output_dir = Path(job_status.outdir)
            logger.info(f"  Looking for frames in: {output_dir}")

            # Find frame PNGs (exclude depth maps in subdirectory)
            frame_files = sorted(
                [f for f in output_dir.glob("*.png") if f.stem.isdigit()],
                key=lambda p: int(p.stem)
            )
            logger.info(f"  Found {len(frame_files)} frames")

            if not frame_files:
                raise ValueError(f"No frames generated for job {job_id} in {output_dir}")

            frames = [load_image_as_numpy(str(f)) for f in frame_files]
            logger.info(f"  Loaded {len(frames)} frames as numpy arrays")

            # Measure when sphere goes off-screen (primary metric)
            logger.info(f"  Checking sphere visibility in each frame...")
            iterations_until_offscreen = self._count_iterations_until_offscreen(frames, width, height)
            logger.info(f"  Sphere stayed in frame for {iterations_until_offscreen} iterations")

            # Also measure drift for additional context
            logger.info(f"  Measuring subject position drift...")
            drift_metrics = measure_subject_position_drift(frames)

            # Analyze depth maps to verify warping quality
            logger.info(f"  Analyzing depth maps...")
            depth_metrics = self._analyze_depth_maps(
                output_dir=output_dir,
                width=width,
                height=height,
                num_frames=len(frames),
            )

            result = {
                "aspect_ratio": round(aspect_ratio, 2),
                "width": width,
                "height": height,
                "rotation_factor": round(rotation_factor, 2),
                "orbit_radius": orbit_radius,
                "total_frames": len(frames),
                "iterations_until_offscreen": iterations_until_offscreen,  # PRIMARY METRIC
                "max_drift": round(drift_metrics['max_drift'], 1),
                "avg_drift": round(drift_metrics['avg_drift'], 1),
                # Depth analysis metrics
                "sphere_mean_depth": depth_metrics['sphere_mean_depth'],
                "sphere_depth_range": depth_metrics['sphere_depth_range'],
                "background_depth": depth_metrics['background_mean_depth'],
                "depth_separation": depth_metrics['depth_separation'],
                "depth_temporal_stability": depth_metrics['temporal_stability'],
                "depth_gradient_quality": depth_metrics['gradient_quality'],
            }

            logger.info(
                f"  Results: iterations_until_offscreen={iterations_until_offscreen}, "
                f"max_drift={drift_metrics['max_drift']:.1f}px, "
                f"depth_quality={depth_metrics['gradient_quality']:.2f}, "
                f"depth_stability={depth_metrics['temporal_stability']:.2f}"
            )

            return result

        except Exception as e:
            import traceback
            logger.error(f"Orbit test failed: {e}")
            logger.error(traceback.format_exc())
            # Return partial results on error
            return {
                "aspect_ratio": round(aspect_ratio, 2),
                "width": width,
                "height": height,
                "rotation_factor": round(rotation_factor, 1),
                "orbit_radius": orbit_radius,
                "iterations": 0,
                "max_drift": 0.0,
                "avg_drift": 0.0,
                "drift_rate": 0.0,
                "temporal_consistency": 0.0,
                "overall_score": 0.0,
                "error": str(e),
            }

    def _run_raft_single_test(
        self,
        test_id: str,
        aspect_ratio: float,
        width: int,
        height: int,
        rotation_factor: float,
        orbit_radius: float,
        orbit_iterations: int,
        model_size: str,
        flow_iterations: int,
        flow_factor: float,
    ) -> Dict[str, Any]:
        """Run a single RAFT test configuration.

        Similar to _run_orbit_single_test but with RAFT optical flow enabled.

        Args:
            test_id: Test identifier
            aspect_ratio: Width/height ratio
            width: Frame width
            height: Frame height
            rotation_factor: Translation/rotation ratio (fixed, from orbit tests)
            orbit_radius: Orbit radius in pixels
            orbit_iterations: Number of depth warp iterations
            model_size: RAFT model size ('Small' or 'Large')
            flow_iterations: RAFT refinement iterations
            flow_factor: Flow guidance strength (0=depth-only, 1=normal, 2=strong)

        Returns:
            Test result dictionary with RAFT-specific metrics
        """
        from pathlib import Path
        import sys
        import numpy as np

        # Add tests directory to path
        tests_dir = Path(__file__).parent.parent.parent / "tests"
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))

        from integration.test_depth_warping_orbit_tuning import (
            generate_orbit_schedules,
            measure_subject_position_drift,
        )
        from integration.metrics import (
            load_image_as_numpy,
        )
        from integration.utils import (
            API_BASE_URL,
            wait_for_job_to_complete,
        )

        # Create test output directory
        import os
        forge_root = Path(os.getcwd())
        output_dir = forge_root / "outputs" / "deforum-tuning" / "raft_tests"
        aspect_str = f"{int(aspect_ratio*100):03d}"
        test_name = (
            f"aspect{aspect_str}_{width}x{height}_"
            f"model{model_size}_iter{flow_iterations}_factor{flow_factor:.2f}"
        )
        test_dir = output_dir / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running RAFT test: {test_name}")

        # Use shared init image (same as orbit tests)
        shared_init_image = output_dir.parent / "depth_warping_orbits" / "shared_sphere_init.png"

        try:
            # Ensure shared sphere exists
            if not shared_init_image.exists():
                logger.info("Generating shared sphere init image...")
                self._generate_sphere_init_image(shared_init_image, width, height)

            # Generate orbit schedules
            schedules = generate_orbit_schedules(orbit_iterations, orbit_radius, rotation_factor)

            # Configure job with RAFT enabled
            options_overrides = {
                "outdir_samples": str(output_dir),
                "deforum_save_gen_info_as_srt": False,
            }

            # Base settings (similar to orbit test)
            base_settings = {
                # Basic settings
                "W": width,
                "H": height,
                "seed": 42,
                "sampler": "euler",
                "steps": 20,
                "cfg_scale": 1.0,
                "distilled_cfg_scale": 3.5,

                # Use static init image
                "use_init": True,
                "strength": 1.0,  # Perfect preservation = pure depth/flow warping
                "strength_0_no_init": False,
                "init_image": str(shared_init_image),

                # Animation settings
                "animation_mode": "3D",
                "render_mode": "keyframes_only",
                "max_frames": orbit_iterations,
                "fps": 24,
                "save_depth_maps": True,

                # Camera schedules
                "translation_x": schedules["translation_x"],
                "translation_y": schedules["translation_y"],
                "translation_z": "0:(0)",
                "rotation_3d_x": "0:(0)",
                "rotation_3d_y": schedules["rotation_3d_y"],
                "rotation_3d_z": "0:(0)",

                # Prompt
                "animation_prompts": json.dumps({
                    "0": "a detailed 3D render of a colorful geometric sculpture, studio lighting"
                }),

                # Disable audio
                "audio_mode": "None",
                "audio_sync": False,
                "add_soundtrack": "None",

                # Depth warping enabled (uses Depth-Anything V2)
                "use_depth_warping": True,
                "padding_mode": "border",
                "sampling_mode": "bicubic",

                # RAFT OPTICAL FLOW SETTINGS (KEY DIFFERENCE FROM ORBIT TESTS)
                "optical_flow_cadence": "RAFT" if flow_factor > 0 else "None",
                "raft_model_size": model_size,
                "raft_flow_iterations": flow_iterations,
                "cadence_flow_factor_schedule": f"0:({flow_factor}), 1:({flow_factor})",
                "show_flow_arrows": True,  # Visualize flow vectors

                # Disable other features
                "color_coherence": "None",
                "enable_subseed_scheduling": False,
                "enable_sampler_scheduling": False,
                "enable_clipskip_scheduling": False,
                "enable_checkpoint_scheduling": False,

                # Required schedules
                "clipskip_schedule": "0:(2), 1:(2)",
                "noise_schedule": "0:(0.02), 1:(0.02)",
                "strength_schedule": "0:(0.65), 1:(0.65)",
                "contrast_schedule": "0:(1.0), 1:(1.0)",
                "cfg_scale_schedule": "0:(1.0), 1:(1.0)",
                "distilled_cfg_scale_schedule": "0:(3.5), 1:(3.5)",
                "steps_schedule": "0:(20), 1:(20)",
                "seed_schedule": "0:(42), 1:(42)",
                "fov_schedule": "0:(70), 1:(70)",
                "near_schedule": "0:(200), 1:(200)",
                "far_schedule": "0:(10000), 1:(10000)",
                "aspect_ratio_schedule": "0:(1.0), 1:(1.0)",
                "subseed_schedule": "0:(1), 1:(1)",
                "subseed_strength_schedule": "0:(0), 1:(0)",
                "noise_multiplier_schedule": "0:(1.0), 1:(1.0)",
                "ddim_eta_schedule": "0:(0), 1:(0)",
                "ancestral_eta_schedule": "0:(1), 1:(1)",
                "amount_schedule": "0:(0), 1:(0)",
                "kernel_schedule": "0:(5), 1:(5)",
                "sigma_schedule": "0:(1), 1:(1)",
                "threshold_schedule": "0:(0), 1:(0)",
                "redo_flow_factor_schedule": "0:(1), 1:(1)",
                "image_strength_schedule": "0:(0.85), 1:(0.85)",
                "image_keyframe_strength_schedule": "0:(0.20), 1:(0.20)",
                "blendFactorMax": "0:(0.35), 1:(0.35)",
                "blendFactorSlope": "0:(0.25), 1:(0.25)",
                "tweening_frames_schedule": "0:(20), 1:(20)",
                "color_correction_factor": "0:(0.075), 1:(0.075)",

                # Output
                "batch_name": test_name,
            }

            # Submit job
            import requests
            response = requests.post(
                f"{API_BASE_URL}/batches",
                json={
                    "deforum_settings": base_settings,
                    "options_overrides": options_overrides,
                }
            )
            response.raise_for_status()
            job_data = response.json()
            job_ids = job_data["job_ids"]
            job_id = job_ids[0]

            logger.info(f"  Submitted RAFT job {job_id}, waiting for completion...")

            # Wait for completion
            job_status = wait_for_job_to_complete(job_id)
            logger.info(f"  Job {job_id} completed")

            # Load frames
            output_dir = Path(job_status.outdir)
            frame_files = sorted(
                [f for f in output_dir.glob("*.png") if f.stem.isdigit()],
                key=lambda p: int(p.stem)
            )

            if not frame_files:
                raise ValueError(f"No frames generated for job {job_id}")

            frames = [load_image_as_numpy(str(f)) for f in frame_files]

            # Measure stability
            iterations_until_offscreen = self._count_iterations_until_offscreen(frames, width, height)
            drift_metrics = measure_subject_position_drift(frames)

            # Analyze depth maps
            depth_metrics = self._analyze_depth_maps(
                output_dir=output_dir,
                width=width,
                height=height,
                num_frames=len(frames),
            )

            # Calculate flow consistency (placeholder for now)
            flow_consistency = 0.0  # TODO: Implement flow vector analysis

            result = {
                "aspect_ratio": round(aspect_ratio, 2),
                "width": width,
                "height": height,
                "rotation_factor": round(rotation_factor, 2),
                "model_size": model_size,
                "flow_iterations": flow_iterations,
                "flow_factor": round(flow_factor, 2),
                "iterations_until_offscreen": iterations_until_offscreen,
                "max_drift": round(drift_metrics['max_drift'], 1),
                "flow_consistency": round(flow_consistency, 3),
                "sphere_mean_depth": depth_metrics['sphere_mean_depth'],
                "depth_temporal_stability": depth_metrics['temporal_stability'],
                "depth_gradient_quality": depth_metrics['gradient_quality'],
            }

            logger.info(
                f"  Results: RAFT({model_size}, iter={flow_iterations}, factor={flow_factor:.2f}) → "
                f"stability={iterations_until_offscreen} frames, drift={drift_metrics['max_drift']:.1f}px"
            )

            return result

        except Exception as e:
            import traceback
            logger.error(f"RAFT test failed: {e}")
            logger.error(traceback.format_exc())
            return {
                "aspect_ratio": round(aspect_ratio, 2),
                "model_size": model_size,
                "flow_iterations": flow_iterations,
                "flow_factor": round(flow_factor, 2),
                "iterations_until_offscreen": 0,
                "max_drift": 0.0,
                "flow_consistency": 0.0,
                "improvement_vs_depth_only": 0.0,
                "error": str(e),
            }

    def _run_single_test(
        self,
        test_id: str,
        steps: int,
        normal_strength: float,
        kf_strength: float,
        max_iterations: int,
        grayscale_threshold: float,
    ) -> Dict[str, Any]:
        """Run a single parameter configuration test.

        Args:
            test_id: Test identifier
            steps: Number of sampling steps
            normal_strength: Normal/tween frame strength
            kf_strength: Keyframe strength
            max_iterations: Max I2V chaining iterations
            grayscale_threshold: Grayscale detection threshold

        Returns:
            Test result dictionary
        """
        from pathlib import Path
        import sys

        # Import test utilities (without pytest dependency)
        from deforum.api.tuning_test_helpers import (
            create_colorful_test_image,
            run_i2v_iteration,
        )

        # Add tests directory to path for metrics
        tests_dir = Path(__file__).parent.parent.parent / "tests"
        if str(tests_dir) not in sys.path:
            sys.path.insert(0, str(tests_dir))

        from tuning.metrics import (
            measure_color_preservation,
            measure_temporal_consistency,
            calculate_comprehensive_quality_score,
            load_image_as_numpy,
        )

        # Create test output directory
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "deforum-tuning" / "color_preservation"
        test_name = f"steps{steps}_norm{normal_strength:.2f}_kf{kf_strength:.2f}"
        test_dir = output_dir / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running real test: {test_name}")

        try:
            # Generate colorful test image (only once, reuse if exists)
            test_image_path = output_dir / "test_input_rainbow.png"
            if not test_image_path.exists():
                test_image_path = create_colorful_test_image(output_dir)
                logger.info(f"Created test image: {test_image_path}")

            # Track metrics
            color_scores = []
            frames = []
            current_image = test_image_path

            # Run I2V chaining iterations
            for iteration in range(max_iterations):
                # Check if cancelled
                with self.test_lock:
                    if self.active_tests[test_id].status == "cancelled":
                        logger.info(f"Test {test_id} cancelled at iteration {iteration}")
                        break

                logger.info(f"  Iteration {iteration + 1}/{max_iterations}...")

                # Run I2V generation
                output_frame = run_i2v_iteration(
                    init_image_path=current_image,
                    strength=normal_strength,
                    keyframe_strength=kf_strength,
                    steps=steps,
                    output_dir=test_dir,
                )

                # Load and measure
                frame_array = load_image_as_numpy(output_frame)
                frames.append(frame_array)

                color_score = measure_color_preservation(frame_array)
                color_scores.append(color_score)

                logger.info(f"    Color score: {color_score:.1f}/100")

                # Check if grayscale threshold reached
                if color_score < grayscale_threshold:
                    logger.info(f"  Grayscale threshold reached at iteration {iteration + 1}")
                    break

                # Use this output as input for next iteration
                current_image = output_frame

            # Calculate comprehensive metrics
            if frames:
                metrics = calculate_comprehensive_quality_score(frames)

                return {
                    "steps": steps,
                    "normal_strength": round(normal_strength, 2),
                    "keyframe_strength": round(kf_strength, 2),
                    "iterations_completed": len(color_scores),
                    "final_color_score": round(color_scores[-1] if color_scores else 0, 1),
                    "avg_temporal_consistency": round(metrics['avg_temporal'], 1),
                    "overall_score": round(metrics['overall_score'], 1),
                    "degradation_rate": round(metrics['degradation_rate'], 2),
                }
            else:
                # No frames generated (cancelled immediately)
                return {
                    "steps": steps,
                    "normal_strength": round(normal_strength, 2),
                    "keyframe_strength": round(kf_strength, 2),
                    "iterations_completed": 0,
                    "final_color_score": 0.0,
                    "avg_temporal_consistency": 0.0,
                    "overall_score": 0.0,
                    "degradation_rate": 0.0,
                }

        except Exception as e:
            import traceback
            logger.error(f"Test failed: {e}")
            logger.error(traceback.format_exc())
            # Return partial results on error
            return {
                "steps": steps,
                "normal_strength": round(normal_strength, 2),
                "keyframe_strength": round(kf_strength, 2),
                "iterations_completed": 0,
                "final_color_score": 0.0,
                "avg_temporal_consistency": 0.0,
                "overall_score": 0.0,
                "degradation_rate": 0.0,
                "error": str(e),
            }


# Global test manager instance
_test_manager = TuningTestManager()


def tuning_api(_: gr.Blocks, app: FastAPI):
    """Register tuning API endpoints.

    Args:
        _: Gradio Blocks instance (unused)
        app: FastAPI application instance
    """
    @app.post(
        "/deforum_api/tuning/start",
        response_model=TuningTestStatus,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["Tuning"],
        summary="Start a new parameter tuning test",
    )
    def start_tuning_test(config: TuningTestConfig):
        """Start a new automated parameter tuning test."""
        import uuid
        test_id = f"tuning_{uuid.uuid4().hex[:8]}"
        return _test_manager.start_test(test_id, config)

    @app.get(
        "/deforum_api/tuning/{test_id}",
        response_model=TuningTestStatus,
        tags=["Tuning"],
        summary="Get status of a tuning test",
    )
    def get_tuning_status(test_id: str):
        """Get the current status of a tuning test."""
        status = _test_manager.get_status(test_id)
        if not status:
            raise HTTPException(status_code=404, detail=f"Test {test_id} not found")
        return status

    @app.post(
        "/deforum_api/tuning/{test_id}/cancel",
        tags=["Tuning"],
        summary="Cancel a running tuning test",
    )
    def cancel_tuning_test(test_id: str):
        """Cancel a running tuning test."""
        if _test_manager.cancel_test(test_id):
            return {"message": f"Test {test_id} cancelled"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Test {test_id} not found or not running"
            )

    logger.info("Registered tuning API endpoints")
