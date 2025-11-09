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
    orbit_radius: Optional[float] = Field(None, ge=20.0, le=100.0)
    orbit_iterations: Optional[int] = Field(None, ge=10, le=40)


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
                logger.info(
                    f"Testing: aspect {aspect_ratio:.2f} ({width}×{height}), "
                    f"rotation_factor={rotation_factor:.1f}"
                )

                result = self._run_orbit_single_test(
                    test_id=test_id,
                    aspect_ratio=aspect_ratio,
                    width=width,
                    height=height,
                    rotation_factor=rotation_factor,
                    orbit_radius=config.orbit_radius or 50.0,
                    orbit_iterations=config.orbit_iterations or 20,
                )

                # Update progress
                completed_tests += 1
                with self.test_lock:
                    status = self.active_tests[test_id]
                    status.progress = completed_tests / total_tests
                    status.results.append(result)

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

        # Create test output directory
        output_dir = Path(__file__).parent.parent.parent / "outputs" / "deforum-tuning" / "depth_warping_orbits"
        aspect_str = f"{int(aspect_ratio*100):03d}"
        test_name = f"aspect{aspect_str}_{width}x{height}_factor{abs(rotation_factor):.1f}"
        test_dir = output_dir / test_name
        test_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running orbit test: {test_name}")

        try:
            # Generate orbit schedules
            schedules = generate_orbit_schedules(orbit_iterations, orbit_radius, rotation_factor)

            # Configure job
            options_overrides = get_test_options_overrides()
            options_overrides.update({
                "deforum_save_gen_info_as_srt": False,
            })

            # Create minimal settings dict from scratch (no template)
            # Avoids 36+ single-keyframe schedules in template that cause KeyError: -1
            base_settings = {
                # Basic settings
                "W": width,
                "H": height,
                "seed": 42,
                "sampler": "euler",
                "steps": 20,
                "cfg_scale": 1.0,
                "distilled_cfg_scale": 3.5,

                # Animation settings
                "animation_mode": "3D",
                "render_mode": "new_3d",
                "max_frames": orbit_iterations,
                "fps": 24,
                "save_depth_maps": True,

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

                # Disable audio
                "audio_mode": "None",
                "audio_sync": False,

                # Depth warping enabled
                "use_depth_warping": True,
                "midas_weight": 0.3,
                "padding_mode": "border",
                "sampling_mode": "bicubic",

                # Disable other features
                "color_coherence": "None",
                "enable_subseed_scheduling": False,
                "enable_sampler_scheduling": False,
                "enable_clipskip_scheduling": False,
                "enable_checkpoint_scheduling": False,

                # Explicitly set ALL schedule fields to prevent defaults from kicking in
                # Must use multi-frame format (not "0:(x)") to avoid KeyError: -1
                # String-based schedules MUST use quoted strings, not numbers
                "checkpoint_schedule": '0:("flux1-dev"), 1:("flux1-dev")',
                "clipskip_schedule": "0:(2), 1:(2)",
                "sampler_schedule": '0:("euler"), 1:("euler")',
                "scheduler_schedule": '0:("Simple"), 1:("Simple")',
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
                "mask_schedule": '0:("{video_mask}"), 1:("{video_mask}")',
                "noise_mask_schedule": '0:("{video_mask}"), 1:("{video_mask}")',
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

                # Output directory
                "batch_name": get_test_batch_name(test_name),
                "outdir": str(test_dir),
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

            # Wait for completion
            wait_for_job_to_complete(job_id, timeout=600)

            # Load generated frames
            batch_name = get_test_batch_name(job_id)
            frame_pattern = f"{batch_name}_*.png"
            frame_files = sorted(
                Path(job_data["outdir_samples"]).glob(frame_pattern),
                key=lambda p: int(p.stem.split('_')[-1])
            )

            if not frame_files:
                raise ValueError(f"No frames generated for job {job_id}")

            frames = [load_image_as_numpy(str(f)) for f in frame_files]
            logger.info(f"  Loaded {len(frames)} frames")

            # Measure metrics
            drift_metrics = measure_subject_position_drift(frames)
            temporal_metrics = measure_temporal_consistency(frames)

            # Calculate overall quality score
            # Lower drift = better, higher temporal = better
            # Normalize drift (assume 100px is "very bad")
            drift_score = max(0, 100 - drift_metrics['max_drift'])
            temporal_score = temporal_metrics['avg_ssim'] * 100

            overall_score = (drift_score * 0.6) + (temporal_score * 0.4)

            result = {
                "aspect_ratio": round(aspect_ratio, 2),
                "width": width,
                "height": height,
                "rotation_factor": round(rotation_factor, 1),
                "orbit_radius": orbit_radius,
                "iterations": len(frames),
                "max_drift": round(drift_metrics['max_drift'], 1),
                "avg_drift": round(drift_metrics['avg_drift'], 1),
                "drift_rate": round(drift_metrics['drift_rate'], 2),
                "temporal_consistency": round(temporal_score, 1),
                "overall_score": round(overall_score, 1),
            }

            logger.info(
                f"  Results: drift={drift_metrics['max_drift']:.1f}px, "
                f"temporal={temporal_score:.1f}, overall={overall_score:.1f}"
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
