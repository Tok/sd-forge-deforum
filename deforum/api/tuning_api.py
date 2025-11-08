"""Tuning API endpoints for automated parameter optimization.

This module provides REST API endpoints for running parameter sweeps
and retrieving quality metrics.
"""

import json
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
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

            # Mark as completed
            with self.test_lock:
                self.active_tests[test_id].status = "completed"
                logger.info(f"Test {test_id}: Completed all {total_tests} tests")

        except Exception as e:
            import traceback
            logger.error(f"Test {test_id} failed: {e}")
            logger.error(traceback.format_exc())
            with self.test_lock:
                self.active_tests[test_id].status = "failed"
                self.active_tests[test_id].error = str(e)

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
