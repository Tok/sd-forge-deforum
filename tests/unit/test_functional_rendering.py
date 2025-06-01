#!/usr/bin/env python3

"""
Comprehensive tests for Deforum's functional rendering system.
Tests immutable frame processing, functional pipelines, and rendering logic.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
from pathlib import Path
import json

# Import the rendering system from the new package structure  
from deforum.rendering.frame_models import (
    FrameData,
    RenderingState,
    ImmutableFrameProcessor,
    create_frame_data,
    validate_frame_data
)

from deforum.rendering.frame_processing import (
    process_frame_functional,
    create_frame_processor,
    apply_frame_transformations,
    validate_processing_pipeline
)

from deforum.rendering.rendering_pipeline import (
    FunctionalRenderingPipeline,
    create_rendering_pipeline,
    execute_rendering_step,
    validate_pipeline_configuration
)

from deforum.rendering.legacy_renderer import (
    render_animation_functional,
    convert_legacy_args,
    ensure_backward_compatibility,
    migrate_to_functional_rendering
)

# Import the functional rendering system
try:
    from scripts.deforum_helpers.rendering.frame_models import (
        FrameState, FrameResult, FrameMetadata, RenderContext,
        ProcessingStage, RenderingError, ModelState, RenderingSession
    )
    from scripts.deforum_helpers.rendering.frame_processing import (
        create_frame_state, process_frame, validate_frame_state,
        apply_frame_transformations, merge_frame_results
    )
    from scripts.deforum_helpers.rendering.rendering_pipeline import (
        RenderingPipeline, create_rendering_pipeline, execute_pipeline,
        render_animation_functional, create_progress_tracker, PipelineConfig
    )
    from scripts.deforum_helpers.rendering.legacy_renderer import (
        convert_legacy_args_to_context, enable_functional_rendering,
        functional_render_animation, validate_legacy_compatibility
    )
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_AVAILABLE = False


@unittest.skipUnless(IMPORTS_AVAILABLE, "Functional rendering modules not available")
class TestFrameModels(unittest.TestCase):
    """Test immutable frame state models."""
    
    def test_frame_metadata_creation(self):
        """Test FrameMetadata creation and validation."""
        metadata = FrameMetadata(
            frame_idx=0,
            timestamp=0.0,
            seed=42,
            strength=0.75,
            cfg_scale=7.0,
            distilled_cfg_scale=7.0,
            noise_level=0.1,
            prompt="test prompt"
        )
        
        self.assertEqual(metadata.frame_idx, 0)
        self.assertEqual(metadata.seed, 42)
        self.assertEqual(metadata.prompt, "test prompt")
    
    def test_frame_metadata_validation(self):
        """Test FrameMetadata validation."""
        # Test invalid frame index
        with self.assertRaises(ValueError):
            FrameMetadata(
                frame_idx=-1,
                timestamp=0.0,
                seed=42,
                strength=0.75,
                cfg_scale=7.0,
                distilled_cfg_scale=7.0,
                noise_level=0.0,
                prompt=""
            )
        
        # Test invalid strength
        with self.assertRaises(ValueError):
            FrameMetadata(
                frame_idx=0,
                timestamp=0.0,
                seed=42,
                strength=1.5,  # Invalid
                cfg_scale=7.0,
                distilled_cfg_scale=7.0,
                noise_level=0.0,
                prompt=""
            )
    
    def test_frame_state_immutability(self):
        """Test that FrameState is immutable."""
        metadata = FrameMetadata(
            frame_idx=0, timestamp=0.0, seed=42, strength=0.75,
            cfg_scale=7.0, distilled_cfg_scale=7.0, noise_level=0.0, prompt=""
        )
        frame_state = FrameState(metadata=metadata)
        
        # Test that we can't modify the frame state directly
        with self.assertRaises(AttributeError):
            frame_state.stage = ProcessingStage.GENERATION
    
    def test_frame_state_with_methods(self):
        """Test FrameState with_* methods for functional updates."""
        metadata = FrameMetadata(
            frame_idx=0, timestamp=0.0, seed=42, strength=0.75,
            cfg_scale=7.0, distilled_cfg_scale=7.0, noise_level=0.0, prompt=""
        )
        frame_state = FrameState(metadata=metadata)
        
        # Test with_stage
        new_state = frame_state.with_stage(ProcessingStage.GENERATION)
        self.assertEqual(new_state.stage, ProcessingStage.GENERATION)
        self.assertEqual(frame_state.stage, ProcessingStage.INITIALIZATION)  # Original unchanged
        
        # Test with_transformation
        transformed_state = frame_state.with_transformation("test_transform")
        self.assertIn("test_transform", transformed_state.transformations_applied)
        self.assertEqual(len(frame_state.transformations_applied), 0)  # Original unchanged
    
    def test_render_context_validation(self):
        """Test RenderContext validation."""
        # Valid context
        context = RenderContext(
            output_dir=Path("/tmp"),
            timestring="test",
            width=512,
            height=512,
            max_frames=10,
            fps=30.0,
            animation_mode="2D",
            use_depth_warping=False,
            save_depth_maps=False,
            hybrid_composite="None",
            hybrid_motion="None",
            depth_algorithm="midas",
            optical_flow_cadence="None",
            optical_flow_redo_generation="None",
            use_looper=False,
            diffusion_cadence=1
        )
        self.assertEqual(context.width, 512)
        
        # Invalid dimensions
        with self.assertRaises(ValueError):
            RenderContext(
                output_dir=Path("/tmp"),
                timestring="test",
                width=0,  # Invalid
                height=512,
                max_frames=10,
                fps=30.0,
                animation_mode="2D",
                use_depth_warping=False,
                save_depth_maps=False,
                hybrid_composite="None",
                hybrid_motion="None",
                depth_algorithm="midas",
                optical_flow_cadence="None",
                optical_flow_redo_generation="None",
                use_looper=False,
                diffusion_cadence=1
            )


@unittest.skipUnless(IMPORTS_AVAILABLE, "Functional rendering modules not available")
class TestFrameProcessing(unittest.TestCase):
    """Test pure frame processing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.context = RenderContext(
            output_dir=Path("/tmp"),
            timestring="test",
            width=512,
            height=512,
            max_frames=10,
            fps=30.0,
            animation_mode="2D",
            use_depth_warping=False,
            save_depth_maps=False,
            hybrid_composite="None",
            hybrid_motion="None",
            depth_algorithm="midas",
            optical_flow_cadence="None",
            optical_flow_redo_generation="None",
            use_looper=False,
            diffusion_cadence=1
        )
    
    def test_create_frame_state(self):
        """Test frame state creation."""
        frame_state = create_frame_state(0, self.context)
        
        self.assertEqual(frame_state.metadata.frame_idx, 0)
        self.assertEqual(frame_state.stage, ProcessingStage.INITIALIZATION)
        self.assertEqual(frame_state.metadata.timestamp, 0.0)
    
    def test_create_frame_state_with_overrides(self):
        """Test frame state creation with metadata overrides."""
        overrides = {
            'seed': 123,
            'strength': 0.8,
            'prompt': 'custom prompt'
        }
        frame_state = create_frame_state(5, self.context, overrides)
        
        self.assertEqual(frame_state.metadata.frame_idx, 5)
        self.assertEqual(frame_state.metadata.seed, 123)
        self.assertEqual(frame_state.metadata.strength, 0.8)
        self.assertEqual(frame_state.metadata.prompt, 'custom prompt')
    
    def test_validate_frame_state(self):
        """Test frame state validation."""
        # Valid frame state
        frame_state = create_frame_state(0, self.context)
        result = validate_frame_state(frame_state)
        self.assertTrue(result)
        
        # Invalid frame state (negative frame index)
        invalid_metadata = FrameMetadata(
            frame_idx=-1, timestamp=0.0, seed=42, strength=0.75,
            cfg_scale=7.0, distilled_cfg_scale=7.0, noise_level=0.0, prompt=""
        )
        # This should raise during metadata creation, not validation
        with self.assertRaises(ValueError):
            FrameState(metadata=invalid_metadata)
    
    def test_apply_frame_transformations(self):
        """Test applying frame transformations."""
        frame_state = create_frame_state(0, self.context)
        
        # Add a dummy image
        dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
        frame_state = frame_state.with_image(dummy_image)
        
        # Apply transformations
        result = apply_frame_transformations(
            frame_state,
            self.context,
            ("noise", "color_correction")
        )
        
        self.assertTrue(result.success)
        self.assertIn("noise", result.frame_state.transformations_applied)
        self.assertIn("color_correction", result.frame_state.transformations_applied)
    
    def test_process_frame(self):
        """Test complete frame processing."""
        frame_state = create_frame_state(0, self.context)
        
        # Add a dummy image
        dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
        frame_state = frame_state.with_image(dummy_image)
        
        result = process_frame(frame_state, self.context)
        
        self.assertTrue(result.success)
        self.assertEqual(result.frame_state.stage, ProcessingStage.COMPLETED)
        self.assertGreater(result.processing_time, 0)
    
    def test_merge_frame_results(self):
        """Test merging multiple frame results."""
        # Create some test results
        results = []
        for i in range(3):
            frame_state = create_frame_state(i, self.context)
            result = FrameResult(
                frame_state=frame_state,
                success=i < 2,  # First two succeed, last one fails
                processing_time=0.1 * (i + 1)
            )
            results.append(result)
        
        merged = merge_frame_results(tuple(results))
        
        self.assertEqual(merged['total_frames'], 3)
        self.assertEqual(merged['successful_frames'], 2)
        self.assertEqual(merged['failed_frames'], 1)
        self.assertAlmostEqual(merged['total_processing_time'], 0.6, places=1)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Functional rendering modules not available")
class TestRenderingPipeline(unittest.TestCase):
    """Test functional rendering pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.context = RenderContext(
            output_dir=Path("/tmp"),
            timestring="test",
            width=64,  # Small for testing
            height=64,
            max_frames=3,
            fps=30.0,
            animation_mode="2D",
            use_depth_warping=False,
            save_depth_maps=False,
            hybrid_composite="None",
            hybrid_motion="None",
            depth_algorithm="midas",
            optical_flow_cadence="None",
            optical_flow_redo_generation="None",
            use_looper=False,
            diffusion_cadence=1
        )
    
    def test_create_rendering_pipeline(self):
        """Test pipeline creation."""
        pipeline = create_rendering_pipeline(self.context)
        
        self.assertIsInstance(pipeline, RenderingPipeline)
        self.assertGreater(len(pipeline.stages), 0)
        self.assertIsInstance(pipeline.config, PipelineConfig)
    
    def test_pipeline_with_custom_config(self):
        """Test pipeline with custom configuration."""
        config = PipelineConfig(
            max_workers=2,
            enable_progress_tracking=False,
            enable_error_recovery=False
        )
        pipeline = create_rendering_pipeline(self.context, config=config)
        
        self.assertEqual(pipeline.config.max_workers, 2)
        self.assertFalse(pipeline.config.enable_progress_tracking)
    
    def test_execute_pipeline_sequential(self):
        """Test sequential pipeline execution."""
        pipeline = create_rendering_pipeline(self.context)
        
        # Create test frame states
        frame_states = []
        for i in range(2):
            frame_state = create_frame_state(i, self.context)
            # Add dummy image
            dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
            frame_state = frame_state.with_image(dummy_image)
            frame_states.append(frame_state)
        
        results = execute_pipeline(pipeline, frame_states)
        
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertTrue(result.success)
    
    def test_render_animation_functional(self):
        """Test functional animation rendering."""
        session = render_animation_functional(self.context)
        
        self.assertIsInstance(session, RenderingSession)
        self.assertEqual(session.context, self.context)
        self.assertEqual(len(session.frame_results), self.context.max_frames)
    
    def test_progress_tracker(self):
        """Test progress tracking callback."""
        progress_calls = []
        
        def test_callback(completed, total, result):
            progress_calls.append((completed, total, result.frame_idx))
        
        # Create a simple pipeline and execute it
        pipeline = create_rendering_pipeline(self.context)
        frame_states = [create_frame_state(0, self.context)]
        
        execute_pipeline(pipeline, frame_states, test_callback)
        
        self.assertEqual(len(progress_calls), 1)
        self.assertEqual(progress_calls[0][:2], (1, 1))


@unittest.skipUnless(IMPORTS_AVAILABLE, "Functional rendering modules not available")
class TestLegacyCompatibility(unittest.TestCase):
    """Test legacy compatibility adapter."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock legacy objects
        self.mock_args = Mock()
        self.mock_args.outdir = "/tmp/test"
        self.mock_args.W = 512
        self.mock_args.H = 512
        self.mock_args.seed = 42
        self.mock_args.strength = 0.75
        self.mock_args.cfg_scale = 7.0
        self.mock_args.distilled_cfg_scale = 7.0
        self.mock_args.prompt = "test prompt"
        self.mock_args.motion_preview_mode = False
        
        self.mock_anim_args = Mock()
        self.mock_anim_args.animation_mode = "2D"
        self.mock_anim_args.max_frames = 5
        self.mock_anim_args.use_depth_warping = False
        self.mock_anim_args.save_depth_maps = False
        self.mock_anim_args.hybrid_composite = "None"
        self.mock_anim_args.hybrid_motion = "None"
        self.mock_anim_args.depth_algorithm = "midas"
        self.mock_anim_args.optical_flow_cadence = "None"
        self.mock_anim_args.diffusion_cadence = 1
        
        self.mock_video_args = Mock()
        self.mock_video_args.fps = 30.0
        
        self.mock_root = Mock()
        self.mock_root.timestring = "test_20231201_120000"
        self.mock_root.device = "cuda"
        self.mock_root.half_precision = True
    
    def test_convert_legacy_args_to_context(self):
        """Test conversion of legacy arguments to RenderContext."""
        context = convert_legacy_args_to_context(
            self.mock_args,
            self.mock_anim_args,
            self.mock_video_args,
            self.mock_root
        )
        
        self.assertIsInstance(context, RenderContext)
        self.assertEqual(context.width, 512)
        self.assertEqual(context.height, 512)
        self.assertEqual(context.max_frames, 5)
        self.assertEqual(context.fps, 30.0)
        self.assertEqual(context.animation_mode, "2D")
        self.assertEqual(context.timestring, "test_20231201_120000")
    
    def test_validate_legacy_compatibility(self):
        """Test legacy compatibility validation."""
        # Valid arguments
        is_compatible, error = validate_legacy_compatibility(
            self.mock_args, self.mock_anim_args
        )
        self.assertTrue(is_compatible)
        self.assertEqual(error, "")
        
        # Missing required attribute
        delattr(self.mock_args, 'outdir')
        is_compatible, error = validate_legacy_compatibility(
            self.mock_args, self.mock_anim_args
        )
        self.assertFalse(is_compatible)
        self.assertIn("Missing required attribute", error)
    
    def test_enable_functional_rendering(self):
        """Test enabling/disabling functional rendering."""
        # Test enabling
        enable_functional_rendering(True)
        from scripts.deforum_helpers.rendering.legacy_renderer import is_functional_rendering_enabled
        self.assertTrue(is_functional_rendering_enabled())
        
        # Test disabling
        enable_functional_rendering(False)
        self.assertFalse(is_functional_rendering_enabled())
    
    @patch('scripts.deforum_helpers.rendering.legacy_renderer.render_animation_functional')
    def test_functional_render_animation_enabled(self, mock_render):
        """Test functional render animation when enabled."""
        enable_functional_rendering(True)
        
        # Mock the functional rendering
        mock_session = Mock()
        mock_session.completed_frames = 5
        mock_session.failed_frames = 0
        mock_session.total_processing_time = 10.0
        mock_session.context.max_frames = 5
        mock_session.frame_results = []
        mock_render.return_value = mock_session
        
        # Call functional render animation
        functional_render_animation(
            self.mock_args, self.mock_anim_args, self.mock_video_args,
            Mock(), Mock(), Mock(), Mock(), Mock(), self.mock_root
        )
        
        # Verify functional rendering was called
        mock_render.assert_called_once()


@unittest.skipUnless(IMPORTS_AVAILABLE, "Functional rendering modules not available")
class TestErrorHandling(unittest.TestCase):
    """Test error handling in functional rendering."""
    
    def test_rendering_error_creation(self):
        """Test RenderingError creation."""
        error = RenderingError(
            message="Test error",
            stage=ProcessingStage.GENERATION,
            frame_idx=5,
            context={'test': 'data'}
        )
        
        self.assertEqual(error.message, "Test error")
        self.assertEqual(error.stage, ProcessingStage.GENERATION)
        self.assertEqual(error.frame_idx, 5)
        self.assertEqual(error.context['test'], 'data')
        self.assertIn("Frame 5", str(error))
    
    def test_frame_result_with_error(self):
        """Test FrameResult with error."""
        metadata = FrameMetadata(
            frame_idx=0, timestamp=0.0, seed=42, strength=0.75,
            cfg_scale=7.0, distilled_cfg_scale=7.0, noise_level=0.0, prompt=""
        )
        frame_state = FrameState(metadata=metadata)
        
        error = RenderingError(
            "Test error", ProcessingStage.GENERATION, 0
        )
        
        result = FrameResult(
            frame_state=frame_state,
            success=False,
            error=error
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error, error)
    
    def test_frame_result_with_warnings(self):
        """Test FrameResult with warnings."""
        metadata = FrameMetadata(
            frame_idx=0, timestamp=0.0, seed=42, strength=0.75,
            cfg_scale=7.0, distilled_cfg_scale=7.0, noise_level=0.0, prompt=""
        )
        frame_state = FrameState(metadata=metadata)
        
        result = FrameResult(
            frame_state=frame_state,
            success=True
        )
        
        # Add warnings
        result_with_warning = result.with_warning("Test warning")
        
        self.assertEqual(len(result.warnings), 0)  # Original unchanged
        self.assertEqual(len(result_with_warning.warnings), 1)
        self.assertIn("Test warning", result_with_warning.warnings)


@unittest.skipUnless(IMPORTS_AVAILABLE, "Functional rendering modules not available")
class TestFunctionalProgrammingPrinciples(unittest.TestCase):
    """Test that functional programming principles are maintained."""
    
    def test_immutability_preservation(self):
        """Test that all data structures remain immutable."""
        metadata = FrameMetadata(
            frame_idx=0, timestamp=0.0, seed=42, strength=0.75,
            cfg_scale=7.0, distilled_cfg_scale=7.0, noise_level=0.0, prompt=""
        )
        frame_state = FrameState(metadata=metadata)
        
        # Test that modifications create new objects
        new_state = frame_state.with_stage(ProcessingStage.GENERATION)
        self.assertNotEqual(id(frame_state), id(new_state))
        self.assertEqual(frame_state.stage, ProcessingStage.INITIALIZATION)
        self.assertEqual(new_state.stage, ProcessingStage.GENERATION)
    
    def test_pure_function_behavior(self):
        """Test that processing functions are pure (no side effects)."""
        context = RenderContext(
            output_dir=Path("/tmp"),
            timestring="test",
            width=64,
            height=64,
            max_frames=1,
            fps=30.0,
            animation_mode="2D",
            use_depth_warping=False,
            save_depth_maps=False,
            hybrid_composite="None",
            hybrid_motion="None",
            depth_algorithm="midas",
            optical_flow_cadence="None",
            optical_flow_redo_generation="None",
            use_looper=False,
            diffusion_cadence=1
        )
        
        frame_state = create_frame_state(0, context)
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
        frame_state = frame_state.with_image(dummy_image)
        
        # Call processing function multiple times
        result1 = process_frame(frame_state, context)
        result2 = process_frame(frame_state, context)
        
        # Results should be consistent (same input -> same output)
        self.assertEqual(result1.success, result2.success)
        self.assertEqual(result1.frame_state.stage, result2.frame_state.stage)
        
        # Original frame state should be unchanged
        self.assertEqual(frame_state.stage, ProcessingStage.INITIALIZATION)
    
    def test_functional_composition(self):
        """Test that functions can be composed functionally."""
        context = RenderContext(
            output_dir=Path("/tmp"),
            timestring="test",
            width=64,
            height=64,
            max_frames=1,
            fps=30.0,
            animation_mode="2D",
            use_depth_warping=False,
            save_depth_maps=False,
            hybrid_composite="None",
            hybrid_motion="None",
            depth_algorithm="midas",
            optical_flow_cadence="None",
            optical_flow_redo_generation="None",
            use_looper=False,
            diffusion_cadence=1
        )
        
        # Create a chain of transformations
        frame_state = create_frame_state(0, context)
        dummy_image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Compose transformations functionally
        result = (frame_state
                 .with_image(dummy_image)
                 .with_stage(ProcessingStage.PRE_PROCESSING)
                 .with_transformation("test1")
                 .with_transformation("test2"))
        
        self.assertEqual(result.stage, ProcessingStage.PRE_PROCESSING)
        self.assertEqual(len(result.transformations_applied), 2)
        self.assertIn("test1", result.transformations_applied)
        self.assertIn("test2", result.transformations_applied)
        
        # Original should be unchanged
        self.assertEqual(frame_state.stage, ProcessingStage.INITIALIZATION)
        self.assertEqual(len(frame_state.transformations_applied), 0)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 