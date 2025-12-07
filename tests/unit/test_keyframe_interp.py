"""Unit tests for keyframe interpolation rendering functions.

Tests the pure functions from deforum/rendering/keyframe_interp.py.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch
from PIL import Image


class TestModelDetection:
    """Test diffusion model detection from checkpoint names."""

    def test_detect_flux_model(self):
        """Should detect Flux from checkpoint name."""
        from deforum.rendering.keyframe_interp import detect_model_prefix

        assert detect_model_prefix("flux1-dev-bnb-nf4-v2.safetensors") == "flux"
        assert detect_model_prefix("FLUX-schnell.safetensors") == "flux"
        assert detect_model_prefix("some_flux_model.ckpt") == "flux"

    def test_detect_zit_model(self):
        """Should detect Z-Image-Turbo from checkpoint name."""
        from deforum.rendering.keyframe_interp import detect_model_prefix

        assert detect_model_prefix("z-image-turbo.safetensors") == "zit"
        assert detect_model_prefix("zimage_model.safetensors") == "zit"
        assert detect_model_prefix("ZIT-v1.safetensors") == "zit"

    def test_detect_lumina_model(self):
        """Should detect Lumina from checkpoint name."""
        from deforum.rendering.keyframe_interp import detect_model_prefix

        assert detect_model_prefix("lumina-v2.safetensors") == "lumina"
        assert detect_model_prefix("LUMINA_anime.safetensors") == "lumina"

    def test_fallback_to_diffusion(self):
        """Should default to 'diffusion' for unknown models."""
        from deforum.rendering.keyframe_interp import detect_model_prefix

        assert detect_model_prefix("unknown_model.safetensors") == "diffusion"
        assert detect_model_prefix("sd15.ckpt") == "diffusion"
        assert detect_model_prefix("") == "diffusion"


class TestOutputFilename:
    """Test output video filename building."""

    def test_build_filename_with_flux_wan(self):
        """Should build correct filename for Flux + Wan."""
        from deforum.rendering.keyframe_interp import build_output_filename

        filename = build_output_filename("20241207_120000", "flux", "Wan")

        assert filename == "20241207_120000_flux_wan.mp4"

    def test_build_filename_with_zit_da3(self):
        """Should build correct filename for ZIT + DA3-3DGS."""
        from deforum.rendering.keyframe_interp import build_output_filename

        filename = build_output_filename("20241207_120000", "zit", "DA3-3DGS")

        assert filename == "20241207_120000_zit_da3-3dgs.mp4"

    def test_build_filename_with_lumina_film(self):
        """Should build correct filename for Lumina + FILM."""
        from deforum.rendering.keyframe_interp import build_output_filename

        filename = build_output_filename("20241207_120000", "lumina", "FILM")

        assert filename == "20241207_120000_lumina_film.mp4"

    def test_preserves_timestring(self):
        """Should preserve exact timestring format."""
        from deforum.rendering.keyframe_interp import build_output_filename

        custom_time = "custom_timestamp_12345"
        filename = build_output_filename(custom_time, "flux", "Wan")

        assert filename.startswith(custom_time)


class TestSaveKeyframe:
    """Test keyframe saving with different modes."""

    def test_saves_to_root_directory_by_default(self):
        """Should save to root output directory by default."""
        from deforum.rendering.keyframe_interp import save_keyframe
        from deforum.rendering.data.render_data import RenderData
        from deforum.rendering.data.frame.diffusion_frame import DiffusionFrame

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock data
            mock_data = Mock(spec=RenderData)
            mock_data.output_directory = tmpdir

            mock_frame = Mock(spec=DiffusionFrame)
            mock_frame.i = 123

            # Create test image
            test_image = Image.new('RGB', (10, 10), (255, 0, 0))

            # Save keyframe
            filepath = save_keyframe(mock_data, mock_frame, test_image, use_diffusion_subdir=False)

            # Check saved to root
            expected_path = os.path.join(tmpdir, "000000123.png")
            assert filepath == expected_path
            assert os.path.exists(expected_path)

    def test_saves_to_diffusion_subdirectory_when_requested(self):
        """Should save to _diffusion/ subdirectory for DA3-3DGS mode."""
        from deforum.rendering.keyframe_interp import save_keyframe
        from deforum.rendering.data.render_data import RenderData
        from deforum.rendering.data.frame.diffusion_frame import DiffusionFrame

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock data
            mock_data = Mock(spec=RenderData)
            mock_data.output_directory = tmpdir

            mock_frame = Mock(spec=DiffusionFrame)
            mock_frame.i = 456

            # Create test image
            test_image = Image.new('RGB', (10, 10), (0, 255, 0))

            # Save keyframe with diffusion subdir
            filepath = save_keyframe(mock_data, mock_frame, test_image, use_diffusion_subdir=True)

            # Check saved to _diffusion/
            expected_path = os.path.join(tmpdir, "_diffusion", "000000456.png")
            assert filepath == expected_path
            assert os.path.exists(expected_path)

    def test_creates_diffusion_subdirectory_if_missing(self):
        """Should create _diffusion/ directory if it doesn't exist."""
        from deforum.rendering.keyframe_interp import save_keyframe
        from deforum.rendering.data.render_data import RenderData
        from deforum.rendering.data.frame.diffusion_frame import DiffusionFrame

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_data = Mock(spec=RenderData)
            mock_data.output_directory = tmpdir

            mock_frame = Mock(spec=DiffusionFrame)
            mock_frame.i = 789

            test_image = Image.new('RGB', (10, 10), (0, 0, 255))

            # Ensure _diffusion/ doesn't exist
            diffusion_dir = os.path.join(tmpdir, "_diffusion")
            assert not os.path.exists(diffusion_dir)

            # Save keyframe
            filepath = save_keyframe(mock_data, mock_frame, test_image, use_diffusion_subdir=True)

            # Check directory was created
            assert os.path.exists(diffusion_dir)
            assert os.path.isdir(diffusion_dir)
            assert os.path.exists(filepath)

    def test_filename_format_uses_nine_digits(self):
        """Should format frame index with 9 digits zero-padded."""
        from deforum.rendering.keyframe_interp import save_keyframe
        from deforum.rendering.data.render_data import RenderData
        from deforum.rendering.data.frame.diffusion_frame import DiffusionFrame

        with tempfile.TemporaryDirectory() as tmpdir:
            mock_data = Mock(spec=RenderData)
            mock_data.output_directory = tmpdir

            # Test various frame indices
            for frame_idx in [0, 5, 99, 12345]:
                mock_frame = Mock(spec=DiffusionFrame)
                mock_frame.i = frame_idx

                test_image = Image.new('RGB', (10, 10))
                filepath = save_keyframe(mock_data, mock_frame, test_image)

                # Check filename format
                filename = os.path.basename(filepath)
                assert filename == f"{frame_idx:09d}.png"


class TestStitchKeyframeInterpolationVideo:
    """Test video stitching functionality."""

    def test_detects_model_from_checkpoint(self):
        """Should detect model prefix from checkpoint name."""
        from deforum.rendering.keyframe_interp import stitch_keyframe_interpolation_video
        from deforum.rendering.data.render_data import RenderData

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock data with nested args structure
            mock_args_inner = Mock()
            mock_args_inner.checkpoint = "flux1-dev-bnb-nf4-v2.safetensors"

            mock_args_outer = Mock()
            mock_args_outer.args = mock_args_inner

            mock_data = Mock(spec=RenderData)
            mock_data.output_directory = tmpdir
            mock_data.timestring = "20241207_120000"
            mock_data.args = mock_args_outer

            mock_video_args = Mock()
            mock_video_args.fps = 24
            mock_video_args.ffmpeg_crf = 17
            mock_video_args.ffmpeg_preset = 'slow'
            mock_video_args.add_soundtrack = 'None'
            mock_video_args.soundtrack_path = ''

            # Create dummy frame files
            frame_paths = []
            for i in range(3):
                frame_path = os.path.join(tmpdir, f"{i:09d}.png")
                Image.new('RGB', (100, 100)).save(frame_path)
                frame_paths.append(frame_path)

            with patch('deforum.rendering.keyframe_interp.ffmpeg_stitch_video') as mock_stitch:
                result = stitch_keyframe_interpolation_video(
                    mock_data,
                    frame_paths,
                    mock_video_args,
                    interp_method="Wan"
                )

                # Check that the result filename contains 'flux_wan'
                assert result is not None
                assert 'flux_wan' in result.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
