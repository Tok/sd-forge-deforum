"""Integration tests for camera path presets.

These tests require full Forge WebUI environment including backend modules.
Run with: pytest tests/integration/ (from Forge WebUI root)
"""

import pytest
import sys
import importlib.util
from pathlib import Path

pytestmark = pytest.mark.integration  # Mark entire module as integration tests


def _import_camera_path_generator():
    """Import camera_path_generator directly without triggering deforum.ui package."""
    module_path = Path(__file__).parent.parent.parent / "deforum" / "ui" / "handlers" / "camera_path_generator.py"
    spec = importlib.util.spec_from_file_location("camera_path_generator", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPresetIntegration:
    """Test preset generation from handlers (requires Forge environment)."""

    def test_all_presets_generate_valid_paths(self):
        """Test that all preset types generate valid camera paths."""
        camera_gen = _import_camera_path_generator()
        generate_preset_path = camera_gen.generate_preset_path

        presets = [
            "rotate-around",
            "figure-eight",
            "forward-zoom",
            "orbit-up",
            "spiral",
            "street",
            "dashcam",
            "bodycam"
        ]

        for preset_type in presets:
            status, schedules, camera_path = generate_preset_path(
                preset_type=preset_type,
                radius=100.0,
                height=10.0,
                rotation_factor=-8.0,
                num_frames=50,  # Small number for fast tests
                closed_loop=False
            )

            # Check that generation succeeded (✓ is from MockEmojiUtils in conftest)
            assert ("✅" in status or "✓" in status), f"Preset {preset_type} failed: {status}"
            assert len(camera_path) == 50, f"Preset {preset_type} wrong length"
            assert len(schedules) == 6, f"Preset {preset_type} missing schedules"

            # Check that all 6 schedules have content
            for key in ['translation_x', 'translation_y', 'translation_z',
                       'rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z']:
                assert key in schedules, f"Preset {preset_type} missing {key}"
                assert schedules[key], f"Preset {preset_type} has empty {key}"

    def test_forward_facing_presets_have_zero_rotation(self):
        """Test that street/dashcam/bodycam presets have zero rotation."""
        camera_gen = _import_camera_path_generator()
        generate_preset_path = camera_gen.generate_preset_path

        forward_presets = ["street", "dashcam", "bodycam"]

        for preset_type in forward_presets:
            status, schedules, camera_path = generate_preset_path(
                preset_type=preset_type,
                radius=100.0,
                height=10.0,
                rotation_factor=0.0,  # Not used for these presets
                num_frames=50,
                closed_loop=False
            )

            # All rotation schedules should be "0: (0.00), 1: (0.00), ..."
            for key in ['rotation_3d_x', 'rotation_3d_y', 'rotation_3d_z']:
                schedule_str = schedules[key]
                # Parse values - all should be 0.00
                import re
                values = [float(m) for m in re.findall(r':\s*\((-?\d+\.?\d*)\)', schedule_str)]
                for val in values:
                    assert abs(val) < 0.01, f"Preset {preset_type} has non-zero {key}"

    def test_orbit_up_cannot_be_closed(self):
        """Test that orbit-up preset always uses closed_loop=False."""
        camera_gen = _import_camera_path_generator()
        generate_preset_path = camera_gen.generate_preset_path

        status, schedules, camera_path = generate_preset_path(
            preset_type="orbit-up",
            radius=100.0,
            height=0.0,
            rotation_factor=0.0,
            num_frames=50,
            closed_loop=True  # Try to request closed loop
        )

        # Should mention that orbit-up cannot be closed
        assert "cannot be closed" in status or "Note:" in status

    def test_rotate_around_uses_sphere(self):
        """Test that rotate-around preset uses sphere rotation by default."""
        camera_gen = _import_camera_path_generator()
        generate_preset_path = camera_gen.generate_preset_path

        status, schedules, camera_path = generate_preset_path(
            preset_type="rotate-around",
            radius=100.0,
            height=0.0,
            rotation_factor=-5.0,
            num_frames=100,
            closed_loop=False
        )

        # Status should mention sphere rotation
        assert "sphere" in status.lower() or "3D" in status

        # Y values should vary (not flat)
        y_values = [p.y for p in camera_path]
        y_range = max(y_values) - min(y_values)
        assert y_range > 5.0, "Rotate-around should use sphere (Y variation expected)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
