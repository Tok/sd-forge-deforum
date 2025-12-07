"""Unit tests for slopcore gradient system."""

import pytest


def get_current_gradient_colors():
    """Helper to get current gradient colors from themes module."""
    import deforum.utils.system.logging.themes as themes
    # Re-import to get latest values
    import importlib
    importlib.reload(themes)
    return themes.HEX_SLOPCORE_1, themes.HEX_SLOPCORE_7


class TestSlopcoreGradientSwitching:
    """Test dynamic gradient switching functionality."""

    def test_default_gradient_is_bb0(self):
        """Should default to BB0 gradient on module load."""
        from deforum.utils.system.logging.themes import set_slopcore_gradient, get_active_gradient
        # Reset to default
        set_slopcore_gradient('BB0')
        assert get_active_gradient() == 'BB0'

    def test_switch_to_da3_gradient(self):
        """Should successfully switch to DA3 gradient."""
        from deforum.utils.system.logging.themes import (
            set_slopcore_gradient, get_active_gradient, SLOPCORE_GRADIENTS
        )
        set_slopcore_gradient('DA3')
        assert get_active_gradient() == 'DA3'

        # Verify colors changed to DA3 values
        import deforum.utils.system.logging.themes as themes
        da3_first = SLOPCORE_GRADIENTS['DA3']['7_shade'][0]
        da3_last = SLOPCORE_GRADIENTS['DA3']['7_shade'][6]
        assert themes.HEX_SLOPCORE_1 == da3_first
        assert themes.HEX_SLOPCORE_7 == da3_last

    def test_switch_back_to_bb0_gradient(self):
        """Should successfully switch back to BB0 gradient."""
        from deforum.utils.system.logging.themes import (
            set_slopcore_gradient, get_active_gradient, SLOPCORE_GRADIENTS
        )
        # Switch to DA3 first
        set_slopcore_gradient('DA3')
        assert get_active_gradient() == 'DA3'

        # Switch back to BB0
        set_slopcore_gradient('BB0')
        assert get_active_gradient() == 'BB0'

        # Verify colors changed back to BB0 values
        import deforum.utils.system.logging.themes as themes
        bb0_first = SLOPCORE_GRADIENTS['BB0']['7_shade'][0]
        bb0_last = SLOPCORE_GRADIENTS['BB0']['7_shade'][6]
        assert themes.HEX_SLOPCORE_1 == bb0_first
        assert themes.HEX_SLOPCORE_7 == bb0_last

    def test_invalid_gradient_raises_error(self):
        """Should raise ValueError for unknown gradient."""
        from deforum.utils.system.logging.themes import set_slopcore_gradient
        with pytest.raises(ValueError) as exc_info:
            set_slopcore_gradient('INVALID')

        assert "Unknown gradient" in str(exc_info.value)
        assert "INVALID" in str(exc_info.value)


class TestGradientRegistry:
    """Test gradient metadata and registry."""

    def test_gradient_count(self):
        """Should have exactly 2 gradients (BB0 and DA3)."""
        from deforum.utils.system.logging.themes import SLOPCORE_GRADIENTS
        assert len(SLOPCORE_GRADIENTS) == 2
        assert 'BB0' in SLOPCORE_GRADIENTS
        assert 'DA3' in SLOPCORE_GRADIENTS

    def test_bb0_gradient_metadata(self):
        """Should have correct BB0 metadata."""
        from deforum.utils.system.logging.themes import SLOPCORE_GRADIENTS
        bb0 = SLOPCORE_GRADIENTS['BB0']
        assert 'description' in bb0
        assert 'reference' in bb0
        assert '7_shade' in bb0
        assert '5_tqdm' in bb0

        assert 'BLANK BANSHEE 0' in bb0['description']
        assert '#5606ff' in bb0['reference'].lower()
        assert '#17a7fe' in bb0['reference'].lower()

        # Verify 7-shade gradient
        assert len(bb0['7_shade']) == 7
        assert bb0['7_shade'][0] == '#5606FF'
        assert bb0['7_shade'][6] == '#17A7FE'

        # Verify 5-tqdm gradient
        assert len(bb0['5_tqdm']) == 5

    def test_da3_gradient_metadata(self):
        """Should have correct DA3 metadata."""
        from deforum.utils.system.logging.themes import SLOPCORE_GRADIENTS
        da3 = SLOPCORE_GRADIENTS['DA3']
        assert 'description' in da3
        assert 'reference' in da3
        assert '7_shade' in da3
        assert '5_tqdm' in da3

        assert 'Depth Anything V3' in da3['description']
        assert 'depth-anything-3.github.io' in da3['reference']
        assert '#667eea' in da3['reference'].lower()
        assert '#764ba2' in da3['reference'].lower()

        # Verify 7-shade gradient
        assert len(da3['7_shade']) == 7
        assert da3['7_shade'][0] == '#667EEA'
        assert da3['7_shade'][6] == '#764BA2'

        # Verify 5-tqdm gradient
        assert len(da3['5_tqdm']) == 5

    def test_list_gradients(self):
        """Should list all available gradients with metadata."""
        from deforum.utils.system.logging.themes import list_slopcore_gradients
        gradients = list_slopcore_gradients()

        assert len(gradients) == 2
        assert 'BB0' in gradients
        assert 'DA3' in gradients

        # Verify BB0 metadata
        assert 'description' in gradients['BB0']
        assert 'reference' in gradients['BB0']

        # Verify DA3 metadata
        assert 'description' in gradients['DA3']
        assert 'reference' in gradients['DA3']


class TestRandomGradientSelection:
    """Test random gradient selection."""

    def test_random_gradient_returns_valid_name(self):
        """Should return a valid gradient name."""
        from deforum.utils.system.logging.themes import get_random_slopcore_gradient
        for _ in range(10):  # Test multiple times
            gradient = get_random_slopcore_gradient()
            assert gradient in ['BB0', 'DA3']

    def test_random_gradient_can_return_both_variants(self):
        """Should be able to return both BB0 and DA3 over multiple calls."""
        from deforum.utils.system.logging.themes import get_random_slopcore_gradient
        # Run 100 times to statistically ensure both appear
        results = set()
        for _ in range(100):
            results.add(get_random_slopcore_gradient())

        # With 100 trials and 50/50 chance, extremely unlikely to not see both
        assert 'BB0' in results
        assert 'DA3' in results


class TestGradientColorValues:
    """Test actual hex color values."""

    def test_bb0_gradient_colors(self):
        """Should have correct BB0 hex colors."""
        from deforum.utils.system.logging.themes import SLOPCORE_GRADIENTS
        bb0 = SLOPCORE_GRADIENTS['BB0']['7_shade']
        assert bb0[0] == '#5606FF'
        assert bb0[1] == '#4C21FF'
        assert bb0[2] == '#413CFF'
        assert bb0[3] == '#3757FF'
        assert bb0[4] == '#2C71FE'
        assert bb0[5] == '#228CFE'
        assert bb0[6] == '#17A7FE'

    def test_da3_gradient_colors(self):
        """Should have correct DA3 hex colors."""
        from deforum.utils.system.logging.themes import SLOPCORE_GRADIENTS
        da3 = SLOPCORE_GRADIENTS['DA3']['7_shade']
        assert da3[0] == '#667EEA'
        assert da3[1] == '#6D74DE'
        assert da3[2] == '#746AD2'
        assert da3[3] == '#7B60C6'
        assert da3[4] == '#7F57BD'
        assert da3[5] == '#7B52AF'
        assert da3[6] == '#764BA2'

    def test_hex_colors_are_uppercase(self):
        """All hex colors should use uppercase."""
        from deforum.utils.system.logging.themes import SLOPCORE_GRADIENTS
        for gradient_data in SLOPCORE_GRADIENTS.values():
            for color in gradient_data['7_shade']:
                assert color == color.upper(), f"Color {color} should be uppercase"
            for color in gradient_data['5_tqdm']:
                assert color == color.upper(), f"Color {color} should be uppercase"
