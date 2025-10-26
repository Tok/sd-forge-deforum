"""Unit tests for deforum.utils.ui.builders module.

Tests pure UI builder functions that create Gradio components.
These tests verify the logic without requiring full Gradio runtime.
"""

import pytest
from unittest.mock import Mock, patch
from deforum.utils.ui.builders import (
    create_gr_elem,
    is_gradio_component,
)


# ============================================================================
# Gradio Element Creation
# ============================================================================

class TestCreateGrElem:
    """Test Gradio element creation from dict specifications."""

    @patch('deforum.utils.ui.builders.gr')
    def test_simple_textbox(self, mock_gr):
        """Test creating a simple textbox component."""
        mock_textbox = Mock()
        mock_gr.Textbox = mock_textbox

        spec = {
            "type": "textbox",
            "label": "Test Label",
            "value": "default"
        }

        create_gr_elem(spec)

        mock_textbox.assert_called_once_with(
            label="Test Label",
            value="default"
        )

    @patch('deforum.utils.ui.builders.gr')
    def test_checkbox_group_camelcase(self, mock_gr):
        """Test that checkbox_group becomes CheckboxGroup."""
        mock_checkbox_group = Mock()
        mock_gr.CheckboxGroup = mock_checkbox_group

        spec = {
            "type": "checkbox_group",
            "choices": ["A", "B"]
        }

        create_gr_elem(spec)

        mock_checkbox_group.assert_called_once_with(choices=["A", "B"])

    @patch('deforum.utils.ui.builders.gr')
    def test_none_values_filtered(self, mock_gr):
        """Test that None values are filtered from parameters."""
        mock_slider = Mock()
        mock_gr.Slider = mock_slider

        spec = {
            "type": "slider",
            "minimum": 0,
            "maximum": 100,
            "value": None  # Should be filtered out
        }

        create_gr_elem(spec)

        mock_slider.assert_called_once_with(minimum=0, maximum=100)

    @patch('deforum.utils.ui.builders.gr')
    def test_type_param_handling(self, mock_gr):
        """Test that type_param is renamed to type."""
        mock_radio = Mock()
        mock_gr.Radio = mock_radio

        spec = {
            "type": "radio",
            "type_param": "index",  # Should become 'type' param
            "choices": ["Option 1", "Option 2"]
        }

        create_gr_elem(spec)

        mock_radio.assert_called_once_with(
            type="index",
            choices=["Option 1", "Option 2"]
        )


# ============================================================================
# Component Type Checking
# ============================================================================

class TestIsGradioComponent:
    """Test Gradio component type checking."""

    @patch('deforum.utils.ui.builders.gr')
    def test_button_is_component(self, mock_gr):
        """Test that gr.Button is recognized as component."""
        mock_button = Mock()
        mock_gr.Button = type('Button', (), {})

        button_instance = mock_gr.Button()
        result = is_gradio_component(button_instance)

        assert result is True

    def test_textbox_is_component_integration(self):
        """Test that real gr.Textbox is recognized as component."""
        try:
            import gradio as gr
            textbox_instance = gr.Textbox()
            result = is_gradio_component(textbox_instance)
            assert result is True
        except ImportError:
            pytest.skip("Gradio not available for integration test")

    def test_string_not_component(self):
        """Test that string is not recognized as component."""
        result = is_gradio_component("not a component")
        assert result is False

    def test_dict_not_component(self):
        """Test that dict is not recognized as component."""
        result = is_gradio_component({"type": "textbox"})
        assert result is False

    def test_none_not_component(self):
        """Test that None is not recognized as component."""
        result = is_gradio_component(None)
        assert result is False


# ============================================================================
# Integration Tests (if Gradio is available)
# ============================================================================

class TestGradioIntegration:
    """Integration tests that require actual Gradio components."""

    def test_real_gradio_import(self):
        """Test that we can import actual Gradio."""
        try:
            import gradio as gr
            assert gr is not None
        except ImportError:
            pytest.skip("Gradio not available for integration test")

    def test_create_real_textbox(self):
        """Test creating a real Gradio textbox (if available)."""
        try:
            import gradio as gr
            from deforum.utils.ui.builders import create_gr_elem

            spec = {
                "type": "textbox",
                "label": "Test",
                "value": "test value"
            }

            result = create_gr_elem(spec)

            # Verify it's a real Textbox
            assert isinstance(result, gr.Textbox)

        except ImportError:
            pytest.skip("Gradio not available for integration test")
