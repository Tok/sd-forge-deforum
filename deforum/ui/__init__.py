"""
User interface components and Gradio integration.
"""

# Essential exports - modules will be imported only when the extension actually loads
__all__ = [
    'on_ui_tabs',
    'on_ui_settings',
    'setup_deforum_left_side_ui',
]

# Only import the essential functions that are needed by the main extension entry point
def get_on_ui_tabs():
    """Get the on_ui_tabs function when needed."""
    from .secondary_interface_panels import on_ui_tabs
    return on_ui_tabs

def get_on_ui_settings():
    """Get the on_ui_settings function when needed."""
    from .settings_interface import on_ui_settings
    return on_ui_settings

def get_setup_deforum_left_side_ui():
    """Get the setup_deforum_left_side_ui function when needed."""
    from .main_interface_panels import setup_deforum_left_side_ui
    return setup_deforum_left_side_ui 