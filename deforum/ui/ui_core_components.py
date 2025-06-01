"""
UI Core Components Module

Core utilities for creating and managing Gradio UI components in Deforum.
This module provides the fundamental building blocks used across all UI tabs.

Functions:
    - create_gr_elem: Creates Gradio elements from configuration dictionaries
    - is_gradio_component: Checks if an object is a Gradio component
    - create_row: Creates FormRow layouts with components
    - Component validation and helper functions
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn


def create_gr_elem(d):
    """
    Create a Gradio element from a configuration dictionary.
    
    Capitalizes and CamelCases the 'type' value to match Gradio component names.
    Examples: "dropdown" becomes gr.Dropdown, "checkbox_group" becomes gr.CheckboxGroup.
    
    Args:
        d (dict): Configuration dictionary with 'type' and other parameters
        
    Returns:
        gr.Component: The created Gradio component
    """
    # Capitalize and CamelCase the orig value under "type", which defines gr.inputs.type in lower_case.
    obj_type_str = ''.join(word.title() for word in d["type"].split('_'))
    obj_type = getattr(gr, obj_type_str)

    # Prepare parameters for gradio element creation
    params = {k: v for k, v in d.items() if k != "type" and v is not None}

    # Special case: Since some elements can have 'type' parameter and we are already using 'type' to specify
    # which element to use we need a separate parameter that will be used to overwrite 'type' at this point.
    # E.g. for Radio element we should specify 'type_param' which is then used to set gr.radio's type.
    if 'type_param' in params:
        params['type'] = params.pop('type_param')

    return obj_type(**params)


def is_gradio_component(args):
    """
    Check if an object is a Gradio component.
    
    Args:
        args: Object to check
        
    Returns:
        bool: True if the object is a Gradio component
    """
    return isinstance(args, (gr.Button, gr.Textbox, gr.Slider, gr.Dropdown,
                             gr.HTML, gr.Radio, gr.Interface, gr.Markdown,
                             gr.Checkbox))


def create_row(args, *attrs):
    """
    Create a FormRow with components.
    
    If attrs are provided, create components from the attributes of args.
    Otherwise, pass through a single component or create one.
    
    Args:
        args: Component configuration or object
        *attrs: Attribute names to extract from args
        
    Returns:
        list or component: List of components if attrs provided, single component otherwise
    """
    with FormRow():
        return [create_gr_elem(getattr(args, attr)) for attr in attrs] if attrs \
            else args if is_gradio_component(args) else create_gr_elem(args)


def create_two_column_row(left_components, right_components, left_scale=1, right_scale=1):
    """
    Create a row with two columns containing components.
    
    Args:
        left_components (list): Components for the left column
        right_components (list): Components for the right column
        left_scale (int): Scale factor for left column
        right_scale (int): Scale factor for right column
        
    Returns:
        tuple: (left_column_components, right_column_components)
    """
    with FormRow():
        with FormColumn(scale=left_scale):
            left_result = [comp if is_gradio_component(comp) else create_gr_elem(comp) 
                          for comp in left_components]
        with FormColumn(scale=right_scale):
            right_result = [comp if is_gradio_component(comp) else create_gr_elem(comp) 
                           for comp in right_components]
    
    return left_result, right_result


def create_compact_row(*components):
    """
    Create a compact FormRow with multiple components.
    
    Args:
        *components: Components or configuration dictionaries
        
    Returns:
        list: Created components
    """
    with FormRow(variant="compact"):
        return [comp if is_gradio_component(comp) else create_gr_elem(comp) 
                for comp in components]


def create_accordion_with_info(title, content, open_state=False):
    """
    Create an accordion with informational content.
    
    Args:
        title (str): Accordion title
        content (str): HTML content to display
        open_state (bool): Whether accordion is initially open
        
    Returns:
        gr.Accordion: The created accordion
    """
    with gr.Accordion(title, open=open_state) as accordion:
        gr.HTML(value=content)
    return accordion


def validate_component_config(config):
    """
    Validate a component configuration dictionary.
    
    Args:
        config (dict): Component configuration
        
    Returns:
        bool: True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError("Component configuration must be a dictionary")
    
    if 'type' not in config:
        raise ValueError("Component configuration must have a 'type' field")
    
    # Check if the type exists in Gradio
    obj_type_str = ''.join(word.title() for word in config["type"].split('_'))
    if not hasattr(gr, obj_type_str):
        raise ValueError(f"Unknown Gradio component type: {config['type']}")
    
    return True


def get_component_by_type(component_type):
    """
    Get a Gradio component class by type string.
    
    Args:
        component_type (str): Component type (e.g., 'textbox', 'dropdown')
        
    Returns:
        class: Gradio component class
    """
    obj_type_str = ''.join(word.title() for word in component_type.split('_'))
    return getattr(gr, obj_type_str)


# Component creation helpers for common patterns
def create_textbox(label, value="", lines=1, **kwargs):
    """Create a standard textbox with common defaults."""
    return gr.Textbox(label=label, value=value, lines=lines, interactive=True, **kwargs)


def create_dropdown(label, choices, value=None, **kwargs):
    """Create a standard dropdown with common defaults."""
    return gr.Dropdown(label=label, choices=choices, value=value, interactive=True, **kwargs)


def create_slider(label, minimum, maximum, value, step=1, **kwargs):
    """Create a standard slider with common defaults."""
    return gr.Slider(label=label, minimum=minimum, maximum=maximum, 
                    value=value, step=step, interactive=True, **kwargs)


def create_checkbox(label, value=False, **kwargs):
    """Create a standard checkbox with common defaults."""
    return gr.Checkbox(label=label, value=value, interactive=True, **kwargs)


def create_button(label, variant="primary", **kwargs):
    """Create a standard button with common defaults."""
    return gr.Button(label, variant=variant, **kwargs) 