"""
Pure Component Builder Functions
Contains utility functions for creating Gradio UI components
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn, ToolButton


def create_gr_elem(d):
    """Create a Gradio element from a dictionary configuration.
    
    Args:
        d: Dictionary containing element configuration with 'type' key
        
    Returns:
        Gradio component instance
    """
    # Capitalize and CamelCase the orig value under "type", which defines gr.inputs.type in lower_case.
    # Examples: "dropdown" becomes gr.Dropdown, and "checkbox_group" becomes gr.CheckboxGroup.
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
    """Check if an object is a Gradio component.
    
    Args:
        args: Object to check
        
    Returns:
        bool: True if object is a Gradio component
    """
    return isinstance(args, (gr.Button, gr.Textbox, gr.Slider, gr.Dropdown,
                             gr.HTML, gr.Radio, gr.Interface, gr.Markdown,
                             gr.Checkbox))


def create_row(args, *attrs):
    """Create a row of Gradio components.
    
    Args:
        args: Either a component configuration or args object
        *attrs: Attribute names to extract from args
        
    Returns:
        Component or list of components in a FormRow
    """
    # If attrs are provided, create components from the attributes of args.
    # Otherwise, pass through a single component or create one.
    with FormRow():
        return [create_gr_elem(getattr(args, attr)) for attr in attrs] if attrs \
            else args if is_gradio_component(args) else create_gr_elem(args)


def create_accordion_md_row(name, markdown, is_open=False):
    """Create an accordion with markdown content.
    
    Args:
        name: Accordion title
        markdown: Markdown content to display
        is_open: Whether accordion starts open
        
    Returns:
        Accordion component
    """
    with gr.Accordion(name, open=is_open):
        gr.Markdown(markdown) 