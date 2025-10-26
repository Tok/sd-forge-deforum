"""Pure UI component builder functions.

This module contains pure functions for building Gradio UI components.
All functions here are side-effect free and focused on UI construction.

Following Phase 2 of REFACTORING_STRATEGY.md:
- Pure UI builders (no event handlers, no I/O)
- Full type hints
- Complexity ≤ 10
- Extracted from ui_elements.py
"""

from typing import Any, Union
import gradio as gr
from modules.ui_components import FormRow


def create_gr_elem(d: dict[str, Any]) -> Any:
    """Create a Gradio element from a dictionary specification.

    Converts dictionary with 'type' key to appropriate Gradio component.
    Examples: "dropdown" -> gr.Dropdown, "checkbox_group" -> gr.CheckboxGroup.

    Args:
        d: Dictionary with 'type' key and component parameters

    Returns:
        Instantiated Gradio component
    """
    obj_type_str = "".join(word.title() for word in d["type"].split("_"))
    obj_type = getattr(gr, obj_type_str)

    params = {k: v for k, v in d.items() if k != "type" and v is not None}

    if "type_param" in params:
        params["type"] = params.pop("type_param")

    return obj_type(**params)


def is_gradio_component(args: Any) -> bool:
    """Check if object is a Gradio component.

    Args:
        args: Object to check

    Returns:
        True if object is a known Gradio component type
    """
    return isinstance(
        args,
        (
            gr.Button,
            gr.Textbox,
            gr.Slider,
            gr.Dropdown,
            gr.HTML,
            gr.Radio,
            gr.Interface,
            gr.Markdown,
            gr.Checkbox,
        ),
    )


def create_row(args: Union[dict, Any], *attrs: str) -> Union[list[Any], Any]:
    """Create a FormRow with Gradio components.

    If attrs are provided, creates components from object attributes.
    Otherwise, passes through existing component or creates from dict.

    Args:
        args: Either a Gradio component, dict spec, or object with attrs
        *attrs: Attribute names to extract from args object

    Returns:
        List of components if attrs provided, single component otherwise
    """
    with FormRow():
        if attrs:
            return [create_gr_elem(getattr(args, attr)) for attr in attrs]
        else:
            return args if is_gradio_component(args) else create_gr_elem(args)


def create_accordion_md_row(name: str, markdown: str, is_open: bool = False) -> None:
    """Create a FormRow with an Accordion containing Markdown.

    Args:
        name: Accordion label
        markdown: Markdown content to display
        is_open: Whether accordion starts open (default: False)
    """
    with FormRow():
        with gr.Accordion(name, open=is_open):
            gr.Markdown(markdown)
