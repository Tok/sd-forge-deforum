"""Masking tab for Deforum UI.

Configure masks for selective image generation including basic masks,
text-based masking (CLIPSeg), composable mask expressions, and video/human masks.
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn
from deforum.utils.system.logging import emoji as emoji_utils
from deforum.utils.ui.builders import create_gr_elem, create_row


def _create_masking_content(d, da):
    """Internal helper to create masking tab content.

    Args:
        d: DeforumArgs namespace
        da: DeforumAnimArgs namespace

    Returns:
        dict: Component dictionary for event binding
    """
    with gr.Tabs():
        # Tab 1: Basic Masks
        with gr.TabItem('Basic Masks'):
            gr.Markdown("Upload mask images to control which areas are regenerated. White areas = regenerate, black areas = keep original.")
            with FormRow():
                use_mask = create_gr_elem(d.use_mask)
                use_alpha_as_mask = create_gr_elem(d.use_alpha_as_mask)
                invert_mask = create_gr_elem(d.invert_mask)
                overlay_mask = create_gr_elem(d.overlay_mask)
            mask_file = create_row(d.mask_file)
            mask_overlay_blur = create_row(d.mask_overlay_blur)
            fill = create_row(d.fill)
            full_res_mask, full_res_mask_padding = create_row(d, 'full_res_mask', 'full_res_mask_padding')
            with FormRow():
                with FormColumn(min_width=240):
                    mask_contrast_adjust = create_gr_elem(d.mask_contrast_adjust)
                with FormColumn(min_width=250):
                    mask_brightness_adjust = create_gr_elem(d.mask_brightness_adjust)

        # Tab 2: Text Masking (CLIPSeg)
        with gr.TabItem('Text Masking (CLIPSeg)'):
            gr.Markdown("""
**CLIPSeg Text-to-Mask**: Generate masks using text descriptions in your composable mask expressions.

**Syntax:** `<text description>` - e.g., `<cat>`, `<sky>`, `<person's face>`

**Examples:**
- `<cat>` - Mask all cats
- `<armor>` - Mask armor/metal
- `<sky>` - Mask the sky
- `<person's face>` - Mask faces

**Usage:** Go to the Composable Masks tab and use text masks in your expressions:
- `0: <cat>` - Mask only cats
- `0: (<cat> | <dog>)` - Cats OR dogs
- `0: !<sky>` - Everything EXCEPT sky

The ViT-B/16 CLIPSeg model auto-downloads on first use.
            """)

        # Tab 3: Composable Masks
        with gr.TabItem('Composable Masks'):
            gr.Markdown("""
**Combine masks using boolean expressions**

**Mask Types:**
- `{variable}` - Variable masks (e.g., `{human_mask}`, `{video_mask}`)
- `[path.png]` - File masks from disk
- `<text>` - CLIPSeg text-to-mask

**Operators:** `&` (AND), `|` (OR), `^` (XOR), `!` (NOT), `\\` (DIFFERENCE)

**Examples:**
- `0: <cat>` - Mask only cats
- `0: (<cat> | <dog>)` - Cats OR dogs
- `0: ({human_mask} & [border.png])` - Humans within border
- `0: !<sky>` - Everything except sky
            """)
            mask_schedule = create_row(da.mask_schedule)
            use_noise_mask = create_row(da.use_noise_mask)
            noise_mask_schedule = create_row(da.noise_mask_schedule)

        # Tab 4: Video & Human Masks
        with gr.TabItem('Video & Human'):
            with gr.Accordion("Video Masks (Animated)", open=False):
                gr.Markdown("""
Per-frame video mask sequences. Useful for rotoscoped masks from video editing software or time-varying region selection.
                """)
                use_mask_video = create_gr_elem(da.use_mask_video)
                video_mask_path = create_row(da.video_mask_path)

            with gr.Accordion("Human Detection (AI)", open=False):
                gr.Markdown("""
**Automatic human detection using RobustVideoMatting**

Use `{human_mask}` in composable mask expressions to automatically detect humans.

**Examples:**
- `0: !{human_mask}` - Regenerate everything EXCEPT humans
- `0: {human_mask}` - Regenerate ONLY humans
- `0: ({human_mask} & <armor>)` - Only humans wearing armor

The RobustVideoMatting resnet50 model auto-downloads from PyTorch Hub on first use.
                """)

    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_masking(d, da, skip_tabitem=False):
    """Create the Masking tab.

    Args:
        d: DeforumArgs namespace
        da: DeforumAnimArgs namespace
        skip_tabitem: If True, don't create TabItem wrapper (default: False)

    Returns:
        dict: Component dictionary for event binding
    """
    components = {}

    if not skip_tabitem:
        with gr.TabItem(f'{emoji_utils.masking()} Masking'):
            components.update(_create_masking_content(d, da))
    else:
        components.update(_create_masking_content(d, da))

    return components
