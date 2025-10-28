"""Zero-HITL Slopcore Generator Tab

One-click video generation where Qwen orchestrates the entire pipeline:
- Audio generation (Meta MusicGen)
- Prompt generation (Qwen + audio sync)
- Camera path generation (presets/splines)
- Full Deforum rendering

Zero Human-In-The-Loop creative playground.
"""

import gradio as gr
from types import SimpleNamespace
from deforum.utils.system.logging import emoji as emoji_utils, emoji_if_enabled
from modules.ui_components import FormRow, FormColumn


def get_tab_zero_hitl(skip_tabitem=False):
    """Create the Zero-HITL tab.

    Args:
        skip_tabitem: If True, don't wrap in TabItem (for embedding)

    Returns:
        Dict of Gradio components
    """
    components = {}

    if skip_tabitem:
        # When embedding as subtab, just build UI in current context
        _build_zero_hitl_ui(components)
        return components
    else:
        # Main tab title with dice emoji (respecting global emoji settings)
        tab_emoji = emoji_if_enabled(emoji_utils.dice())
        tab_title = f"{tab_emoji} Zero-HITL" if tab_emoji else "Zero-HITL"
        with gr.TabItem(tab_title, elem_id='zero_hitl_tab'):
            _build_zero_hitl_ui(components)
            return components


def _build_zero_hitl_ui(components: dict):
    """Build the Zero-HITL UI components."""

    gr.HTML(value="""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 10px; margin-bottom: 20px; color: white;'>
            <h2 style='margin: 0 0 10px 0; font-size: 24px;'>🎲 Zero-HITL Slopcore Generator</h2>
            <p style='margin: 0; font-size: 14px; opacity: 0.9;'>
                Let Qwen create a complete animated video with zero human intervention.
                Just set duration and optionally provide a theme or vibe.
                Everything else is randomized creatively.
            </p>
        </div>
    """)

    # Duration input
    with FormRow(variant="compact"):
        zero_hitl_duration = gr.Slider(
            minimum=1.0,
            maximum=10.0,
            value=3.0,
            step=0.5,
            label="Duration (seconds)",
            info="How long should the video be?"
        )

    # Theme/instructions input
    with FormRow(variant="compact"):
        zero_hitl_theme = gr.Textbox(
            label="Theme/Instructions (optional)",
            placeholder="e.g., 'cyberpunk neon city', 'underwater dreamscape', or leave empty for pure randomness",
            lines=3,
            info="Describe the vibe, style, or mood you want (or leave empty for full chaos)"
        )

    gr.Markdown("""
        **Example themes:**
        - `cyberpunk neon city` - Futuristic urban landscapes with neon lights
        - `underwater dreamscape` - Surreal oceanic environments
        - `glitch art chaos` - Digital corruption and artifacts
        - `80s synthwave sunset` - Retro aesthetic with pink/purple gradients
        - `jungle dnb energy` - Fast-paced nature scenes
        - *Leave empty for pure randomness* ✨
    """)

    # Random seed
    with FormRow(variant="compact"):
        zero_hitl_seed = gr.Number(
            value=-1,
            label="Random Seed",
            info="-1 = random seed each time, fixed value = reproducible generation"
        )

    # Comically huge slopcore button
    with FormRow(variant="compact"):
        btn_slop_it = gr.Button(
            "🔥 SLOP IT! 🔥",
            variant="primary",
            size="lg",
            elem_id="btn_slop_it",
            elem_classes=["slopcore-button"],
            scale=2
        )

    gr.HTML(value="""
        <p style='text-align: center; font-size: 11px; opacity: 0.6; margin: -10px 0 10px 0;'>
            NO HUMAN NEEDED (PROBABLY)
        </p>
    """)

    # Status and log output
    with FormRow(variant="compact"):
        zero_hitl_status = gr.Textbox(
            label="Status",
            value="Idle - Ready to generate!",
            interactive=False,
            lines=1
        )

    with FormRow(variant="compact"):
        zero_hitl_log = gr.Textbox(
            label="Generation Log",
            value="",
            interactive=False,
            lines=12,
            max_lines=20
        )

    # Action buttons
    with FormRow(variant="compact"):
        with gr.Column(scale=1):
            btn_view_settings = gr.Button(
                f"{emoji_utils.gear()} View Generated Settings",
                variant="secondary",
                size="sm"
            )
        with gr.Column(scale=1):
            btn_open_output = gr.Button(
                f"{emoji_utils.folder()} Open Output Folder",
                variant="secondary",
                size="sm"
            )

    # Hidden state for generated settings (JSON)
    zero_hitl_generated_settings = gr.State(value=None)

    # Store components for event handlers
    components.update({
        'zero_hitl_duration': zero_hitl_duration,
        'zero_hitl_theme': zero_hitl_theme,
        'zero_hitl_seed': zero_hitl_seed,
        'btn_slop_it': btn_slop_it,
        'zero_hitl_status': zero_hitl_status,
        'zero_hitl_log': zero_hitl_log,
        'btn_view_settings': btn_view_settings,
        'btn_open_output': btn_open_output,
        'zero_hitl_generated_settings': zero_hitl_generated_settings,
    })

    return components
