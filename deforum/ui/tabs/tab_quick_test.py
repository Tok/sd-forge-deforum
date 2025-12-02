"""Quick Test Tab

Simple one-click test to verify Deforum installation.
Generates a 5-second video with synthetic audio and basic camera movement.
"""

import gradio as gr
from types import SimpleNamespace
from deforum.utils.system.logging import emoji as emoji_utils, emoji_if_enabled
from modules.ui_components import FormRow, FormColumn


def get_tab_quick_test(skip_tabitem=False):
    """Create the Quick Test tab.

    Args:
        skip_tabitem: If True, don't wrap in TabItem (for embedding)

    Returns:
        Dict of Gradio components
    """
    components = {}

    if skip_tabitem:
        # When embedding as subtab, just build UI in current context
        _build_quick_test_ui(components)
        return components
    else:
        # Main tab title with rocket emoji (respecting global emoji settings)
        tab_emoji = emoji_if_enabled(emoji_utils.rocket())
        tab_title = f"{tab_emoji} Quick Test" if tab_emoji else "Quick Test"
        with gr.TabItem(tab_title, elem_id='quick_test_tab'):
            _build_quick_test_ui(components)
            return components


def _build_quick_test_ui(components: dict):
    """Build the Quick Test UI components."""
    # Respect global emoji settings
    rocket = emoji_if_enabled(emoji_utils.rocket())
    warning_emoji = emoji_utils.maybe_warning()  # Already respects emoji settings

    gr.HTML(value=f"""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 10px; margin-bottom: 20px; color: white;'>
            <h2 style='margin: 0 0 10px 0; font-size: 24px;'>{rocket or ''} Quick Test Setup</h2>
            <p style='margin: 0; font-size: 14px; opacity: 0.9;'>
                Generates optimized test animation settings for your current model.
                Creates audio, AI prompts, and camera movement, then <strong>loads them into the UI automatically</strong>.
            </p>
        </div>
    """)

    gr.HTML(value=f"""
        <div style='background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%);
                    border-left: 4px solid #ff1493; padding: 12px; margin-bottom: 15px;
                    border-radius: 5px; color: white;'>
            <p style='margin: 0; font-size: 13px; font-weight: 500;'>
                {warning_emoji or '⚠️'} <strong>Warning:</strong> Clicking "Prepare Quick Test Settings" will replace your current prompts,
                audio path, camera movement, and other settings in the UI. A backup JSON file is saved automatically.
            </p>
        </div>
    """)

    # Theme inputs for generation
    with FormRow(variant="compact"):
        quick_test_prompt_theme = gr.Textbox(
            label="Prompt Theme",
            value="bunny",
            lines=2,
            info="Theme for AI prompt generation (Qwen will create escalating synthwave prompts)"
        )

    with FormRow(variant="compact"):
        quick_test_audio_theme = gr.Textbox(
            label="Audio Theme",
            value="synthetic amen break",
            lines=1,
            info="Theme for audio generation (e.g., 'synthetic amen break', 'jungle dnb', 'lo-fi beats')"
        )

    # Test configuration
    with FormRow(variant="compact"):
        with gr.Column(scale=1):
            quick_test_duration = gr.Slider(
                minimum=3.0,
                maximum=10.0,
                value=5.0,
                step=0.5,
                label="Duration (seconds)",
                info="Test video length (5s default for better prompt sync)"
            )
        with gr.Column(scale=1):
            quick_test_seed = gr.Number(
                value=-1,
                label="Random Seed",
                info="-1 = random"
            )

    # Generate button (slopcore-style)
    gear = emoji_if_enabled(emoji_utils.gear())
    with FormRow(variant="compact"):
        btn_generate_test = gr.Button(
            f"{gear} Prepare Quick Test Settings" if gear else "Prepare Quick Test Settings",
            variant="primary",
            size="lg",
            elem_id="btn_generate_test",
            elem_classes=["slopcore-button"],
            scale=2
        )

    # Respect emoji settings for preparation time message
    clock_emoji = emoji_if_enabled(emoji_utils.stopwatch())

    gr.HTML(value=f"""
        <p style='text-align: center; font-size: 12px; opacity: 0.8; margin: -10px 0 10px 0; font-weight: 500;'>
            {clock_emoji or ''} PREPARATION TIME: ~30-60 seconds (audio + prompts generation)<br>
            {warning_emoji or ''} You MUST click this button BEFORE clicking Generate in the Run tab!
        </p>
    """)

    # Status and log output
    with FormRow(variant="compact"):
        quick_test_status = gr.Textbox(
            label="Status",
            value=f"{warning_emoji or ''} Click 'Prepare Quick Test Settings' first, then go to Run tab and click 'Generate'",
            interactive=False,
            lines=2
        )

    with FormRow(variant="compact"):
        quick_test_log = gr.Textbox(
            label="Generation Log",
            value="",
            interactive=False,
            lines=8,
            max_lines=15
        )

    # Action buttons
    gear = emoji_if_enabled(emoji_utils.gear())
    folder = emoji_if_enabled(emoji_utils.folder())
    with FormRow(variant="compact"):
        with gr.Column(scale=1):
            btn_view_quick_test_settings = gr.Button(
                f"{gear} View Settings" if gear else "View Settings",
                variant="secondary",
                size="sm"
            )
        with gr.Column(scale=1):
            btn_open_quick_test_output = gr.Button(
                f"{folder} Open Output Folder" if folder else "Open Output Folder",
                variant="secondary",
                size="sm"
            )

    # Hidden state for generated settings (JSON)
    quick_test_generated_settings = gr.State(value=None)

    # Info section
    gr.Markdown("""
### What This Preparation Does:

- **Auto-detects** your current model (Flux, Lumina, Z-Image, etc.)
- **Generates** synthetic audio with your chosen theme
- **Creates** AI-generated escalating synthwave prompts with Qwen
- **Optimizes** settings for your model (steps, CFG, sampler, scheduler)
- **Saves** complete settings file ready to load

### Configuration:

- **Resolution**: 720p (1280x720) - fast rendering
- **FPS**: 60 - smooth playback
- **Duration**: 3-10 seconds (your choice)
- **Render Mode**: New 3D with keyframe redistribution
- **Camera**: Simple forward movement with gentle rotation
- **Depth**: Depth-Anything V2 (auto-downloads ~300MB on first run)

### After Preparation:

1. Settings saved to `output/deforum/quick_test_<timestamp>/quick_test_settings.json`
2. Audio file saved in same directory
3. Go to **Run** tab → **Load All Settings** → select the JSON file
4. Click **Generate** to render your test video!

This workflow ensures settings are optimized for your model and lets you review/modify before rendering.
    """)

    # Store components for event handlers
    components.update({
        'quick_test_prompt_theme': quick_test_prompt_theme,
        'quick_test_audio_theme': quick_test_audio_theme,
        'quick_test_duration': quick_test_duration,
        'quick_test_seed': quick_test_seed,
        'btn_generate_test': btn_generate_test,
        'quick_test_status': quick_test_status,
        'quick_test_log': quick_test_log,
        'btn_view_quick_test_settings': btn_view_quick_test_settings,
        'btn_open_quick_test_output': btn_open_quick_test_output,
        'quick_test_generated_settings': quick_test_generated_settings,
    })

    return components
