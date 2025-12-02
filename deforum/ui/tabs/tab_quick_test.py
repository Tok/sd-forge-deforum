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
    gr.HTML(value="""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px; border-radius: 10px; margin-bottom: 20px; color: white;'>
            <h2 style='margin: 0 0 10px 0; font-size: 24px;'>🚀 Quick Test Generator</h2>
            <p style='margin: 0; font-size: 14px; opacity: 0.9;'>
                Verify your Deforum installation with a simple 5-second test video.
                Uses synthetic audio, basic camera movement, and optimized settings.
            </p>
        </div>
    """)

    # Theme inputs for generation
    with FormRow(variant="compact"):
        quick_test_prompt_theme = gr.Textbox(
            label="Prompt Theme",
            value="cute bunny",
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
                info="Test video length"
            )
        with gr.Column(scale=1):
            quick_test_seed = gr.Number(
                value=-1,
                label="Random Seed",
                info="-1 = random"
            )

    # Generate button (slopcore-style)
    movie_camera = emoji_utils.movie_camera()
    with FormRow(variant="compact"):
        btn_generate_test = gr.Button(
            f"{movie_camera} Generate Test Clip" if movie_camera else "Generate Test Clip",
            variant="primary",
            size="lg",
            elem_id="btn_generate_test",
            elem_classes=["slopcore-button"],
            scale=2
        )

    gr.HTML(value="""
        <p style='text-align: center; font-size: 11px; opacity: 0.6; margin: -10px 0 10px 0;'>
            EXPECTED TIME: ~2-3 MINUTES (depending on hardware)
        </p>
    """)

    # Status and log output
    with FormRow(variant="compact"):
        quick_test_status = gr.Textbox(
            label="Status",
            value="Ready to generate test video",
            interactive=False,
            lines=1
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
    gear = emoji_utils.gear()
    folder = emoji_utils.folder()
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
### What This Test Does:

- **Resolution**: 720p (1280x720) - fast rendering
- **FPS**: 60 - smooth playback
- **Duration**: 5 seconds (~300 frames)
- **Render Mode**: New 3D with keyframe redistribution
- **Camera**: Simple forward movement with gentle rotation
- **Audio**: Synthetic beat-synchronized audio (173 BPM)
- **Depth**: Depth-Anything V2 (auto-downloads ~300MB on first run)

### Expected Behavior:

1. Downloads Depth-Anything V2 model if needed
2. Generates synthetic audio with beat pattern
3. Creates keyframes with depth warping
4. Renders smooth video with camera movement
5. Outputs to: `output/deforum/<timestamp>/`

If this test completes successfully, your Deforum installation is working correctly!
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
