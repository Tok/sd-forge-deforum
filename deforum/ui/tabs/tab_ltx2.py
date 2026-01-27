"""LTX-2 Audio-Video subtab for Deforum UI.

LTX-2 is the first truly open audio-video AI model (19B params: 14B video + 5B audio).
Provides high-quality video generation with frame-accurate audio synchronization.
"""

import gradio as gr
from types import SimpleNamespace
from modules.ui_components import FormRow
from deforum.utils.system.logging import emoji as emoji_utils, emoji_if_enabled
from deforum.utils.ui.builders import create_gr_elem


def get_subtab_ltx2(dw: SimpleNamespace, skip_tabitem=False):
    """LTX-2 Audio-Video Subtab

    Args:
        dw: DeforumWanArgs namespace (contains ltx2_model_variant, ltx2_audio_mode)
        skip_tabitem: If True, don't create TabItem wrapper (always True for subtabs)
    """

    video_icon = emoji_utils.video_camera()  # 📹
    audio_icon = emoji_utils.sound()  # 🎵
    warning = emoji_utils.maybe_warning()  # ⚠️

    gr.Markdown(f"""
    ## {video_icon} LTX-2 Audio-Video AI

    **First truly open audio-video model** (19B params: 14B video + 5B audio)

    ### {warning} VRAM Requirements

    **LTX-2-4K-NF4 (Recommended):** 12GB VRAM with 4-bit quantization
    - ✅ Works on RTX 4070 Ti (16GB), RTX 4080 (16GB), RTX 4070 (12GB)
    - Uses BitsAndBytes NF4 quantization (~75% memory reduction)

    **LTX-2-4K (Full Precision):** 24GB+ VRAM required
    - Only for RTX 4090, RTX 6000 Ada, A6000, etc.

    ### ✨ Features

    - **Native 4K resolution @ 50fps**
    - **Up to 20 seconds per clip**
    - **Audio-guided conditioning** for perfect sync
    - **Frame-accurate synchronization** (<1 frame error)

    ### {audio_icon} Audio Integration

    LTX-2 uses Deforum's audio track as conditioning, ensuring perfect frame-accurate
    synchronization. The model generates video matched to your existing audio.

    **Supported Audio Formats:**
    - WAV (lossless, best reliability)
    - FLAC (lossless, compressed)
    - MP3 (lossy, widely supported)
    - OGG, M4A/AAC (requires ffmpeg)

    **Audio Processing:**
    - Automatically extracts segments on-the-fly
    - Resamples to 16kHz mono for conditioning
    - No pre-conversion needed (librosa handles everything)

    ### 💡 Tips

    - Select **"Auto"** to automatically choose the best variant for your GPU
    - Use **"condition_only"** audio mode for best quality (discards LTX-2's generated audio)
    - LTX-2 generates at 50fps native, automatically resampled to match Deforum FPS

    ---
    """)

    # Model Configuration
    gr.Markdown(f"### {emoji_if_enabled('⚙️')} Model Configuration")

    with FormRow():
        ltx2_model_variant = create_gr_elem(dw.ltx2_model_variant)
        ltx2_audio_mode = create_gr_elem(dw.ltx2_audio_mode)

    with FormRow():
        ltx2_num_inference_steps = create_gr_elem(dw.ltx2_num_inference_steps)
        ltx2_guidance_scale = create_gr_elem(dw.ltx2_guidance_scale)

    # Generation Settings
    gr.Markdown(f"### {emoji_if_enabled('🎨')} Generation Settings")

    ltx2_negative_prompt = create_gr_elem(dw.ltx2_negative_prompt)

    # Model Info
    with gr.Accordion(f"{emoji_utils.bulb()} Model Details", open=False):
        gr.Markdown("""
        **LTX-2 Model Variants:**

        | Variant | VRAM | Quantization | Resolution | FPS | Notes |
        |---------|------|--------------|------------|-----|-------|
        | **LTX-2-4K-NF4** | 12GB | 4-bit NF4 | 4K (3840x2160) | 50 | Recommended for 16GB cards |
        | **LTX-2-4K** | 24GB | None (FP16) | 4K (3840x2160) | 50 | Full precision, high-end GPUs |
        | **LTX-2-HD** | 18GB | None (FP16) | 1080p (1920x1080) | 30 | Mid-tier quality |

        **Audio Modes:**
        - **condition_only** (Recommended): Uses Deforum audio as conditioning, discards LTX-2 audio
        - **blend**: Blends Deforum and LTX-2 audio (experimental)
        - **replace**: Uses LTX-2's generated audio instead (not recommended for sync)

        **VRAM Management:**
        - Model loads on-demand when LTX-2 interpolation is selected
        - Automatically offloads after generation to free VRAM
        - Uses BitsAndBytes for efficient 4-bit quantization
        """)

    # Integration Details
    with gr.Accordion(f"{emoji_utils.link()} Deforum Integration", open=False):
        check = emoji_utils.maybe_check()
        memo = emoji_utils.memo()
        movie_camera = emoji_utils.movie_camera()
        fps_icon = emoji_utils.stopwatch()

        gr.Markdown(f"""
        **{check} LTX-2 seamlessly integrates with Deforum:**

        - **{memo} Prompts:** Uses prompts from Deforum Prompts tab
        - **{audio_icon} Audio:** Uses audio track from Output tab as conditioning
        - **{fps_icon} FPS:** Generates at 50fps, resamples to match Deforum FPS
        - **{movie_camera} Segments:** Automatically handles first-last-frame pairs
        - **{check} Sync Verification:** Ensures <1 frame audio alignment error

        **Workflow:**
        1. Generate keyframes with Flux (Phase 1)
        2. Extract audio segments for each tween chunk
        3. LTX-2 generates video conditioned on audio + keyframes
        4. Resample to match Deforum FPS
        5. Verify audio sync accuracy
        6. Stitch final video with original audio track

        **Quality Settings:**
        - Uses same steps/CFG as Flux keyframes
        - Guidance scale automatically adapts based on prompt similarity
        - Temporal consistency enforced via keyframe conditioning
        """)

    # Return components dict (matches pattern from other subtabs)
    return {
        'ltx2_model_variant': ltx2_model_variant,
        'ltx2_audio_mode': ltx2_audio_mode,
        'ltx2_num_inference_steps': ltx2_num_inference_steps,
        'ltx2_guidance_scale': ltx2_guidance_scale,
        'ltx2_negative_prompt': ltx2_negative_prompt,
    }
