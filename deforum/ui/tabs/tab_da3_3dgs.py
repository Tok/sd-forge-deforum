"""DA3-3DGS (Depth Anything V3 + 3D Gaussian Splatting) interpolation tab.

Completely separate from Wan - handles 3DGS-specific settings for novel view synthesis.
"""

import gradio as gr
from types import SimpleNamespace
from deforum.utils.system.logging import emoji_if_enabled
from deforum.utils.ui.builders import create_gr_elem


def get_tab_da3_3dgs(dw: SimpleNamespace, skip_tabitem=False):
    """DA3-3DGS Settings Tab - 3D Gaussian Splatting interpolation settings.
    
    Args:
        dw: DeforumWanArgs namespace (contains DA3-3DGS params)
        skip_tabitem: If True, don't create TabItem wrapper
        
    Returns:
        Dict of all created Gradio components
    """
    
    gr.Markdown(f"""
    ## {emoji_if_enabled('🔍')} DA3-3DGS Interpolation Settings
    
    **3D Gaussian Splatting with Depth Anything V3 GIANT models**
    
    This mode uses multi-view 3D reconstruction to interpolate between keyframes:
    - Collects multiple consecutive keyframes (2-10, default 5)
    - DA3 auto-estimates camera poses from image content
    - Builds 3D Gaussian Splatting scene (~705k splats)
    - Renders novel views via camera pose interpolation
    - **Note:** Deforum camera schedules are NOT used - DA3 drives movement
    
    ---
    """)
    
    # Model Selection
    gr.Markdown(f"### {emoji_if_enabled('🎛')} Model Configuration")
    with gr.Row():
        da3_3dgs_model = create_gr_elem(dw.da3_3dgs_model)
        da3_3dgs_neighbor_segments = create_gr_elem(dw.da3_3dgs_neighbor_segments)
    
    # Output Settings
    gr.Markdown(f"### {emoji_if_enabled('💾')} Output Settings")
    with gr.Row():
        da3_3dgs_render_keyframes = create_gr_elem(dw.da3_3dgs_render_keyframes)
    
    # Technical Info
    with gr.Accordion(f"{emoji_if_enabled('ℹ️')} Technical Details", open=False):
        gr.Markdown("""
        **How DA3-3DGS Works:**
        
        1. **Multi-View Collection**: Gathers N consecutive keyframes around each segment
        2. **3D Reconstruction**: DA3 GIANT model estimates camera poses and builds gaussian scene
        3. **Novel View Synthesis**: Interpolates camera poses and renders intermediate views
        4. **Visual Consistency**: Optionally renders 3DGS versions of keyframes
        
        **Camera Pose Estimation:**
        - DA3 automatically estimates poses from image content
        - Uses SLERP for rotation interpolation
        - Linear interpolation for translation
        - Deforum movement schedules are ignored
        
        **Output Organization:**
        - Original diffusion keyframes → `_diffusion/` subdirectory
        - 3DGS-rendered keyframes → main directory
        - 3DGS tweens → main directory
        - Result: Seamless 3DGS video
        
        **VRAM Requirements:**
        - DA3-GIANT: ~3GB for model, ~1-2GB per scene
        - DA3NESTED-GIANT-LARGE: ~4GB for model, ~1-2GB per scene
        - Recommended: 16GB minimum, 24GB+ for best quality
        
        **Performance:**
        - ~705k gaussian splats per scene
        - 5 keyframes = good balance of quality/speed
        - More keyframes = better geometry but slower
        """)
    
    # CRITICAL: Immediately capture components in locals() for registration
    locals()['da3_3dgs_model'] = da3_3dgs_model
    locals()['da3_3dgs_neighbor_segments'] = da3_3dgs_neighbor_segments
    locals()['da3_3dgs_render_keyframes'] = da3_3dgs_render_keyframes
    
    return {k: v for k, v in {**locals(), **vars()}.items()}
