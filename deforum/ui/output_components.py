"""
Output Components
Contains display elements, outputs, and video processing UI
"""

import gradio as gr
from modules.ui_components import FormRow, FormColumn, ToolButton
from .component_builders import create_gr_elem, create_row
from ..utils import emoji_utils
from ..media.ffmpeg_operations import FFmpegProcessor
from .gradio_functions import upload_pics_to_interpolate, upload_vid_to_depth, ncnn_upload_vid_to_upscale
from ..media.video_audio_pipeline import ffmpeg_stitch_video, direct_stitch_vid_from_frames


def get_tab_output(da, dv):
    """Create the Output tab with video and rendering controls.
    
    Args:
        da: DeforumAnimArgs instance
        dv: DeforumVideoArgs instance
        
    Returns:
        Dict of component references
    """
    with gr.TabItem(f"{emoji_utils.video_camera()} Output"):
        
        # VIDEO OUTPUT SECTION
        with gr.Accordion("🎬 Video Output Settings", open=True):
            with FormRow():
                fps = create_gr_elem(dv.fps)
                max_video_frames = create_gr_elem(dv.max_video_frames)
                add_soundtrack = create_gr_elem(dv.add_soundtrack)
                
            with FormRow():
                soundtrack_path = create_gr_elem(dv.soundtrack_path)
                
            with FormRow():
                skip_video_for_run_all = create_gr_elem(dv.skip_video_for_run_all)
                delete_imgs = create_gr_elem(dv.delete_imgs)
                delete_input_frames = create_gr_elem(dv.delete_input_frames)
                
        # IMAGE OUTPUT SECTION
        with gr.Accordion("🖼️ Image Output Settings", open=False):
            with FormRow():
                save_images = create_gr_elem(dv.save_images)
                save_individual_images = create_gr_elem(dv.save_individual_images)
                
            with FormRow():
                image_format = create_gr_elem(dv.image_format)
                jpeg_quality = create_gr_elem(dv.jpeg_quality)
                
        # FFMPEG SETTINGS
        with gr.Accordion("⚙️ FFmpeg Settings", open=False):
            with FormRow():
                ffmpeg_mode = create_gr_elem(dv.ffmpeg_mode)
                ffmpeg_outdir = create_gr_elem(dv.ffmpeg_outdir)
                
            with FormRow():
                ffmpeg_crf = create_gr_elem(dv.ffmpeg_crf)
                ffmpeg_preset = create_gr_elem(dv.ffmpeg_preset)
                
        # INTERPOLATION SETTINGS
        with gr.Accordion("🔄 Frame Interpolation", open=False):
            with FormRow():
                store_frames_in_ram = create_gr_elem(da.store_frames_in_ram)
                
            # RIFE Section
            with gr.Accordion("RIFE Frame Interpolation", open=False):
                with FormRow():
                    frame_interpolation_engine = create_gr_elem(dv.frame_interpolation_engine)
                    frame_interpolation_x_amount = create_gr_elem(dv.frame_interpolation_x_amount)
                    
                with FormRow():
                    frame_interpolation_slow_mo_enabled = create_gr_elem(dv.frame_interpolation_slow_mo_enabled)
                    frame_interpolation_slow_mo_amount = create_gr_elem(dv.frame_interpolation_slow_mo_amount)
                    
                with FormRow():
                    frame_interpolation_keep_imgs = create_gr_elem(dv.frame_interpolation_keep_imgs)
                    frame_interpolation_use_upscaled = create_gr_elem(dv.frame_interpolation_use_upscaled)
                    
            # FILM Section  
            with gr.Accordion("FILM Frame Interpolation", open=False):
                with FormRow():
                    film_interpolation_x_amount = create_gr_elem(dv.film_interpolation_x_amount)
                    film_interpolation_slow_mo_enabled = create_gr_elem(dv.film_interpolation_slow_mo_enabled)
                    
                with FormRow():
                    film_interpolation_slow_mo_amount = create_gr_elem(dv.film_interpolation_slow_mo_amount)
                    film_interpolation_keep_imgs = create_gr_elem(dv.film_interpolation_keep_imgs)
                    
        # UPSCALING SETTINGS
        with gr.Accordion("📐 Upscaling Settings", open=False):
            with FormRow():
                r_upscale_video = create_gr_elem(dv.r_upscale_video)
                r_upscale_factor = create_gr_elem(dv.r_upscale_factor)
                
            with FormRow():
                r_upscale_model = create_gr_elem(dv.r_upscale_model)
                r_upscale_keep_imgs = create_gr_elem(dv.r_upscale_keep_imgs)
                
    return {k: v for k, v in {**locals(), **vars()}.items()}


def get_tab_ffmpeg():
    """Create the FFmpeg tab with video processing tools.
    
    Returns:
        Dict of component references
    """
    with gr.TabItem("🎬 FFmpeg"):
        gr.Markdown("""
        ## Video Processing Tools
        Use FFmpeg to process and enhance your videos with professional quality tools.
        """)
        
        # File Upload Section
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Upload Video")
                video_file = gr.File(
                    label="Select Video File",
                    file_types=["video"],
                    type="filepath"
                )
                
                # Video Info Display
                video_info = gr.Textbox(
                    label="Video Information",
                    lines=10,
                    interactive=False,
                    placeholder="Upload a video to see its properties..."
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Video", 
                    variant="primary"
                )
                
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Processing Options")
                
                # Upscaling Section
                with gr.Accordion("📐 Upscale Video", open=True):
                    resolution = gr.Dropdown(
                        label="Target Resolution",
                        choices=["720p", "1080p", "1440p", "4K", "Custom"],
                        value="1080p"
                    )
                    
                    custom_res = gr.Textbox(
                        label="Custom Resolution (WxH)",
                        placeholder="1920x1080",
                        visible=False
                    )
                    
                    quality = gr.Slider(
                        label="Output Quality (CRF)",
                        minimum=15,
                        maximum=35,
                        value=23,
                        step=1,
                        info="Lower = better quality, larger file"
                    )
                    
                    upscale_suffix = gr.Textbox(
                        label="Output Suffix",
                        value="_upscaled",
                        placeholder="_upscaled"
                    )
                    
                    upscale_btn = gr.Button(
                        "📐 Upscale Video",
                        variant="secondary"
                    )
        
        # Processing Results Section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔄 Frame Interpolation")
                
                target_fps = gr.Dropdown(
                    label="Target FPS",
                    choices=["30", "60", "120", "Custom"],
                    value="60"
                )
                
                custom_fps = gr.Number(
                    label="Custom FPS",
                    value=60,
                    visible=False
                )
                
                interpolation_method = gr.Dropdown(
                    label="Interpolation Method",
                    choices=["RIFE", "FILM"],
                    value="RIFE"
                )
                
                interpolation_suffix = gr.Textbox(
                    label="Output Suffix",
                    value="_interpolated",
                    placeholder="_interpolated"
                )
                
                interpolate_btn = gr.Button(
                    "🔄 Interpolate Video",
                    variant="secondary"
                )
                
            with gr.Column():
                gr.Markdown("### 🎵 Audio Replacement")
                
                audio_file = gr.File(
                    label="Select Audio File",
                    file_types=["audio"],
                    type="filepath"
                )
                
                start_time = gr.Number(
                    label="Start Time (seconds)",
                    value=0,
                    minimum=0
                )
                
                audio_suffix = gr.Textbox(
                    label="Output Suffix",
                    value="_audio_replaced",
                    placeholder="_audio_replaced"
                )
                
                replace_audio_btn = gr.Button(
                    "🎵 Replace Audio",
                    variant="secondary"
                )
        
        # Progress and Results
        with gr.Row():
            processing_status = gr.Textbox(
                label="Processing Status",
                lines=5,
                interactive=False,
                placeholder="Processing status will appear here..."
            )
        
        # Event Handlers
        def analyze_video_handler(video_file):
            """Analyze uploaded video and return information."""
            if not video_file:
                return "No video file selected."
            
            try:
                processor = FFmpegProcessor()
                info = processor.get_video_info(video_file)
                
                return f"""📹 **Video Analysis Results:**

🎬 **File:** {info.get('filename', 'Unknown')}
📐 **Resolution:** {info.get('width', '?')}x{info.get('height', '?')}
🎯 **FPS:** {info.get('fps', '?')}
⏱️ **Duration:** {info.get('duration', '?')} seconds
💾 **Size:** {info.get('size', '?')} MB
🎵 **Audio:** {info.get('audio_codec', '?')}
🎥 **Video Codec:** {info.get('video_codec', '?')}
📊 **Bitrate:** {info.get('bitrate', '?')} kbps"""
                
            except Exception as e:
                return f"❌ Error analyzing video: {str(e)}"
        
        def upscale_video_handler(video_file, resolution, custom_res, quality, suffix, progress=gr.Progress()):
            """Handle video upscaling."""
            if not video_file:
                return "❌ No video file selected."
            
            try:
                def progress_callback(status):
                    progress(0.5, desc=status)
                
                processor = FFmpegProcessor()
                
                # Determine target resolution
                if resolution == "Custom":
                    if not custom_res or 'x' not in custom_res:
                        return "❌ Invalid custom resolution. Use format: WIDTHxHEIGHT"
                    target_res = custom_res
                else:
                    res_map = {
                        "720p": "1280x720",
                        "1080p": "1920x1080", 
                        "1440p": "2560x1440",
                        "4K": "3840x2160"
                    }
                    target_res = res_map[resolution]
                
                result = processor.upscale_video(
                    video_file, 
                    target_res, 
                    quality, 
                    suffix,
                    progress_callback
                )
                
                return f"✅ Video upscaled successfully!\n📁 Output: {result}"
                
            except Exception as e:
                return f"❌ Upscaling failed: {str(e)}"
        
        def interpolate_video_handler(video_file, target_fps, custom_fps, method, suffix, progress=gr.Progress()):
            """Handle frame interpolation."""
            if not video_file:
                return "❌ No video file selected."
            
            try:
                def progress_callback(status):
                    progress(0.5, desc=status)
                
                processor = FFmpegProcessor()
                
                # Determine target FPS
                fps = custom_fps if target_fps == "Custom" else int(target_fps)
                
                result = processor.interpolate_video(
                    video_file,
                    fps,
                    method,
                    suffix,
                    progress_callback
                )
                
                return f"✅ Video interpolated successfully!\n📁 Output: {result}"
                
            except Exception as e:
                return f"❌ Interpolation failed: {str(e)}"
        
        def replace_audio_handler(video_file, audio_file, start_time, suffix, progress=gr.Progress()):
            """Handle audio replacement."""
            if not video_file:
                return "❌ No video file selected."
            if not audio_file:
                return "❌ No audio file selected."
            
            try:
                def progress_callback(status):
                    progress(0.5, desc=status)
                
                processor = FFmpegProcessor()
                result = processor.replace_audio(
                    video_file,
                    audio_file,
                    start_time,
                    suffix,
                    progress_callback
                )
                
                return f"✅ Audio replaced successfully!\n📁 Output: {result}"
                
            except Exception as e:
                return f"❌ Audio replacement failed: {str(e)}"
        
        # Toggle handlers for custom inputs
        def toggle_custom_resolution(resolution):
            return gr.update(visible=(resolution == "Custom"))
        
        def toggle_custom_fps(target_fps):
            return gr.update(visible=(target_fps == "Custom"))
        
        # Connect event handlers - safely
        if analyze_btn is not None and hasattr(analyze_btn, '_id'):
            valid_inputs = [comp for comp in [video_file] if comp is not None and hasattr(comp, '_id')]
            valid_outputs = [comp for comp in [video_info] if comp is not None and hasattr(comp, '_id')]
            if valid_inputs and valid_outputs:
                try:
                    analyze_btn.click(
                        fn=analyze_video_handler,
                        inputs=valid_inputs,
                        outputs=valid_outputs
                    )
                except Exception as e:
                    print(f"⚠️ Failed to connect analyze button: {e}")
        
        if upscale_btn is not None and hasattr(upscale_btn, '_id'):
            valid_inputs = [comp for comp in [video_file, resolution, custom_res, quality, upscale_suffix] if comp is not None and hasattr(comp, '_id')]
            valid_outputs = [comp for comp in [processing_status] if comp is not None and hasattr(comp, '_id')]
            if valid_inputs and valid_outputs:
                try:
                    upscale_btn.click(
                        fn=upscale_video_handler,
                        inputs=valid_inputs,
                        outputs=valid_outputs
                    )
                except Exception as e:
                    print(f"⚠️ Failed to connect upscale button: {e}")
        
        if interpolate_btn is not None and hasattr(interpolate_btn, '_id'):
            valid_inputs = [comp for comp in [video_file, target_fps, custom_fps, interpolation_method, interpolation_suffix] if comp is not None and hasattr(comp, '_id')]
            valid_outputs = [comp for comp in [processing_status] if comp is not None and hasattr(comp, '_id')]
            if valid_inputs and valid_outputs:
                try:
                    interpolate_btn.click(
                        fn=interpolate_video_handler,
                        inputs=valid_inputs,
                        outputs=valid_outputs
                    )
                except Exception as e:
                    print(f"⚠️ Failed to connect interpolate button: {e}")
        
        if replace_audio_btn is not None and hasattr(replace_audio_btn, '_id'):
            valid_inputs = [comp for comp in [video_file, audio_file, start_time, audio_suffix] if comp is not None and hasattr(comp, '_id')]
            valid_outputs = [comp for comp in [processing_status] if comp is not None and hasattr(comp, '_id')]
            if valid_inputs and valid_outputs:
                try:
                    replace_audio_btn.click(
                        fn=replace_audio_handler,
                        inputs=valid_inputs,
                        outputs=valid_outputs
                    )
                except Exception as e:
                    print(f"⚠️ Failed to connect replace audio button: {e}")
        
        # Toggle handlers - safely
        if resolution is not None and hasattr(resolution, '_id'):
            valid_outputs = [comp for comp in [custom_res] if comp is not None and hasattr(comp, '_id')]
            if valid_outputs:
                try:
                    resolution.change(
                        fn=toggle_custom_resolution,
                        inputs=[resolution],
                        outputs=valid_outputs
                    )
                except Exception as e:
                    print(f"⚠️ Failed to connect resolution change handler: {e}")
        
        if target_fps is not None and hasattr(target_fps, '_id'):
            valid_outputs = [comp for comp in [custom_fps] if comp is not None and hasattr(comp, '_id')]
            if valid_outputs:
                try:
                    target_fps.change(
                        fn=toggle_custom_fps,
                        inputs=[target_fps],
                        outputs=valid_outputs
                    )
                except Exception as e:
                    print(f"⚠️ Failed to connect target_fps change handler: {e}")
        
    return {k: v for k, v in {**locals(), **vars()}.items()} 