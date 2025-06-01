from ...media.video_audio_pipeline import ffmpeg_stitch_video, get_ffmpeg_params
import os


def create_video(data):
    """
    Create a video from frames using the standard Deforum video creation logic.
    This wraps the ffmpeg_stitch_video function with proper parameter extraction.
    """
    # Get FFmpeg parameters from settings
    f_location, f_crf, f_preset = get_ffmpeg_params()
    
    # Extract parameters from data
    args = data.args.args
    anim_args = data.args.anim_args
    video_args = data.args.video_args
    root = data.args.root
    
    # Build output paths
    outdir = args.outdir
    timestring = root.timestring
    
    # Create video filename
    mp4_path = os.path.join(outdir, f"{timestring}.mp4")
    
    # Create pattern for input images
    imgs_path = os.path.join(outdir, f"{timestring}_%09d.png")
    
    # Audio settings
    add_soundtrack = video_args.add_soundtrack if hasattr(video_args, 'add_soundtrack') else 'None'
    audio_path = video_args.soundtrack_path if hasattr(video_args, 'soundtrack_path') else None
    
    # SRT path if subtitles are enabled
    srt_path = os.path.join(outdir, f"{timestring}.srt") if hasattr(video_args, 'save_gen_info_as_srt') else None
    
    # Call the actual FFmpeg stitching function
    ffmpeg_stitch_video(
        ffmpeg_location=f_location,
        fps=video_args.fps,
        outmp4_path=mp4_path,
        stitch_from_frame=0,
        stitch_to_frame=anim_args.max_frames,
        imgs_path=imgs_path,
        add_soundtrack=add_soundtrack,
        audio_path=audio_path,
        crf=f_crf,
        preset=f_preset,
        srt_path=srt_path
    ) 