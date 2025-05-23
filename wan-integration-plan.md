# Wan 2.1 Integration Plan for SD-Forge-Deforum

## Overview
This document outlines the step-by-step integration of Wan 2.1 (https://github.com/Wan-Video/Wan2.1) into the sd-forge-deforum extension. The integration will add a new "Wan Video" tab that enables text-to-video and image-to-video generation with proper frame synchronization and audio timing.

## Core Integration Strategy

### 1. Architecture Overview
- **New Animation Mode**: Add "Wan Video" as a new animation mode alongside 2D, 3D, Video Input, and Interpolation
- **Frame-to-Frame Continuity**: Ensure the last frame of each clip becomes the first frame of the next clip
- **Audio Synchronization**: Calculate frame timing based on audio cues and prompt schedules
- **Clip Management**: Generate individual video clips and stitch them together using FFmpeg

### 2. Integration Points

#### A. Arguments System (`scripts/deforum_helpers/args.py`)
Add new `WanArgs()` function with parameters:
```python
def WanArgs():
    return {
        "wan_enabled": {
            "label": "Enable Wan Video Generation",
            "type": "checkbox",
            "value": False,
            "info": "Use Wan 2.1 for video generation instead of traditional diffusion"
        },
        "wan_model_path": {
            "label": "Wan Model Path",
            "type": "textbox", 
            "value": "",
            "info": "Path to Wan 2.1 model checkpoint"
        },
        "wan_clip_duration": {
            "label": "Clip Duration (seconds)",
            "type": "slider",
            "minimum": 1,
            "maximum": 30,
            "step": 0.5,
            "value": 4.0,
            "info": "Duration of each generated video clip"
        },
        "wan_fps": {
            "label": "Wan FPS",
            "type": "slider", 
            "minimum": 8,
            "maximum": 60,
            "step": 1,
            "value": 24,
            "info": "Frames per second for Wan video generation"
        },
        "wan_resolution": {
            "label": "Wan Resolution",
            "type": "dropdown",
            "choices": ["512x512", "768x768", "1024x1024"],
            "value": "768x768",
            "info": "Resolution for Wan video generation"
        },
        "wan_inference_steps": {
            "label": "Inference Steps",
            "type": "slider",
            "minimum": 20,
            "maximum": 100,
            "step": 5,
            "value": 50,
            "info": "Number of inference steps for Wan generation"
        },
        "wan_guidance_scale": {
            "label": "Guidance Scale",
            "type": "slider",
            "minimum": 1.0,
            "maximum": 20.0,
            "step": 0.5,
            "value": 7.5,
            "info": "Guidance scale for prompt adherence"
        },
        "wan_seed": {
            "label": "Wan Seed",
            "type": "number",
            "precision": 0,
            "value": -1,
            "info": "Seed for Wan generation (-1 for random)"
        },
        "wan_frame_overlap": {
            "label": "Frame Overlap",
            "type": "slider",
            "minimum": 0,
            "maximum": 10,
            "step": 1,
            "value": 2,
            "info": "Number of overlapping frames between clips for smoother transitions"
        },
        "wan_motion_strength": {
            "label": "Motion Strength",
            "type": "slider",
            "minimum": 0.0,
            "maximum": 2.0,
            "step": 0.1,
            "value": 1.0,
            "info": "Strength of motion in generated videos"
        }
    }
```

#### B. UI Integration (`scripts/deforum_helpers/ui_elements.py`)
Add new tab function:
```python
def get_tab_wan(dw):
    with gr.TabItem(f"{emoji_utils.wan_video()} Wan Video"):
        # Wan Info Accordion
        with gr.Accordion("Wan 2.1 Video Generation Info", open=False):
            gr.HTML(value=get_gradio_html('wan_video'))
        
        # Main Wan Settings
        wan_enabled = create_row(dw.wan_enabled)
        wan_model_path = create_row(dw.wan_model_path)
        
        with FormRow():
            wan_clip_duration = create_gr_elem(dw.wan_clip_duration)
            wan_fps = create_gr_elem(dw.wan_fps) 
            wan_resolution = create_gr_elem(dw.wan_resolution)
        
        with FormRow():
            wan_inference_steps = create_gr_elem(dw.wan_inference_steps)
            wan_guidance_scale = create_gr_elem(dw.wan_guidance_scale)
            wan_seed = create_gr_elem(dw.wan_seed)
            
        with FormRow():
            wan_frame_overlap = create_gr_elem(dw.wan_frame_overlap)
            wan_motion_strength = create_gr_elem(dw.wan_motion_strength)
        
        # Advanced Settings
        with gr.Accordion("Advanced Wan Settings", open=False):
            # Additional parameters for fine-tuning
            pass
            
    return {k: v for k, v in {**locals(), **vars()}.items()}
```

#### C. Core Wan Module (`scripts/deforum_helpers/wan_integration.py`)
Create new module for Wan integration:
```python
"""
Wan 2.1 Integration Module for Deforum
Handles text-to-video and image-to-video generation using Wan 2.1
"""

import os
import torch
import numpy as np
from PIL import Image
import cv2
from typing import List, Tuple, Optional

class WanVideoGenerator:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.pipeline = None
        
    def load_model(self):
        """Load Wan 2.1 model"""
        # Implementation for loading Wan 2.1 model
        pass
        
    def generate_txt2video(self, prompt: str, **kwargs) -> List[np.ndarray]:
        """Generate video from text prompt"""
        # Implementation for text-to-video generation
        pass
        
    def generate_img2video(self, init_image: np.ndarray, prompt: str, **kwargs) -> List[np.ndarray]:
        """Generate video from initial image and text prompt"""
        # Implementation for image-to-video generation
        pass
        
    def calculate_frame_count(self, duration: float, fps: float) -> int:
        """Calculate number of frames for given duration and FPS"""
        return int(duration * fps)
        
    def extract_last_frame(self, video_frames: List[np.ndarray]) -> np.ndarray:
        """Extract the last frame from a video sequence"""
        return video_frames[-1]
        
    def unload_model(self):
        """Free GPU memory"""
        del self.model
        del self.pipeline
        torch.cuda.empty_cache()
```

#### D. Wan Rendering Module (`scripts/deforum_helpers/render_wan.py`)
```python
"""
Wan Video Rendering Module
Handles the main rendering loop for Wan video generation
"""

def render_wan_animation(args, anim_args, video_args, wan_args, parseq_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, root):
    """
    Main rendering function for Wan video generation
    
    Process:
    1. Parse prompts and calculate timing
    2. Generate first clip using txt2video
    3. For subsequent clips, use img2video with last frame as init
    4. Handle frame overlaps and transitions
    5. Stitch clips together with FFmpeg
    """
    
    # Initialize Wan generator
    wan_generator = WanVideoGenerator(wan_args.wan_model_path, root.device)
    wan_generator.load_model()
    
    try:
        # Parse animation prompts and calculate timing
        prompt_schedule = parse_wan_prompts(root.animation_prompts, wan_args, video_args)
        
        # Generate video clips
        all_clips = []
        previous_frame = None
        
        for i, (prompt, start_time, duration) in enumerate(prompt_schedule):
            print(f"Generating clip {i+1}/{len(prompt_schedule)}: {prompt}")
            
            if i == 0:
                # First clip: text-to-video
                frames = wan_generator.generate_txt2video(
                    prompt=prompt,
                    duration=duration,
                    fps=wan_args.wan_fps,
                    resolution=wan_args.wan_resolution,
                    steps=wan_args.wan_inference_steps,
                    guidance_scale=wan_args.wan_guidance_scale,
                    seed=wan_args.wan_seed
                )
            else:
                # Subsequent clips: image-to-video
                frames = wan_generator.generate_img2video(
                    init_image=previous_frame,
                    prompt=prompt,
                    duration=duration,
                    fps=wan_args.wan_fps,
                    resolution=wan_args.wan_resolution,
                    steps=wan_args.wan_inference_steps,
                    guidance_scale=wan_args.wan_guidance_scale,
                    seed=wan_args.wan_seed
                )
            
            # Handle frame overlap and save frames
            processed_frames = handle_frame_overlap(frames, previous_frame, wan_args.wan_frame_overlap)
            save_clip_frames(processed_frames, args.outdir, root.timestring, i)
            
            all_clips.append(processed_frames)
            previous_frame = wan_generator.extract_last_frame(frames)
            
            # Update progress
            state.job = f"Wan clip {i+1}/{len(prompt_schedule)}"
            state.job_no = i + 1
            
        # Stitch all clips together
        final_video_path = stitch_wan_clips(all_clips, args.outdir, root.timestring, video_args)
        
    finally:
        wan_generator.unload_model()
        
    return final_video_path

def parse_wan_prompts(animation_prompts, wan_args, video_args):
    """
    Parse animation prompts and calculate timing for each clip
    Synchronize with audio if provided
    """
    # Implementation for parsing prompts and calculating timing
    pass

def handle_frame_overlap(frames, previous_frame, overlap_count):
    """
    Handle frame overlapping between clips for smooth transitions
    """
    # Implementation for frame overlap handling
    pass

def save_clip_frames(frames, outdir, timestring, clip_index):
    """
    Save frames from a clip to disk
    """
    # Implementation for saving frames
    pass

def stitch_wan_clips(clips, outdir, timestring, video_args):
    """
    Stitch all video clips together using FFmpeg
    """
    # Implementation for video stitching
    pass
```

#### E. Animation Mode Integration (`scripts/deforum_helpers/args.py`)
Update animation mode choices:
```python
"animation_mode": {
    "label": "Animation mode",
    "type": "radio",
    "choices": ['2D', '3D', 'Video Input', 'Interpolation', 'Wan Video'],
    "value": "2D",
    "info": "control animation mode, will hide non relevant params upon change"
},
```

#### F. Main Rendering Dispatcher (`scripts/deforum_helpers/run_deforum.py`)
Add Wan rendering case:
```python
# In the rendering dispatch section
elif anim_args.animation_mode == 'Wan Video':
    if not wan_args.wan_enabled:
        raise ValueError("Wan Video mode selected but Wan is not enabled")
    render_wan_animation(args, anim_args, video_args, wan_args, parseq_args, loop_args, controlnet_args, freeu_args, kohya_hrfix_args, root)
```

### 3. Technical Implementation Details

#### A. Frame Synchronization Strategy
1. **Prompt Timing Calculation**: 
   - Parse animation_prompts JSON to extract frame numbers
   - Convert frame numbers to milliseconds using FPS
   - Calculate clip durations based on prompt timing

2. **Frame Continuity**:
   - Extract last frame of each clip
   - Use as initialization image for next clip's img2video generation
   - Handle frame overlaps for smoother transitions

3. **Audio Synchronization**:
   - If audio is provided, analyze timing markers
   - Adjust clip durations to match audio cues
   - Maintain lip-sync for talking head videos

#### B. Memory Management
1. **Model Loading**: Load Wan model only when needed
2. **Frame Caching**: Use existing deforum frame caching system
3. **GPU Memory**: Clear VRAM between clips to prevent OOM
4. **Clip Storage**: Save clips temporarily and clean up after stitching

#### C. Settings Integration
1. **Disable Conflicting Settings**: When Wan mode is active, disable:
   - Camera movement parameters (translation, rotation, zoom)
   - Depth warping settings
   - Optical flow settings
   - Traditional diffusion parameters

2. **Preserve Compatible Settings**:
   - Frame interpolation
   - Video output settings
   - Audio settings
   - Upscaling options

### 4. File Structure Changes

```
scripts/deforum_helpers/
├── wan_integration.py          # New: Core Wan integration
├── render_wan.py              # New: Wan rendering logic  
├── wan_utils.py               # New: Wan utility functions
├── args.py                    # Modified: Add WanArgs
├── ui_elements.py             # Modified: Add Wan tab
├── ui_left.py                 # Modified: Include Wan tab
├── run_deforum.py             # Modified: Add Wan dispatch
├── defaults.py                # Modified: Add Wan defaults
└── rendering/
    └── wan_core.py            # New: Wan experimental core
```

### 5. Implementation Phases

#### Phase 1: Foundation (Week 1)
- [ ] Create basic Wan integration module
- [ ] Add WanArgs to args.py
- [ ] Create Wan UI tab
- [ ] Add Wan animation mode option

#### Phase 2: Core Generation (Week 2)
- [ ] Implement WanVideoGenerator class
- [ ] Add txt2video generation
- [ ] Add img2video generation
- [ ] Test basic clip generation

#### Phase 3: Integration (Week 3)
- [ ] Implement render_wan_animation function
- [ ] Add prompt parsing and timing calculation
- [ ] Implement frame continuity system
- [ ] Add clip stitching functionality

#### Phase 4: Optimization (Week 4)
- [ ] Optimize memory usage
- [ ] Add frame overlap handling
- [ ] Implement audio synchronization
- [ ] Add error handling and validation

#### Phase 5: Polish (Week 5)
- [ ] Add comprehensive settings
- [ ] Implement progress tracking
- [ ] Add documentation and examples
- [ ] Performance testing and optimization

### 6. Testing Strategy

#### A. Unit Tests
- Test Wan model loading/unloading
- Test frame extraction and continuity
- Test prompt parsing and timing calculation

#### B. Integration Tests  
- Test complete Wan animation generation
- Test with different prompt schedules
- Test audio synchronization
- Test with various Wan settings

#### C. Performance Tests
- Memory usage profiling
- Generation speed benchmarks
- GPU utilization monitoring

### 7. Documentation Requirements

#### A. User Documentation
- Wan setup and installation guide
- Parameter explanations and recommendations
- Example workflows and use cases
- Troubleshooting guide

#### B. Developer Documentation
- Code architecture overview
- API documentation
- Extension points for future enhancements
- Performance optimization guidelines

### 8. Compatibility Considerations

#### A. Existing Features
- Ensure all existing deforum features continue to work
- Maintain backward compatibility with existing settings
- Preserve existing UI behavior for non-Wan modes

#### B. Model Requirements
- Document Wan 2.1 system requirements
- Provide model download instructions
- Handle model compatibility checks

#### C. Hardware Requirements
- Specify minimum GPU memory requirements
- Document performance expectations
- Provide optimization recommendations

### 9. Future Enhancements

#### A. Advanced Features
- Multi-character video generation
- Advanced motion control integration
- Custom Wan model training integration
- Real-time preview capabilities

#### B. Performance Improvements
- Model optimization and quantization
- Parallel clip generation
- Advanced caching strategies
- GPU memory optimization

### 10. Success Metrics

#### A. Functionality
- [ ] Successfully generate Wan videos from text prompts
- [ ] Achieve smooth frame-to-frame transitions
- [ ] Maintain audio synchronization
- [ ] Generate videos with consistent quality

#### B. Performance
- [ ] Generate 4-second clips in under 60 seconds (target)
- [ ] Memory usage under 12GB VRAM for 768x768 generation
- [ ] Support for clips up to 30 seconds duration
- [ ] Stable generation without crashes

#### C. User Experience
- [ ] Intuitive UI that follows deforum patterns
- [ ] Clear error messages and validation
- [ ] Comprehensive documentation and examples
- [ ] Smooth integration with existing workflows

## Conclusion

This integration plan provides a comprehensive roadmap for adding Wan 2.1 video generation capabilities to the sd-forge-deforum extension. The approach maintains consistency with the existing codebase while adding powerful new video generation features. The phased implementation ensures steady progress and allows for testing and refinement at each stage.

The key innovations of this integration include:
1. **Seamless Frame Continuity**: Ensuring smooth transitions between clips
2. **Audio Synchronization**: Timing clips to match audio cues
3. **Unified Interface**: Integrating Wan controls into the familiar deforum UI
4. **Performance Optimization**: Efficient memory and GPU usage
5. **Extensible Architecture**: Framework for future enhancements

With this plan, users will be able to create high-quality, synchronized video content that combines the power of Wan 2.1's video generation with deforum's extensive animation and scheduling capabilities.
