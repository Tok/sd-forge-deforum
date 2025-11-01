"""Integration tests for visual processing features.

Tests features that require models, GPU, and produce visual outputs:
- Depth map generation (Depth-Anything V2)
- Tween frame generation with FLF2V
- RAFT optical flow
- Camera shakify patterns
- Keyframe distribution modes
- 3D depth warping
- Prompt/seed/strength scheduling

These tests verify the visual processing pipeline works correctly end-to-end.
"""

import glob
import json
import math
import os
from pathlib import Path

import pytest
import requests
import numpy as np
from PIL import Image
from moviepy.editor import VideoFileClip

from .utils import (API_BASE_URL, get_test_options_overrides, gpu_disabled, wait_for_job_to_complete, get_test_batch_name)
from deforum.api.models import DeforumJobStatusCategory


# Path to testdata directory relative to this test file
TESTDATA_DIR = Path(__file__).parent / 'testdata'


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_depth_map_generation():
    """Test that Depth-Anything V2 generates depth maps correctly."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_depth_map_generation')

    # Configure for 3D mode with depth
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 3  # Just a few frames
    deforum_settings['save_depth_maps'] = True  # Save depth maps

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Check that depth maps were generated
    depth_dir = os.path.join(jobStatus.outdir, "depth-maps")
    assert os.path.exists(depth_dir), "Depth maps directory should exist"

    depth_files = glob.glob(os.path.join(depth_dir, "*.png"))
    assert len(depth_files) == deforum_settings['max_frames'], \
        f"Should have {deforum_settings['max_frames']} depth maps, found {len(depth_files)}"

    # Verify depth maps are valid images with correct dimensions
    for depth_file in depth_files:
        img = Image.open(depth_file)
        assert img.size == (deforum_settings['W'], deforum_settings['H']), \
            f"Depth map should match output dimensions: {img.size}"
        # Depth maps should be grayscale or RGB
        assert img.mode in ['L', 'RGB'], f"Unexpected depth map mode: {img.mode}"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_3d_depth_warping():
    """Test that 3D mode performs depth-based warping."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_3d_depth_warping')

    # Configure 3D animation with camera movement
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 5
    deforum_settings['translation_z'] = "0:(0), 4:(10)"  # Move forward
    deforum_settings['rotation_3d_x'] = "0:(0), 4:(5)"  # Tilt up slightly

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video was generated
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated at least one video"

    video_file = video_files[0]
    clip = VideoFileClip(video_file)
    assert math.ceil(clip.duration * clip.fps) == deforum_settings['max_frames'], \
        "Video should have correct frame count"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_keyframe_distribution_keyframes_only():
    """Test keyframe distribution in 'Keyframes Only' mode."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_keyframe_distribution_keyframes_only')

    # Set up keyframe distribution
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 10
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['prompts'] = {
        "0": "a red apple",
        "5": "a blue orange",
        "9": "a green banana"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # In Keyframes Only mode, only frames 0, 5, 9 should be diffused
    # Others should be tweened
    # We can't directly verify this without inspecting logs, but we can
    # verify the video was generated successfully
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated video"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_prompt_scheduling():
    """Test that prompt scheduling works across multiple keyframes."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_prompt_scheduling')

    # Set up prompt schedule with distinct prompts
    deforum_settings['max_frames'] = 8
    deforum_settings['prompts'] = {
        "0": "a sunny day in the park",
        "3": "a rainy evening in the city",
        "6": "a snowy night in the mountains"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify all frames were generated
    frame_files = glob.glob(os.path.join(jobStatus.outdir, "*.png"))
    # Should have max_frames worth of PNG files
    assert len(frame_files) >= deforum_settings['max_frames'], \
        f"Should have at least {deforum_settings['max_frames']} frames"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_seed_scheduling():
    """Test that seed scheduling works across frames."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_seed_scheduling')

    # Set up seed schedule
    deforum_settings['max_frames'] = 6
    deforum_settings['seed_schedule'] = "0:(42), 3:(1337), 5:(9999)"
    deforum_settings['prompts'] = {"0": "a landscape"}

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video was generated
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated video"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_strength_scheduling():
    """Test that strength (denoise) scheduling works."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_strength_scheduling')

    # Set up strength schedule
    deforum_settings['max_frames'] = 6
    deforum_settings['strength_schedule'] = "0:(0.6), 3:(0.8), 5:(0.4)"
    deforum_settings['prompts'] = {"0": "a morphing sculpture"}

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video was generated
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated video"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_camera_shakify():
    """Test that camera shakify patterns are applied."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_camera_shakify')

    # Enable shakify
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 8
    deforum_settings['enable_shakify'] = True
    deforum_settings['shakify_pattern'] = "GENTLE_HANDHELD"
    deforum_settings['shakify_magnitude'] = 0.5

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video was generated with shakify applied
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated video with shakify"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_optical_flow_raft():
    """Test RAFT optical flow for color matching."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_optical_flow_raft')

    # Enable RAFT optical flow with HSV color coherence
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 5
    deforum_settings['color_coherence'] = 'HSV'
    deforum_settings['optical_flow_redo_generation'] = 'RAFT'

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video was generated
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated video with RAFT flow"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_subtitle_generation():
    """Test that .srt subtitle files are generated correctly."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_subtitle_generation')

    deforum_settings['max_frames'] = 5
    deforum_settings['prompts'] = {
        "0": "first prompt",
        "2": "second prompt",
        "4": "third prompt"
    }

    from deforum.media.subtitle_handler import get_user_values

    options_overrides = get_test_options_overrides()
    options_overrides.update({
        "deforum_save_gen_info_as_srt": True,
        "deforum_save_gen_info_as_srt_params": get_user_values(),
    })

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": options_overrides
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify .srt file was generated
    srt_file = os.path.join(jobStatus.outdir, f"{jobStatus.timestring}.srt")
    assert os.path.exists(srt_file), "Subtitle file should be generated"

    # Verify .srt file has content
    with open(srt_file, 'r') as f:
        content = f.read()
        assert len(content) > 0, "Subtitle file should not be empty"
        assert "first prompt" in content, "Subtitle should contain first prompt"
        assert "second prompt" in content, "Subtitle should contain second prompt"
        assert "third prompt" in content, "Subtitle should contain third prompt"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_video_stitching_basic():
    """Test that ffmpeg correctly stitches frames into video."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_video_stitching_basic')

    # Basic settings for video stitching
    deforum_settings['max_frames'] = 4
    deforum_settings['fps'] = 10

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video file exists and has correct properties
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated video file"

    video_file = video_files[0]
    clip = VideoFileClip(video_file)

    # Verify FPS
    assert clip.fps == deforum_settings['fps'], \
        f"Video FPS should be {deforum_settings['fps']}, got {clip.fps}"

    # Verify frame count
    expected_frames = deforum_settings['max_frames']
    actual_frames = math.ceil(clip.duration * clip.fps)
    assert actual_frames == expected_frames, \
        f"Video should have {expected_frames} frames, got {actual_frames}"

    # Verify dimensions
    assert clip.size == [deforum_settings['W'], deforum_settings['H']], \
        f"Video dimensions should be {deforum_settings['W']}x{deforum_settings['H']}"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_cfg_scale_schedule():
    """Test that CFG scale scheduling works."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_cfg_scale_schedule')

    # Set up CFG scale schedule
    deforum_settings['max_frames'] = 6
    deforum_settings['cfg_scale_schedule'] = "0:(7.0), 3:(10.0), 5:(5.0)"
    deforum_settings['prompts'] = {"0": "a controlled generation"}

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video was generated
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated video"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_color_coherence():
    """Test color coherence modes."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_color_coherence')

    # Test with HSV color coherence
    deforum_settings['animation_mode'] = "3D"
    deforum_settings['max_frames'] = 5
    deforum_settings['color_coherence'] = 'HSV'

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video was generated
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated video with color coherence"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
def test_noise_schedule():
    """Test noise scheduling across frames."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_noise_schedule')

    # Set up noise schedule
    deforum_settings['max_frames'] = 6
    deforum_settings['noise_schedule'] = "0:(0.02), 3:(0.05), 5:(0.01)"
    deforum_settings['prompts'] = {"0": "evolving noise patterns"}

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video was generated
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated video"
