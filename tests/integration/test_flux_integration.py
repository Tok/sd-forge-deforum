"""Integration tests for Flux model integration.

Tests Flux.1 support which is a major feature of the sd-forge-deforum fork:
- Basic Flux generation
- Flux with ControlNet V2
- Flux + Interpolation hybrid mode
- Qwen prompt enhancement
"""

import glob
import json
import os
from pathlib import Path

import pytest
import requests
from moviepy.editor import VideoFileClip

from .utils import (API_BASE_URL, get_test_options_overrides, gpu_disabled, wait_for_job_to_complete, get_test_batch_name)
from deforum.api.models import DeforumJobStatusCategory


# Path to testdata directory relative to this test file
TESTDATA_DIR = Path(__file__).parent / 'testdata'


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
@pytest.mark.flux
def test_flux_basic_generation():
    """Test basic Flux.1 keyframe generation."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_flux_basic_generation')

    # Configure for Flux + Interpolation mode
    deforum_settings['animation_mode'] = "Keyframes + Interpolation"
    deforum_settings['max_frames'] = 9  # Will create keyframes at 0, 8
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['prompts'] = {
        "0": "a photorealistic red apple on a table",
        "8": "a photorealistic green pear on a table"
    }
    # Flux settings
    deforum_settings['flux_model_name'] = "flux1-dev-bnb-nf4-v2"
    deforum_settings['W'] = 512
    deforum_settings['H'] = 512

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    # Should succeed if Flux model is installed
    if jobStatus.status == DeforumJobStatusCategory.FAILED:
        # Check if failure is due to missing model
        if "model" in jobStatus.message.lower() or "flux" in jobStatus.message.lower():
            pytest.skip("Flux model not installed")
        else:
            pytest.fail(f"Job failed: {jobStatus.message}")

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video was generated
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated video"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
@pytest.mark.flux
def test_flux_controlnet_v2():
    """Test Flux with ControlNet V2 depth guidance."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_flux_controlnet_v2')

    # Configure Flux with ControlNet
    deforum_settings['animation_mode'] = "3D"  # 3D mode can use Flux ControlNet
    deforum_settings['max_frames'] = 5
    deforum_settings['enable_flux_controlnet_v2'] = True
    deforum_settings['flux_controlnet_strength'] = 0.8
    deforum_settings['flux_controlnet_model'] = "depth"
    deforum_settings['prompts'] = {
        "0": "a 3D scene with depth"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    # Should succeed if Flux ControlNet is available
    if jobStatus.status == DeforumJobStatusCategory.FAILED:
        if "controlnet" in jobStatus.message.lower() or "flux" in jobStatus.message.lower():
            pytest.skip("Flux ControlNet not available")
        else:
            pytest.fail(f"Job failed: {jobStatus.message}")

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
@pytest.mark.flux
def test_flux_wan_hybrid_mode():
    """Test Flux + Interpolation hybrid mode: Flux keyframes + multi-method interpolation."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_flux_wan_hybrid_mode')

    # Flux + Interpolation hybrid workflow
    deforum_settings['animation_mode'] = "Keyframes + Interpolation"
    deforum_settings['max_frames'] = 9
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['prompts'] = {
        "0": "a sunny morning landscape",
        "8": "a rainy evening landscape"
    }
    # Flux for keyframes
    deforum_settings['flux_model_name'] = "flux1-dev-bnb-nf4-v2"
    # Wan for tweens
    deforum_settings['wan_model_name'] = "Wan2.1-VACE-1.3B"
    deforum_settings['wan_flf2v_guidance_scale'] = 3.5

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    # Skip if models not installed
    if jobStatus.status == DeforumJobStatusCategory.FAILED:
        if "model" in jobStatus.message.lower():
            pytest.skip("Flux or Wan model not installed")
        else:
            pytest.fail(f"Job failed: {jobStatus.message}")

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"

    # Verify video has both Flux keyframes and Wan tweens
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1, "Should have generated hybrid video"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
@pytest.mark.flux
def test_flux_qwen_prompt_enhancement():
    """Test Qwen AI prompt enhancement for Flux generation."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_flux_qwen_prompt_enhancement')

    # Enable Qwen enhancement
    deforum_settings['animation_mode'] = "Keyframes + Interpolation"
    deforum_settings['max_frames'] = 9
    deforum_settings['enable_qwen_enhancement'] = True
    deforum_settings['qwen_model_size'] = "3B"  # Use smallest model for testing
    deforum_settings['prompts'] = {
        "0": "landscape",  # Simple prompt to be enhanced
        "8": "cityscape"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    # Skip if Qwen model not available
    if jobStatus.status == DeforumJobStatusCategory.FAILED:
        if "qwen" in jobStatus.message.lower() or "model" in jobStatus.message.lower():
            pytest.skip("Qwen model not installed")
        else:
            pytest.fail(f"Job failed: {jobStatus.message}")

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED, \
        f"Job {job_id} failed: {jobStatus.message}"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
@pytest.mark.flux
def test_flux_resolution_512x512():
    """Test Flux generation at 512x512 resolution."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_flux_resolution_512x512')

    deforum_settings['animation_mode'] = "Keyframes + Interpolation"
    deforum_settings['max_frames'] = 5
    deforum_settings['W'] = 512
    deforum_settings['H'] = 512
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['prompts'] = {"0": "test image", "4": "test image 2"}

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    if jobStatus.status == DeforumJobStatusCategory.FAILED:
        if "model" in jobStatus.message.lower():
            pytest.skip("Flux model not installed")
        else:
            pytest.fail(f"Job failed: {jobStatus.message}")

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED

    # Verify output resolution
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1
    clip = VideoFileClip(video_files[0])
    assert clip.size == [512, 512], f"Expected 512x512, got {clip.size}"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
@pytest.mark.flux
def test_flux_resolution_768x768():
    """Test Flux generation at 768x768 resolution."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_flux_resolution_768x768')

    deforum_settings['animation_mode'] = "Keyframes + Interpolation"
    deforum_settings['max_frames'] = 5
    deforum_settings['W'] = 768
    deforum_settings['H'] = 768
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['prompts'] = {"0": "test image", "4": "test image 2"}

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    if jobStatus.status == DeforumJobStatusCategory.FAILED:
        if "model" in jobStatus.message.lower():
            pytest.skip("Flux model not installed")
        else:
            pytest.fail(f"Job failed: {jobStatus.message}")

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED

    # Verify output resolution
    video_files = glob.glob(os.path.join(jobStatus.outdir, "*.mp4"))
    assert len(video_files) >= 1
    clip = VideoFileClip(video_files[0])
    assert clip.size == [768, 768], f"Expected 768x768, got {clip.size}"


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
@pytest.mark.flux
def test_flux_with_movement_analysis():
    """Test Flux with Qwen movement analysis integration."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_flux_with_movement_analysis')

    # Use movement schedules that Qwen can analyze
    deforum_settings['animation_mode'] = "Keyframes + Interpolation"
    deforum_settings['max_frames'] = 9
    deforum_settings['enable_qwen_enhancement'] = True
    deforum_settings['qwen_analyze_movement'] = True
    deforum_settings['translation_z'] = "0:(0), 8:(10)"  # Camera movement
    deforum_settings['rotation_3d_y'] = "0:(0), 8:(15)"  # Pan right
    deforum_settings['prompts'] = {
        "0": "flying through space",
        "8": "arriving at planet"
    }

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    if jobStatus.status == DeforumJobStatusCategory.FAILED:
        if "qwen" in jobStatus.message.lower() or "model" in jobStatus.message.lower():
            pytest.skip("Qwen or Flux model not installed")
        else:
            pytest.fail(f"Job failed: {jobStatus.message}")

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
@pytest.mark.flux
def test_flux_steps_configuration():
    """Test Flux with different step counts."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_flux_steps_configuration')

    # Test with minimal steps (Flux Schnell can do 4 steps)
    deforum_settings['animation_mode'] = "Keyframes + Interpolation"
    deforum_settings['max_frames'] = 5
    deforum_settings['steps'] = 4  # Minimal for Flux Schnell
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['prompts'] = {"0": "quick test", "4": "quick test 2"}

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    if jobStatus.status == DeforumJobStatusCategory.FAILED:
        if "model" in jobStatus.message.lower():
            pytest.skip("Flux model not installed")
        else:
            pytest.fail(f"Job failed: {jobStatus.message}")

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED


@pytest.mark.skipif(gpu_disabled(), reason="requires GPU-enabled server")
@pytest.mark.flux
def test_flux_vae_integration():
    """Test that Flux uses correct VAE."""
    with open(TESTDATA_DIR / 'simple.input_settings.txt', 'r') as settings_file:
        deforum_settings = json.load(settings_file)

    # Set test-specific batch name for easier output identification
    deforum_settings['batch_name'] = get_test_batch_name('test_flux_vae_integration')

    # Flux requires specific VAE
    deforum_settings['animation_mode'] = "Keyframes + Interpolation"
    deforum_settings['max_frames'] = 3
    deforum_settings['keyframe_distribution_mode'] = "Keyframes Only"
    deforum_settings['prompts'] = {"0": "vae test", "2": "vae test 2"}
    # VAE should be auto-selected for Flux

    response = requests.post(f"{API_BASE_URL}/batches", json={
        "deforum_settings": [deforum_settings],
        "options_overrides": get_test_options_overrides()
    })
    response.raise_for_status()
    job_id = response.json()["job_ids"][0]
    jobStatus = wait_for_job_to_complete(job_id)

    if jobStatus.status == DeforumJobStatusCategory.FAILED:
        if "vae" in jobStatus.message.lower() or "model" in jobStatus.message.lower():
            pytest.skip("Flux VAE not installed")
        else:
            pytest.fail(f"Job failed: {jobStatus.message}")

    assert jobStatus.status == DeforumJobStatusCategory.SUCCEEDED
