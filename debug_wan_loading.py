#!/usr/bin/env python3
"""
Quick test script to verify Wan 2.2 TI2V model loading works
"""

import sys
import os
import torch
from pathlib import Path

# Change to WebUI root directory (models are relative to this)
webui_root = Path(__file__).parent.parent.parent
os.chdir(webui_root)
print(f"Working directory: {os.getcwd()}")

# Add script path for imports
script_path = Path(__file__).parent / "scripts"
sys.path.insert(0, str(script_path))

print("=" * 60)
print("Testing Wan 2.2 TI2V Model Loading")
print("=" * 60)

# Test 1: Model Discovery
print("\n[1/3] Testing model discovery...")
try:
    from deforum.integrations.wan.wan_simple_integration import WanSimpleIntegration

    integration = WanSimpleIntegration()
    models = integration.discover_models()

    if not models:
        print("❌ FAILED: No models found")
        sys.exit(1)

    print(f"✅ SUCCESS: Found {len(models)} model(s)")
    for model in models:
        print(f"   - {model['name']} ({model['type']}, {model['size']})")

except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Get TI2V-5B model
print("\n[2/3] Testing TI2V-5B model selection...")
try:
    # Look for TI2V-5B model
    ti2v_5b_model = None
    for model in models:
        if model['type'] == 'TI2V' and model['size'] == '5B':
            ti2v_5b_model = model
            break

    if not ti2v_5b_model:
        print("⚠️ WARNING: No TI2V-5B model found, using best available")
        ti2v_5b_model = integration.get_best_model()

    print(f"✅ SUCCESS: Selected model: {ti2v_5b_model['name']}")
    print(f"   Path: {ti2v_5b_model['path']}")

except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Pipeline Loading
print("\n[3/3] Testing WanPipeline loading...")
try:
    from diffusers import WanPipeline, AutoencoderKLWan

    print(f"   Loading from: {ti2v_5b_model['path']}")
    print("   This may take a few minutes...")

    # Use float16 for faster loading, CPU offload to avoid OOM
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"   Using device: {device}, dtype: {dtype}")

    # Load VAE separately
    print("   Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        ti2v_5b_model['path'],
        subfolder="vae",
        torch_dtype=torch.float32  # VAE needs float32
    )

    # Load main pipeline with CPU offload to save VRAM
    print("   Loading main pipeline...")
    pipeline = WanPipeline.from_pretrained(
        ti2v_5b_model['path'],
        vae=vae,
        torch_dtype=dtype
    )

    # Enable CPU offload to reduce VRAM usage
    if torch.cuda.is_available():
        print("   Enabling CPU offload...")
        pipeline.enable_model_cpu_offload()

    print("✅ SUCCESS: Pipeline loaded successfully!")
    print(f"   Pipeline components: {list(pipeline.components.keys())}")

    # Clean up
    del pipeline
    del vae
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nWan 2.2 TI2V-5B model is ready for generation.")

except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
