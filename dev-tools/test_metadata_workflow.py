#!/usr/bin/env python3
"""Integration test for video metadata workflow.

Tests the complete workflow:
1. Create comprehensive metadata from settings
2. Encode metadata for video embedding
3. Create ffmpeg metadata arguments
4. Decode metadata (simulates extraction from video)

This verifies all components work together without requiring video generation.
"""

import sys
import importlib.util
from pathlib import Path

# Import metadata module directly bypassing package __init__.py
# (avoids heavy dependencies in deforum.media.__init__.py)
metadata_path = Path(__file__).parent / "deforum" / "media" / "metadata.py"
spec = importlib.util.spec_from_file_location("metadata", metadata_path)
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)

# Import specific functions and constants
create_comprehensive_metadata = metadata.create_comprehensive_metadata
encode_settings_for_metadata = metadata.encode_settings_for_metadata
decode_settings_from_metadata = metadata.decode_settings_from_metadata
create_ffmpeg_metadata_args = metadata.create_ffmpeg_metadata_args
METADATA_PREFIX = metadata.METADATA_PREFIX


def test_full_metadata_workflow():
    """Test complete metadata workflow."""
    print("\n" + "="*70)
    print("TESTING VIDEO METADATA WORKFLOW")
    print("="*70 + "\n")

    # Step 1: Create settings (simulating what run_deforum.py does)
    print("1. Creating comprehensive settings metadata...")
    settings = {
        # Core generation settings
        "W": 1280,
        "H": 720,
        "seed": 123456789,
        "steps": 20,
        "cfg_scale": 7.5,
        "distilled_cfg_scale": 3.5,
        "sampler": "euler_a",
        "scheduler": "exponential",
        "sd_model_checkpoint": "flux1-dev-bnb-nf4-v2.safetensors",

        # Animation settings
        "fps": 60,
        "max_frames": 240,
        "render_mode": "New 3D",
        "animation_mode": "3D",

        # Batch info
        "batch_name": "test_batch",
        "n_batch": 1,

        # Prompts
        "animation_prompts": {
            0: "a beautiful landscape",
            120: "a stunning sunset",
        },
    }

    comprehensive = create_comprehensive_metadata(settings)
    print(f"   ✓ Added commit_id: {comprehensive.get('commit_id', 'Unknown')}")
    print(f"   ✓ Added github_url: {comprehensive.get('github_url')}")
    print(f"   ✓ Added fork_name: {comprehensive.get('fork_name', 'Unknown')[:50]}...")
    print(f"   ✓ Total settings: {len(comprehensive)}")

    # Step 2: Create ffmpeg metadata arguments
    print("\n2. Creating ffmpeg metadata arguments...")
    metadata_args = create_ffmpeg_metadata_args(comprehensive, include_readable_fields=True)
    metadata_count = metadata_args.count('-metadata')
    print(f"   ✓ Generated {metadata_count} metadata fields")

    # Find the comment field (base64-encoded comprehensive data)
    comment_value = None
    for i, arg in enumerate(metadata_args):
        if arg == '-metadata' and i + 1 < len(metadata_args):
            if metadata_args[i + 1].startswith('comment='):
                comment_value = metadata_args[i + 1][8:]
                break

    if not comment_value:
        print("   ✗ ERROR: No comment field found!")
        return False

    print(f"   ✓ Base64 comment: {comment_value[:50]}...")

    # Verify human-readable fields
    args_str = ' '.join(metadata_args)
    readable_fields = [
        'deforum_resolution=1280x720',
        'deforum_fps=60',
        'deforum_seed=123456789',
        'deforum_steps=20',
        'deforum_cfg_scale=7.5',
        'deforum_distilled_cfg=3.5',
        'deforum_model=flux1-dev-bnb-nf4-v2.safetensors',
        'deforum_sampler=euler_a',
        'deforum_scheduler=exponential',
        'deforum_batch_name=test_batch',
    ]

    for field in readable_fields:
        if field in args_str:
            print(f"   ✓ Human-readable: {field}")
        else:
            print(f"   ✗ Missing: {field}")

    # Step 3: Decode metadata (simulates extraction)
    print("\n3. Decoding metadata (simulates video extraction)...")
    try:
        decoded = decode_settings_from_metadata(comment_value)
        settings_decoded = decoded["settings"]

        # Verify key fields
        checks = [
            ("W", 1280),
            ("H", 720),
            ("seed", 123456789),
            ("steps", 20),
            ("cfg_scale", 7.5),
            ("distilled_cfg_scale", 3.5),
            ("fps", 60),
            ("max_frames", 240),
            ("render_mode", "New 3D"),
            ("sd_model_checkpoint", "flux1-dev-bnb-nf4-v2.safetensors"),
            ("batch_name", "test_batch"),
        ]

        all_good = True
        for key, expected in checks:
            actual = settings_decoded.get(key)
            if actual == expected:
                print(f"   ✓ {key}: {actual}")
            else:
                print(f"   ✗ {key}: expected {expected}, got {actual}")
                all_good = False

        # Check metadata fields
        if "commit_id" in settings_decoded:
            print(f"   ✓ commit_id: {settings_decoded['commit_id']}")
        else:
            print("   ✗ commit_id: missing")
            all_good = False

        if "github_url" in settings_decoded:
            print(f"   ✓ github_url: {settings_decoded['github_url']}")
        else:
            print("   ✗ github_url: missing")
            all_good = False

        if "fork_name" in settings_decoded:
            print(f"   ✓ fork_name: {settings_decoded['fork_name'][:50]}...")
        else:
            print("   ✗ fork_name: missing")
            all_good = False

        # Step 4: Verify privacy (no user-identifying info)
        print("\n4. Verifying privacy (no user-identifying info)...")
        privacy_checks = ['title=', 'artist=', 'copyright=', 'encoder=', 'description=']
        privacy_good = True
        for field in privacy_checks:
            if field in args_str:
                print(f"   ✗ WARNING: Found {field} in metadata (should not be present)")
                privacy_good = False

        if privacy_good:
            print("   ✓ No user-identifying fields found (privacy maintained)")

        print("\n" + "="*70)
        if all_good and privacy_good:
            print("✅ WORKFLOW TEST PASSED - All systems operational!")
        else:
            print("⚠️  WORKFLOW TEST COMPLETED WITH WARNINGS")
        print("="*70 + "\n")

        return all_good and privacy_good

    except Exception as e:
        print(f"\n   ✗ ERROR decoding metadata: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_full_metadata_workflow()
    sys.exit(0 if success else 1)
