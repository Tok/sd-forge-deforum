#!/usr/bin/env python3
"""
Test script for enhanced Wan integration with QwenPromptExpander and movement analysis
"""

import sys
import os
sys.path.append('scripts')

from types import SimpleNamespace

def test_qwen_manager():
    """Test QwenModelManager functionality"""
    print("🧠 Testing QwenModelManager...")
    
    try:
        from scripts.deforum_helpers.wan.utils.qwen_manager import qwen_manager
        
        # Test model info
        print("\n📊 Available models:")
        for model_name in qwen_manager.MODEL_SPECS.keys():
            info = qwen_manager.get_model_info(model_name)
            downloaded = qwen_manager.is_model_downloaded(model_name)
            print(f"  {model_name}: {info.get('description', 'N/A')} ({'✅' if downloaded else '❌'})")
        
        # Test auto-selection
        auto_selected = qwen_manager.auto_select_model()
        print(f"\n🤖 Auto-selected model: {auto_selected}")
        
        # Test VRAM detection
        vram = qwen_manager.get_available_vram()
        print(f"💾 Available VRAM: {vram:.1f}GB")
        
        print("✅ QwenModelManager test passed")
        return True
        
    except Exception as e:
        print(f"❌ QwenModelManager test failed: {e}")
        return False

def test_movement_analyzer():
    """Test MovementAnalyzer functionality"""
    print("\n📐 Testing MovementAnalyzer...")
    
    try:
        from scripts.deforum_helpers.wan.utils.movement_analyzer import analyze_deforum_movement
        
        # Create mock animation arguments
        anim_args = SimpleNamespace()
        anim_args.angle = "0: (0)"
        anim_args.zoom = "0: (1.0), 50: (1.5)"  # Zoom in over 50 frames
        anim_args.translation_x = "0: (0), 30: (100)"  # Pan right
        anim_args.translation_y = "0: (0)"
        anim_args.translation_z = "0: (0), 60: (50)"  # Move forward
        anim_args.rotation_3d_x = "0: (0)"
        anim_args.rotation_3d_y = "0: (0), 40: (15)"  # Yaw right
        anim_args.rotation_3d_z = "0: (0)"
        anim_args.perspective_flip_theta = "0: (0)"
        anim_args.perspective_flip_phi = "0: (0)"
        anim_args.perspective_flip_gamma = "0: (0)"
        anim_args.perspective_flip_fv = "0: (53)"
        anim_args.max_frames = 100
        
        # Analyze movement
        movement_desc, motion_strength = analyze_deforum_movement(
            anim_args=anim_args,
            sensitivity=1.0,
            max_frames=100
        )
        
        print(f"📝 Movement description: {movement_desc}")
        print(f"🎬 Motion strength: {motion_strength:.2f}")
        
        # Test with different sensitivity
        movement_desc_low, motion_strength_low = analyze_deforum_movement(
            anim_args=anim_args,
            sensitivity=0.5,
            max_frames=100
        )
        
        print(f"📝 Low sensitivity description: {movement_desc_low}")
        print(f"🎬 Low sensitivity motion strength: {motion_strength_low:.2f}")
        
        print("✅ MovementAnalyzer test passed")
        return True
        
    except Exception as e:
        print(f"❌ MovementAnalyzer test failed: {e}")
        return False

def test_prompt_enhancement():
    """Test prompt enhancement functionality"""
    print("\n🎨 Testing prompt enhancement...")
    
    try:
        from scripts.deforum_helpers.wan.utils.qwen_manager import qwen_manager
        
        # Test prompts
        test_prompts = {
            "0": "a serene beach at sunset",
            "60": "a misty forest in the morning",
            "120": "a bustling city street at night"
        }
        
        print("📝 Original prompts:")
        for frame, prompt in test_prompts.items():
            print(f"  Frame {frame}: {prompt}")
        
        # Test without actual model (will fail gracefully)
        enhanced_prompts = qwen_manager.enhance_prompts(
            prompts=test_prompts,
            model_name="Auto-Select",
            language="English",
            auto_download=False  # Don't actually download for test
        )
        
        print("🎨 Enhanced prompts:")
        for frame, prompt in enhanced_prompts.items():
            print(f"  Frame {frame}: {prompt[:100]}...")
        
        print("✅ Prompt enhancement test passed (graceful fallback)")
        return True
        
    except Exception as e:
        print(f"❌ Prompt enhancement test failed: {e}")
        return False

def test_arguments_integration():
    """Test that new arguments are properly integrated"""
    print("\n⚙️ Testing arguments integration...")
    
    try:
        from scripts.deforum_helpers.args import WanArgs
        
        wan_args = WanArgs()
        
        # Check new prompt enhancement arguments
        enhancement_args = [
            'wan_enable_prompt_enhancement', 'wan_qwen_model', 'wan_qwen_auto_download',
            'wan_qwen_language', 'wan_enhanced_prompts', 'wan_enable_movement_analysis',
            'wan_movement_description', 'wan_movement_sensitivity', 'wan_motion_strength_override'
        ]
        
        missing_args = []
        for arg in enhancement_args:
            if arg not in wan_args:
                missing_args.append(arg)
        
        if missing_args:
            print(f"❌ Missing arguments: {missing_args}")
            return False
        
        print("✅ All new arguments present")
        
        # Check default values
        print("📋 Default values:")
        for arg in enhancement_args:
            value = wan_args[arg]['value']
            print(f"  {arg}: {value}")
        
        print("✅ Arguments integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Arguments integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Enhanced Wan Integration")
    print("=" * 50)
    
    tests = [
        test_arguments_integration,
        test_qwen_manager,
        test_movement_analyzer,
        test_prompt_enhancement,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Enhanced Wan integration is ready.")
    else:
        print("⚠️ Some tests failed. Check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 