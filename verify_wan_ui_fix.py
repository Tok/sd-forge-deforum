#!/usr/bin/env python3
"""
Verification script for Wan UI minimum inference steps fix
Run this after restarting WebUI to verify the fix is working
"""

def verify_wan_ui_fix():
    """Verify that Wan inference steps minimum is set to 5 in the UI"""
    print("🔍 Verifying Wan UI Fix...")
    print("=" * 50)
    
    try:
        # Test 1: Check args configuration
        print("1️⃣ Checking Wan args configuration...")
        
        # This will only work when run from within WebUI context
        try:
            from deforum_helpers.args import WanArgs  # type: ignore
            wan_args = WanArgs()
            inference_steps_config = wan_args['wan_inference_steps']
            
            print(f"📋 Wan Inference Steps Configuration:")
            print(f"   Minimum: {inference_steps_config['minimum']}")
            print(f"   Maximum: {inference_steps_config['maximum']}")
            print(f"   Default: {inference_steps_config['value']}")
            print(f"   Step: {inference_steps_config['step']}")
            
            if inference_steps_config['minimum'] == 5:
                print("✅ Args configuration correct: minimum = 5")
                args_correct = True
            else:
                print(f"❌ Args configuration wrong: minimum = {inference_steps_config['minimum']}, should be 5")
                args_correct = False
                
        except Exception as e:
            print(f"⚠️ Cannot test args (WebUI context required): {e}")
            args_correct = None
        
        # Test 2: Instructions for manual UI verification
        print("\n2️⃣ Manual UI Verification Steps:")
        print("   1. 🌐 Open WebUI in browser")
        print("   2. 📂 Go to Deforum extension")
        print("   3. 🎬 Click on 'Wan Video' tab")
        print("   4. 🔧 Look at 'Inference Steps' slider")
        print("   5. ✅ Verify minimum value shows as 5 (not 20)")
        
        print("\n3️⃣ Expected UI Behavior:")
        print("   ✅ Inference Steps slider should allow values from 5 to 100")
        print("   ✅ You should be able to set values like 5, 10, 15, etc.")
        print("   ❌ Values below 5 should not be selectable")
        
        print("\n4️⃣ If UI still shows minimum 20:")
        print("   🔄 Restart WebUI completely (stop and start)")
        print("   🧹 Clear browser cache (Ctrl+F5 or Ctrl+Shift+R)")
        print("   📱 Try in incognito/private browser window")
        
        if args_correct is True:
            print("\n✅ VERIFICATION PASSED")
            print("   Args configuration is correct (minimum = 5)")
            print("   If UI still shows 20, restart WebUI and clear browser cache")
        elif args_correct is False:
            print("\n❌ VERIFICATION FAILED")
            print("   Args configuration is wrong - this shouldn't happen!")
        else:
            print("\n⚠️ PARTIAL VERIFICATION")
            print("   Cannot test args outside WebUI context")
            print("   Please check UI manually after WebUI restart")
            
        return args_correct
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False

def main():
    """Main verification function"""
    print("🚀 Wan UI Fix Verification")
    print("Checking if inference steps minimum is correctly set to 5")
    print("=" * 60)
    
    result = verify_wan_ui_fix()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY:")
    print("   Issue: Wan inference steps minimum was 20, should be 5")
    print("   Fix: Updated args.py to set minimum = 5")
    print("   Status: Requires WebUI restart to take effect")
    print("\n🔧 NEXT STEPS:")
    print("   1. Restart WebUI completely")
    print("   2. Clear browser cache")
    print("   3. Check Wan tab - inference steps should allow 5-100")
    print("   4. Test setting inference steps to 5, 10, 15, etc.")
    
    return 0 if result is not False else 1

if __name__ == "__main__":
    exit(main()) 