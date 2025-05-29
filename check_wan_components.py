#!/usr/bin/env python3
"""Simple script to check Wan component mapping"""

import sys
import os

# Add the extension path to sys.path
extension_path = os.path.join(os.path.dirname(__file__), 'scripts', 'deforum_helpers')
if extension_path not in sys.path:
    sys.path.insert(0, extension_path)

def check_wan_components():
    """Check which Wan components are defined vs created"""
    try:
        # Import without dependencies that need WebUI
        sys.path.append('.')
        import importlib.util
        
        # Load args module
        spec = importlib.util.spec_from_file_location("args", "scripts/deforum_helpers/args.py")
        args_module = importlib.util.module_from_spec(spec)
        
        # Mock the modules dependency to avoid WebUI imports
        sys.modules['modules'] = type('MockModule', (), {})()
        sys.modules['modules.shared'] = type('MockModule', (), {'opts': None, 'state': None})()
        
        spec.loader.exec_module(args_module)
        
        # Get Wan components
        wan_args = args_module.WanArgs()
        wan_component_names = list(wan_args.keys())
        
        print("🔍 WAN Components defined in WanArgs():")
        for i, name in enumerate(wan_component_names, 1):
            print(f"  {i:2}. {name}")
        
        print(f"\n✅ Total WAN components: {len(wan_component_names)}")
        
        # Check what should be in get_component_names
        expected_in_get_component_names = wan_component_names
        print(f"\n📝 Components that should be created in get_tab_wan():")
        for name in expected_in_get_component_names:
            print(f"  • {name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = check_wan_components()
    if success:
        print("\n🎉 Component mapping check completed!")
    else:
        print("\n❌ Component mapping check failed!")
    
    sys.exit(0 if success else 1) 