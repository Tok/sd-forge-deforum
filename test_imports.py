#!/usr/bin/env python3
"""
Test script to check if our refactored imports work correctly.
"""

import sys
import os

# Add the extension path to sys.path
extension_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, extension_path)

# Add the webui path to sys.path so we can import modules
webui_path = os.path.join(os.path.dirname(extension_path), "..", "..")
sys.path.insert(0, webui_path)

def test_imports():
    print("Testing imports...")
    print(f"Extension path: {extension_path}")
    print(f"WebUI path: {webui_path}")
    
    try:
        # Test the main entry point imports directly
        print("Testing direct imports...")
        
        from deforum.ui.secondary_interface_panels import on_ui_tabs
        print("✅ Successfully imported on_ui_tabs")
        
        from deforum.ui.settings_interface import on_ui_settings
        print("✅ Successfully imported on_ui_settings")
        
        print("\n🎉 All critical imports successful! The extension should load properly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1) 