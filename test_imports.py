#!/usr/bin/env python3
"""
Simple test script to check if Deforum imports work without circular dependency errors.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test key imports that were causing circular dependency issues"""
    
    print("Testing basic Python imports...")
    try:
        import json
        import pandas as pd
        import numpy as np
        print("✓ Basic dependencies available")
    except Exception as e:
        print(f"✗ Basic dependencies failed: {e}")
        return False
    
    print("Testing internal module structure...")
    try:
        # Test if we can import the modules without webui dependencies
        import deforum
        print("✓ Deforum package imported successfully")
    except Exception as e:
        print(f"✗ Deforum package import failed: {e}")
        return False
    
    print("Testing keyframe animation import...")
    try:
        # This should work without webui modules
        import deforum.core.keyframe_animation
        print("✓ Keyframe animation module imported successfully")
    except Exception as e:
        print(f"✗ Keyframe animation import failed: {e}")
        return False
    
    print("Testing schedule models import...")
    try:
        import deforum.models.schedule_models
        print("✓ Schedule models imported successfully")
    except Exception as e:
        print(f"✗ Schedule models import failed: {e}")
        return False
    
    print("Testing utils import...")
    try:
        # This will fail due to webui modules, but we can check for circular import specifically
        try:
            import deforum.utils
        except ImportError as e:
            if "circular import" in str(e).lower() or "cannot import name" in str(e).lower():
                print(f"✗ Circular import detected: {e}")
                return False
            else:
                print(f"✓ No circular import detected (expected webui module error: {e})")
        except Exception as e:
            if "circular import" in str(e).lower():
                print(f"✗ Circular import detected: {e}")
                return False
            else:
                print(f"✓ No circular import detected (other error: {e})")
    except Exception as e:
        print(f"✗ Utils import test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Testing Deforum imports for circular dependencies...")
    success = test_imports()
    if success:
        print("\n🎉 No circular import issues detected! The extension structure appears to be fixed.")
    else:
        print("\n❌ Circular import issues may still exist.")
    sys.exit(0 if success else 1) 