#!/usr/bin/env python3
"""Diagnostic script to check which version of ui_right.py is being loaded."""

import sys
import os

# Add paths like Forge does
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Try to import the module
    from deforum.ui import ui_right

    # Check the source file
    source_file = ui_right.__file__
    print(f"✓ Module loaded from: {source_file}")

    # Check if it's a .pyc file
    if source_file.endswith('.pyc'):
        print(f"⚠ Loading from bytecode cache!")
        print(f"  Delete: {source_file}")
    else:
        print(f"✓ Loading from source (.py file)")

    # Read the actual file
    with open(source_file, 'r') as f:
        content = f.read()
        if 'slopcore_css = f"""' in content:
            print("✓ CSS string found in loaded file")
            # Check line 191
            lines = content.split('\n')
            if len(lines) > 190:
                print(f"  Line 191: {lines[190][:60]}...")
        else:
            print("✗ CSS string NOT found in loaded file!")

except Exception as e:
    print(f"✗ Error loading module: {e}")
    import traceback
    traceback.print_exc()
