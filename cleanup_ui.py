#!/usr/bin/env python3
"""
Cleanup script to remove FreeU, Kohya HR Fix, and Hybrid Video functions from ui_elements.py
Also updates naming throughout to use "Zirteqs Deforum Fork"
"""

import re

def main():
    # Read the original file
    with open('scripts/deforum_helpers/ui_elements.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("🧹 Cleaning up ui_elements.py...")
    
    # Remove get_tab_freeu function
    freeu_pattern = r'def get_tab_freeu\(dfu: SimpleNamespace\):.*?return \{k: v for k, v in \{\*\*locals\(\), \*\*vars\(\)\}\.items\(\)\}\s*\n\s*\n'
    content = re.sub(freeu_pattern, '', content, flags=re.DOTALL)
    print("✅ Removed get_tab_freeu function")
    
    # Remove get_tab_kohya_hrfix function
    kohya_pattern = r'def get_tab_kohya_hrfix\(dku: SimpleNamespace\):.*?return \{k: v for k, v in \{\*\*locals\(\), \*\*vars\(\)\}\.items\(\)\}\s*\n\s*\n'
    content = re.sub(kohya_pattern, '', content, flags=re.DOTALL)
    print("✅ Removed get_tab_kohya_hrfix function")
    
    # Remove get_tab_hybrid function
    hybrid_pattern = r'def get_tab_hybrid\(da\):.*?return \{k: v for k, v in \{\*\*locals\(\), \*\*vars\(\)\}\.items\(\)\}\s*\n\s*\n'
    content = re.sub(hybrid_pattern, '', content, flags=re.DOTALL)
    print("✅ Removed get_tab_hybrid function")
    
    # Update references to "Deforum" to "Zirteqs Deforum Fork" in comments and strings
    content = re.sub(r'# Copyright \(C\) 2023 Deforum LLC', '# Copyright (C) 2023 Deforum LLC\n# Modified by Zirteq for Zirteqs Deforum Fork', content)
    
    # Write the cleaned content back
    with open('scripts/deforum_helpers/ui_elements.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ Cleanup completed!")
    print("🎯 Next: Update references in ui_left.py to remove missing functions")

if __name__ == "__main__":
    main() 