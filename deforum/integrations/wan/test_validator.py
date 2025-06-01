#!/usr/bin/env python3

from wan_model_validator import WanModelValidator
from pathlib import Path

def test_wan_validator():
    print("🔍 Testing Wan Model Validator...")
    
    validator = WanModelValidator()
    models = validator.discover_models()
    
    print(f"\n✅ Found {len(models)} Wan models:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model['name']}")
        print(f"     Type: {model['type']}")
        print(f"     Size: {model['size_formatted']}")
        print(f"     Path: {model['path']}")
        
        # Test validation
        print(f"     Validating...")
        result = validator.validate_model_integrity(Path(model['path']))
        if result['valid']:
            print(f"     ✅ Status: VALID")
        else:
            print(f"     ❌ Status: INVALID")
            print(f"     Errors: {', '.join(result['errors'])}")
        print()
    
    print("🎯 Validation complete!")

if __name__ == "__main__":
    test_wan_validator() 