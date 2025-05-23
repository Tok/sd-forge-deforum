# Copyright (C) 2023 Deforum LLC
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

# Contact the authors: https://deforum.github.io/

"""
Wan 2.1 Integration Test and Validation Script
"""

def test_wan_integration():
    """
    Test basic Wan integration components
    """
    print("=== Testing Wan 2.1 Integration ===")
    
    try:
        # Test 1: Check if WanArgs can be imported and created
        print("Test 1: WanArgs import and creation...")
        from .args import WanArgs
        wan_args_dict = WanArgs()
        assert wan_args_dict is not None
        assert 'wan_enabled' in wan_args_dict
        assert 'wan_model_path' in wan_args_dict
        print("✓ WanArgs creation successful")
        
        # Test 2: Check if WanVideoGenerator can be imported
        print("\nTest 2: WanVideoGenerator import...")
        from .wan_integration import WanVideoGenerator, WanPromptScheduler, validate_wan_settings
        print("✓ Wan integration modules imported successfully")
        
        # Test 3: Check if Wan UI elements can be imported
        print("\nTest 3: Wan UI import...")
        from .ui_elements import get_tab_wan
        print("✓ Wan UI elements imported successfully")
        
        # Test 4: Check if emoji is available
        print("\nTest 4: Wan emoji...")
        from .rendering.util.emoji_utils import wan_video
        emoji = wan_video()
        print(f"✓ Wan emoji available: {emoji}")
        
        # Test 5: Check if HTML info is available
        print("\nTest 5: Wan HTML info...")
        from .defaults import get_gradio_html
        html_info = get_gradio_html('wan_video')
        assert 'Wan 2.1' in html_info
        print("✓ Wan HTML info available")
        
        # Test 6: Check animation mode choices
        print("\nTest 6: Animation mode choices...")
        from .args import DeforumAnimArgs
        anim_args = DeforumAnimArgs()
        choices = anim_args['animation_mode']['choices']
        assert 'Wan Video' in choices
        print(f"✓ Animation mode choices: {choices}")
        
        # Test 7: Test validation function
        print("\nTest 7: Wan validation...")
        from types import SimpleNamespace
        test_wan_args = SimpleNamespace(**{key: value['value'] if isinstance(value, dict) else value 
                                         for key, value in wan_args_dict.items()})
        
        # Test with default (disabled) settings
        errors = validate_wan_settings(test_wan_args)
        print(f"✓ Validation function works (errors with disabled Wan: {len(errors)})")
        
        print("\n=== All Integration Tests Passed! ===")
        return True
        
    except ImportError as e:
        print(f"✗ Import Error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def get_wan_integration_status():
    """
    Get the current status of Wan integration
    """
    status = {
        'phase_1_foundation': {
            'wan_args': False,
            'wan_integration_module': False,
            'wan_ui_tab': False,
            'wan_defaults': False,
            'animation_mode_added': False
        },
        'phase_2_core_generation': {
            'render_wan_module': False,
            'run_deforum_dispatch': False,
            'emoji_support': False
        },
        'phase_3_integration': {
            'ui_left_integration': False,
            'prompt_scheduling': False,
            'frame_continuity': False
        }
    }
    
    try:
        # Check Phase 1
        from .args import WanArgs
        status['phase_1_foundation']['wan_args'] = True
        
        from .wan_integration import WanVideoGenerator
        status['phase_1_foundation']['wan_integration_module'] = True
        
        from .ui_elements import get_tab_wan
        status['phase_1_foundation']['wan_ui_tab'] = True
        
        from .defaults import get_wan_video_info_html
        status['phase_1_foundation']['wan_defaults'] = True
        
        from .args import DeforumAnimArgs
        anim_args = DeforumAnimArgs()
        if 'Wan Video' in anim_args['animation_mode']['choices']:
            status['phase_1_foundation']['animation_mode_added'] = True
        
        # Check Phase 2
        try:
            from .render_wan import render_wan_animation
            status['phase_2_core_generation']['render_wan_module'] = True
        except:
            pass
            
        # Check run_deforum dispatch (by reading file content)
        try:
            import os
            current_dir = os.path.dirname(__file__)
            run_deforum_path = os.path.join(current_dir, 'run_deforum.py')
            with open(run_deforum_path, 'r') as f:
                content = f.read()
                if "elif anim_args.animation_mode == 'Wan Video':" in content:
                    status['phase_2_core_generation']['run_deforum_dispatch'] = True
        except:
            pass
            
        from .rendering.util.emoji_utils import wan_video
        status['phase_2_core_generation']['emoji_support'] = True
        
        # Check Phase 3
        try:
            from .ui_left import setup_deforum_left_side_ui
            # Check if WanArgs is imported in ui_left
            import os
            current_dir = os.path.dirname(__file__)
            ui_left_path = os.path.join(current_dir, 'ui_left.py')
            with open(ui_left_path, 'r') as f:
                content = f.read()
                if 'WanArgs' in content and 'get_tab_wan' in content:
                    status['phase_3_integration']['ui_left_integration'] = True
        except:
            pass
            
        from .wan_integration import WanPromptScheduler
        status['phase_3_integration']['prompt_scheduling'] = True
        
        # Frame continuity is implemented in render_wan.py
        if status['phase_2_core_generation']['render_wan_module']:
            status['phase_3_integration']['frame_continuity'] = True
            
    except Exception as e:
        print(f"Error checking integration status: {e}")
    
    return status


def print_integration_report():
    """
    Print a comprehensive integration report
    """
    print("\n" + "="*60)
    print("WAN 2.1 INTEGRATION STATUS REPORT")
    print("="*60)
    
    status = get_wan_integration_status()
    
    def print_phase(phase_name, phase_data):
        completed = sum(1 for v in phase_data.values() if v)
        total = len(phase_data)
        percentage = (completed / total) * 100
        
        status_emoji = "✅" if completed == total else "🔄" if completed > 0 else "❌"
        print(f"\n{status_emoji} {phase_name}: {completed}/{total} ({percentage:.1f}%)")
        
        for item, done in phase_data.items():
            item_status = "✓" if done else "✗"
            item_name = item.replace('_', ' ').title()
            print(f"  {item_status} {item_name}")
    
    for phase_name, phase_data in status.items():
        phase_display_name = phase_name.replace('_', ' ').title()
        print_phase(phase_display_name, phase_data)
    
    # Calculate overall progress
    all_items = []
    for phase_data in status.values():
        all_items.extend(phase_data.values())
    
    total_completed = sum(1 for v in all_items if v)
    total_items = len(all_items)
    overall_percentage = (total_completed / total_items) * 100
    
    print(f"\n" + "="*60)
    print(f"OVERALL PROGRESS: {total_completed}/{total_items} ({overall_percentage:.1f}%)")
    
    if overall_percentage == 100:
        print("🎉 WAN 2.1 INTEGRATION COMPLETE!")
    elif overall_percentage >= 75:
        print("🚀 WAN 2.1 INTEGRATION NEARLY COMPLETE!")
    elif overall_percentage >= 50:
        print("⚡ WAN 2.1 INTEGRATION IN PROGRESS...")
    else:
        print("🔧 WAN 2.1 INTEGRATION STARTING...")
    
    print("="*60)


if __name__ == "__main__":
    # Run tests if this script is executed directly
    print_integration_report()
    test_wan_integration()
