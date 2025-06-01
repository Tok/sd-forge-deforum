import os

import modules.paths as ph
from modules import script_callbacks
from modules.shared import cmd_opts
from deforum_extend_paths import deforum_sys_extend

print("🚀 Deforum extension entry point started...")

def init_deforum():
    print("🔧 Initializing Deforum extension...")
    
    # use sys.path.extend to make sure all of our files are available for importation
    deforum_sys_extend()
    print("✅ Path extension completed")

    # create the Models/Deforum folder, where many of the deforum related models/ packages will be downloaded
    os.makedirs(ph.models_path + '/Deforum', exist_ok=True)
    print("✅ Models/Deforum directory created")

    try:
        print("📦 Attempting to import UI functions...")
        
        # import our on_ui_tabs and on_ui_settings functions from the new package structure
        from deforum.ui.secondary_interface_panels import on_ui_tabs
        print("✅ Successfully imported on_ui_tabs")
        
        from deforum.ui.settings_interface import on_ui_settings
        print("✅ Successfully imported on_ui_settings")

        print("🎯 Registering callbacks with webui...")
        
        # trigger webui's extensions mechanism using our imported main functions -
        # first to create the actual deforum gui, then to make the deforum tab in webui's settings section
        script_callbacks.on_ui_tabs(on_ui_tabs)
        print("✅ UI tabs callback registered")
        
        script_callbacks.on_ui_settings(on_ui_settings)
        print("✅ UI settings callback registered")
        
        print("🎉 Deforum extension loaded successfully!")
        
    except ImportError as e:
        print(f"❌ Failed to load Deforum extension due to import error: {e}")
        print("Check that all required dependencies are installed and the package structure is correct.")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Failed to load Deforum extension due to unexpected error: {e}")
        import traceback
        traceback.print_exc()

print("🚀 About to call init_deforum()...")
init_deforum()
print("🏁 Deforum extension initialization completed.")

