⚠️  FORGE EXTENSION ENTRY POINT - DO NOT ADD NEW FILES HERE  ⚠️

This directory is the EXTENSION ENTRY POINT for Stable Diffusion WebUI Forge.

Forge scans this directory to discover and load extensions. Only the following
files should exist here:

  ✅ deforum.py              - Main extension registration
  ✅ deforum_helpers/        - Core implementation subdirectory
  ✅ deforum_extend_paths.py - Python path setup

❌ DO NOT add utility scripts, tools, or other files here!

For development tools, use:      dev-tools/
For shell launcher scripts, use: shell_scripts/
For API endpoints, use:          deforum/api/
For core logic, use:             deforum/

Adding files here can cause conflicts with Forge's extension loading system
and may break the extension or interfere with other extensions.

If you need to add a new Python module, put it in:
  - deforum/           (for core functionality)
  - deforum/api/       (for API endpoints)
  - deforum/ui/        (for UI components)
  - deforum/utils/     (for utilities)
  - dev-tools/         (for development scripts)

This directory structure follows Forge extension conventions and must be
maintained for proper extension loading.
