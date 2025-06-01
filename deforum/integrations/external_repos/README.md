# External Repositories

This directory contains original, unmodified repositories that are integrated into Deforum.

## ⚠️ IMPORTANT: DO NOT MODIFY THESE REPOSITORIES

The repositories in this directory should remain **exactly as they are** from their original sources. They are meant to be:

1. **1:1 copies** of the original GitHub repositories
2. **Unchanged** and **unrefactored** 
3. **Preserved** for future updates and compatibility
4. **Isolated** from Deforum's refactoring efforts

## Directory Structure

```
external_repos/
├── README.md                    # This file
├── wan2.1/                      # Original Wan 2.1 repository (DO NOT MODIFY)
│   ├── models/                  # Original Wan models
│   ├── pipelines/              # Original Wan pipelines
│   ├── configs/                # Original Wan configs
│   └── ...                     # All original Wan files
└── [future external repos]/    # Other external repositories as needed
```

## Integration Pattern

- **External Repository**: Stored unchanged in `external_repos/[repo_name]/`
- **Deforum Integration**: Wrapper/adapter code in `deforum/integrations/[name]/`
- **Clear Separation**: No mixing of original code with Deforum adaptations

## Adding New External Repositories

1. Place the **unchanged** repository in `external_repos/[repo_name]/`
2. Create integration wrapper in `deforum/integrations/[name]/`
3. Document the separation clearly
4. Never modify the original repository files

## Benefits

- **Easy Updates**: Can replace entire external repo when upstream updates
- **Clear Attribution**: Original code remains intact and identifiable
- **Reduced Conflicts**: Deforum refactoring doesn't break external dependencies
- **Legal Clarity**: Original licenses and code remain unmodified 