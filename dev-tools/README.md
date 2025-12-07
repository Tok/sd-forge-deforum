# Development Tools

Utility scripts for development, testing, and maintenance.

**⚠️ These scripts are NOT part of the extension itself** - they're development tools only.

## Scripts

### generate_test_dataset.py
Generate realistic test images with ZIT for DA3-3DGS parameter tuning.

```bash
# Generate default dataset (20 interior images)
python dev-tools/generate_test_dataset.py

# Custom dataset
python dev-tools/generate_test_dataset.py \
  --num-images 30 \
  --prompt "modern office interior with windows" \
  --dataset-name office-scene
```

**Output:** `output/deforum-tuning/test-datasets/{dataset_name}/`

### migrate_prints_to_logger.py
Automated migration tool to convert print() statements to logger calls.

```bash
python dev-tools/migrate_prints_to_logger.py <file.py>
```

Converts:
```python
print("✨ Starting generation...")
```

To:
```python
logger.info("Starting generation...", emoji="sparkles")
```

### test_logger_themes.py
Test script to preview all logger themes and emoji support.

```bash
python dev-tools/test_logger_themes.py
```

Shows output examples for all theme modes (colorful, neutral, minimal).

## Directory Structure

```
scripts/              - Extension entry point (Forge loads from here)
├── deforum.py        - Main extension registration
├── deforum_helpers/  - Core implementation
└── deforum_extend_paths.py

dev-tools/            - Development utilities (NOT loaded by Forge)
├── generate_test_dataset.py
├── migrate_prints_to_logger.py
└── test_logger_themes.py

shell_scripts/        - Shell/batch launcher scripts
├── run-tuning-lab.sh
└── download-all-models.sh
```

**Important:** Only files in `scripts/` are loaded by Forge. Development tools live in `dev-tools/`.
