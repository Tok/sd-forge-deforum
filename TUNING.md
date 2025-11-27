# Deforum Parameter Tuning

## Quick Start

**Launch Tuning Lab:**
```bash
# Linux/Mac
./run-tuning-lab.sh

# Windows
run-tuning-lab.bat
```

This starts Forge with the Deforum tuning tab where you can:
- Run parameter sweeps interactively
- See real-time results and metrics
- Compare visualizations
- Export optimal parameters

## Architecture

### Tuning UI (Interactive)
- **Location:** Tuning tab in WebUI (when `--deforum-run-tuning` flag is active)
- **Purpose:** Interactive parameter exploration with GUI
- **Usage:** Click "Run Tests", monitor progress, analyze results

### Integration Tests (Automated)
- **Location:** `tests/integration/`
- **Purpose:** API-based validation and CI/CD
- **Usage:** `pytest tests/integration/test_depth_warping_orbit_tuning.py --start-server`

### Tuning Tests (Legacy)
- **Location:** `tests/tuning/`  
- **Purpose:** Batch parameter sweeps for research
- **Usage:** `./run-tuning-tests.sh` (still works but use tuning UI instead)

## Depth Warping Orbit Tuning

**What it tests:** Optimal translation/rotation factors for orbital camera paths

**Parameters:**
- Aspect ratios: 16:9, 9:16, 1:1
- Rotation factors: -3.0 to -7.0
- 13 combinations total

**Via Tuning UI:**
1. `./launch-tuning.sh`
2. Navigate to Tuning tab
3. Select "Depth Warping Orbit Sweep"
4. Click "Run Tests"
5. Analyze results in charts/tables

**Via Integration Tests:**
```bash
pytest tests/integration/test_depth_warping_orbit_tuning.py::test_orbit_rotation_factor_sweep -v --start-server
```

## Output

All results save to: `outputs/deforum-tuning/`

```
outputs/deforum-tuning/
├── depth_warping_orbits/
│   ├── aspect178_512x288_factor5.0/
│   │   ├── metrics_{timestamp}.json
│   │   └── {timestamp}/
│   │       ├── *.png (20 frames)
│   │       └── depth-maps/*.png
│   └── ...
└── color_preservation/
    └── ...
```

## Workflow Comparison

| Task | Old Way | New Way |
|------|---------|---------|
| Run parameter sweep | `./run-tuning-tests.sh` | `./launch-tuning.sh` → Tuning tab |
| Single test validation | `pytest tests/tuning/` | `pytest tests/integration/` |
| Batch overnight runs | `./run-tuning-tests.sh` | Still works! |
| View results | Check JSON files | Interactive charts in UI |

## Migration Guide

**Before:**
```bash
./run-tuning-tests.sh tests/tuning/test_depth_warping_orbit_tuning.py
# Wait for completion
# Manually analyze JSON files
```

**After:**
```bash
./launch-tuning.sh
# Use Tuning tab → interactive results
```

Or for automated testing:
```bash
pytest tests/integration/test_depth_warping_orbit_tuning.py --start-server
```
