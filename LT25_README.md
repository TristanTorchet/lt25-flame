# LT25 Flame Installation Guide

This guide covers the installation process for the LT25 version of Flame, which uses a custom fork of flash-linear-attention.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/TristanTorchet/lt25-flame.git
cd flame
```

2. Add the custom flash-linear-attention fork as a submodule:
```bash
# Then add the submodule
git submodule add git@github.com:TristanTorchet/lt25-flash-linear-attention.git flash-linear-attention
```

3. Initialize and update the submodule:
```bash
git submodule update --init --recursive
```

4. Sync the project dependencies (this will also install torchtitan):
```bash
uv sync
```

6. Force reinstall the local flash-linear-attention to ensure it overrides the pinned version:
```bash
uv pip install -e ./flash-linear-attention --force-reinstall
```

7. Modify permissions for .sh files to make them executable:
```bash
chmod +x launch_training.sh
chmod +x train.sh
chmod +x train_timeseries.sh
```

## Verification

After installation, verify that the correct version of flash-linear-attention is installed:
```bash
uv pip list | grep flash-linear-attention
```

You should see version 0.3.0 from your local fork, not the pinned 0.2.2 version.

## Notes

- The custom fork at `git@github.com:TristanTorchet/lt25-flash-linear-attention.git` contains LT25-specific modifications
- The submodule approach ensures reproducible builds and easy updates to the fork
- The `--force-reinstall` flag in step 7 is necessary because `uv sync` will try to downgrade to the pinned version in pyproject.toml