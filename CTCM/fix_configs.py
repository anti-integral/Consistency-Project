#!/usr/bin/env python3
"""Fix missing training.mode in config files."""

import os
import yaml
from pathlib import Path

def fix_config_file(filepath):
    """Add missing training.mode to config file."""
    print(f"Checking {filepath}...")

    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)

    # Check if training.mode exists
    if 'training' in config and 'mode' not in config['training']:
        print(f"  - Adding missing training.mode to {filepath}")
        config['training']['mode'] = 'consistency_training'

        # Write back
        with open(filepath, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"  - Fixed!")
    elif 'training' in config and 'mode' in config['training']:
        print(f"  - training.mode already exists: {config['training']['mode']}")
    else:
        print(f"  - No training section found")

def main():
    """Fix all config files."""
    config_dir = Path('configs')

    # Fix existing configs
    for config_file in config_dir.glob('*.yaml'):
        fix_config_file(config_file)

    # Also create the fixed cifar10_ctcm.yaml if it doesn't have mode
    cifar10_ctcm_path = config_dir / 'cifar10_ctcm.yaml'
    if cifar10_ctcm_path.exists():
        fix_config_file(cifar10_ctcm_path)

    print("\nAll configs checked and fixed!")

if __name__ == '__main__':
    main()