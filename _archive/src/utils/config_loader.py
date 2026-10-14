"""
Module that loads the configuration file
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Loads the YAML configuration file.
    
    Args:
        config_path: Path to the configuration file. If None, the default path is used.
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Find the project root directory
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_output_path(config: Dict[str, Any], file_key: str) -> Path:
    """
    Returns the full path of an output file.
    
    Args:
        config: Configuration dictionary
        file_key: Key in output.files
    
    Returns:
        File path
    """
    project_root = Path(__file__).parent.parent.parent
    base_dir = config.get('output', {}).get('base_dir', 'outputs')
    file_name = config.get('output', {}).get('files', {}).get(file_key, file_key)
    
    # Choose the directory by file type
    if file_key.endswith('_csv') or file_key.endswith('_csv'):
        subdir = config.get('output', {}).get('results_dir', 'outputs/results')
    elif file_key.endswith('_pkl') or file_key.endswith('_npy'):
        subdir = config.get('output', {}).get('processed_data_dir', 'outputs/processed')
    else:
        subdir = base_dir
    
    return project_root / subdir / file_name

