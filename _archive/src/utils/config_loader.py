"""
Konfigürasyon dosyasını yükleyen modül
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    YAML konfigürasyon dosyasını yükler.
    
    Args:
        config_path: Konfigürasyon dosyasının yolu. None ise varsayılan yol kullanılır.
    
    Returns:
        Konfigürasyon sözlüğü
    """
    if config_path is None:
        # Proje kök dizinini bul
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Konfigürasyon dosyası bulunamadı: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def get_output_path(config: Dict[str, Any], file_key: str) -> Path:
    """
    Çıktı dosyasının tam yolunu döndürür.
    
    Args:
        config: Konfigürasyon sözlüğü
        file_key: output.files içindeki anahtar
    
    Returns:
        Dosya yolu
    """
    project_root = Path(__file__).parent.parent.parent
    base_dir = config.get('output', {}).get('base_dir', 'outputs')
    file_name = config.get('output', {}).get('files', {}).get(file_key, file_key)
    
    # Dosya tipine göre dizin belirle
    if file_key.endswith('_csv') or file_key.endswith('_csv'):
        subdir = config.get('output', {}).get('results_dir', 'outputs/results')
    elif file_key.endswith('_pkl') or file_key.endswith('_npy'):
        subdir = config.get('output', {}).get('processed_data_dir', 'outputs/processed')
    else:
        subdir = base_dir
    
    return project_root / subdir / file_name

