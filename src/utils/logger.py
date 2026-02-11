"""
Logging sistemi
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "tubitak_proje",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> logging.Logger:
    """
    Logger kurulumu yapar.
    
    Args:
        name: Logger adı
        log_level: Log seviyesi (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Log dosyası yolu (None ise dosyaya yazmaz)
        console: Konsola yazdırma (True/False)
        log_format: Log formatı
    
    Returns:
        Yapılandırılmış logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Mevcut handler'ları temizle
    logger.handlers.clear()
    
    # Format oluştur
    formatter = logging.Formatter(log_format)
    
    # Konsol handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Dosya handler
    if log_file:
        # Log dizinini oluştur
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

