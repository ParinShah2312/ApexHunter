import logging
import sys
import json
from pathlib import Path

# Common Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_LAKE_DIR = PROJECT_ROOT / "data_lake"
CACHE_DIR = PROJECT_ROOT / "cache"
CONFIG_FILE = PROJECT_ROOT / "backend" / "config.json"

# Load config
try:
    with open(CONFIG_FILE, "r") as f:
        CONFIG = json.load(f)
except Exception as e:
    print(f"Failed to load config from {CONFIG_FILE}: {e}")
    CONFIG = {}

# Ensure common directories exist
DATA_LAKE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(name: str) -> logging.Logger:
    """Sets up a standardized logger for backend scripts."""
    logger = logging.getLogger(name)
    
    # Avoid adding multiple handlers if the logger is already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Define formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger
