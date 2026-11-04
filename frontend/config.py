"""
ApexHunter Frontend - Configuration
Centralized paths, constants, and driver mappings.
"""

from pathlib import Path

import fastf1

# ── Paths ─────────────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent
PROJECT_ROOT = FRONTEND_DIR.parent
DATA_LAKE_DIR = PROJECT_ROOT / "data_lake" / "clean_data"
CACHE_DIR = PROJECT_ROOT / "cache"

MISTAKE_DATA_DIR = PROJECT_ROOT / "data_lake" / "mistake_data"
PROCESSED_VIDEO_DIR = PROJECT_ROOT / "data_lake" / "processed_video"
PROCESSED_CSV_DIR = PROJECT_ROOT / "data_lake" / "processed_csv"
RACING_LINES_DIR = PROJECT_ROOT / "data_lake" / "racing_lines"
TYRE_PREDICTIONS_DIR = PROJECT_ROOT / "data_lake" / "tyre_predictions"
MODELS_DIR = PROJECT_ROOT / "models"

try:
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
except Exception:
    pass   # cache is optional — dashboard works without it

# ── Session constants ─────────────────────────────────────────────────────────
AVAILABLE_YEARS = [2023, 2024]

SESSION_LABEL_MAP = {
    "Q": "Qualifying",
    "R": "Race",
    "Sprint": "Sprint",
    "SQ": "Sprint Shootout",
}

# ── Driver mapping ────────────────────────────────────────────────────────────
DRIVER_MAPPING = {
    "1": "Max Verstappen",
    "2": "Logan Sargeant",
    "3": "Daniel Ricciardo",
    "4": "Lando Norris",
    "10": "Pierre Gasly",
    "11": "Sergio Perez",
    "14": "Fernando Alonso",
    "16": "Charles Leclerc",
    "18": "Lance Stroll",
    "20": "Kevin Magnussen",
    "21": "Nyck de Vries",
    "22": "Yuki Tsunoda",
    "23": "Alexander Albon",
    "24": "Zhou Guanyu",
    "27": "Nico Hulkenberg",
    "31": "Esteban Ocon",
    "38": "Oliver Bearman",
    "40": "Liam Lawson",
    "43": "Franco Colapinto",
    "44": "Lewis Hamilton",
    "55": "Carlos Sainz",
    "63": "George Russell",
    "77": "Valtteri Bottas",
    "81": "Oscar Piastri",
}
