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

# Enable fastf1 cache
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

# ── Constants ─────────────────────────────────────────────────────────────────
AVAILABLE_YEARS = [2023, 2024]

SESSION_LABEL_MAP = {
    "Q": "Qualifying",
    "R": "Race",
    "Sprint": "Sprint",
    "SQ": "Sprint Shootout",
}

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

# ── Data lake output paths ────────────────────────────────────────────────────
MISTAKE_DATA_DIR = PROJECT_ROOT / "data_lake" / "mistake_data"
PROCESSED_VIDEO_DIR = PROJECT_ROOT / "data_lake" / "processed_video"
PROCESSED_CSV_DIR = PROJECT_ROOT / "data_lake" / "processed_csv"

# ── Team mapping (driver number → team name) ──────────────────────────────────
TEAM_MAPPING = {
    "1":  "Red Bull Racing",
    "2":  "Williams",
    "3":  "RB",
    "4":  "McLaren",
    "10": "Alpine",
    "11": "Red Bull Racing",
    "14": "Aston Martin",
    "16": "Ferrari",
    "18": "Aston Martin",
    "20": "Haas",
    "21": "AlphaTauri",
    "22": "RB",
    "23": "Williams",
    "24": "Kick Sauber",
    "27": "Haas",
    "31": "Alpine",
    "38": "Ferrari",
    "40": "RB",
    "43": "Williams",
    "44": "Mercedes",
    "55": "Ferrari",
    "63": "Mercedes",
    "77": "Kick Sauber",
    "81": "McLaren",
}

# ── Color constants ───────────────────────────────────────────────────────────
COLOR_CYAN   = "#00d4ff"
COLOR_GREEN  = "#00ff88"
COLOR_RED    = "#ff3a3a"
COLOR_AMBER  = "#ffb800"
COLOR_PURPLE = "#a855f7"
COLOR_MUTED  = "#6b7890"
COLOR_BG     = "#0a0c0f"
COLOR_PANEL  = "#1a2030"
COLOR_BORDER = "#ffffff12"
