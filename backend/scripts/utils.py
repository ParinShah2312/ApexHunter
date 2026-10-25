"""Shared utilities, paths, and logging configuration for all ApexHunter backend scripts."""

import json
import logging
import sys
from datetime import timezone, timedelta
from pathlib import Path

# ── Timezone ──────────────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_LAKE_DIR = PROJECT_ROOT / "data_lake"
CACHE_DIR = PROJECT_ROOT / "cache"
CONFIG_FILE = PROJECT_ROOT / "backend" / "config.json"

try:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        CONFIG = json.load(f)
except Exception as e:
    print(f"Failed to load config from {CONFIG_FILE}: {e}")
    CONFIG = {}

# ── Directory Management ──────────────────────────────────────────────────────
DATA_LAKE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def enable_fastf1_cache() -> Path:
    """Point FastF1 at the single project cache (PROJECT_ROOT/cache) and return its path.

    Every backend script that calls FastF1 must use this. Without it FastF1 silently falls back
    to its own OS-level default directory, so downloads are duplicated outside the project.
    fastf1 is imported lazily because most scripts that import utils never need it.
    """
    import fastf1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    return CACHE_DIR


class _SafeStreamHandler(logging.StreamHandler):
    """StreamHandler that never loses a log line to the stream's encoding.

    When stdout is redirected on Windows it uses cp1252, which cannot encode characters such as
    '→'; the standard handler then prints a "Logging error" traceback and drops the message.
    Here unencodable characters are replaced (shown as '?') instead.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = self.format(record) + self.terminator
            try:
                self.stream.write(text)
            except UnicodeEncodeError:
                encoding = getattr(self.stream, "encoding", None) or "ascii"
                self.stream.write(text.encode(encoding, errors="replace").decode(encoding))
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(name: str) -> logging.Logger:
    """Sets up a standardized logger for backend scripts."""
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if the logger is already configured
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Create console handler
        console_handler = _SafeStreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Define formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
