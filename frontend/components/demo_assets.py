"""Small, versioned public telemetry bundle; no startup network downloads."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

BUNDLE_DIR = Path(__file__).resolve().parents[2] / "demo_data"
SESSION_NAME = re.compile(r"\d{4}_\d+_(?:Q|R|Sprint|SQ)\.parquet")


def validate_bundle(bundle_dir: Path) -> dict:
    """Validate all bundled bytes before use, including confinement to the bundle."""
    root = bundle_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("Unsupported or empty demo manifest")
    seen = set()
    for entry in manifest["files"]:
        relative = entry["path"]
        path = (root / relative).resolve()
        if (path.parent != root / "clean_data" or not path.is_relative_to(root)
                or not SESSION_NAME.fullmatch(path.name) or relative in seen):
            raise ValueError("Invalid demo asset path")
        seen.add(relative)
        if (path.stat().st_size != entry["bytes"]
                or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]):
            raise ValueError(f"Demo integrity check failed: {path.name}")
        if path.stem not in manifest["sessions"]:
            raise ValueError(f"Demo session metadata missing: {path.name}")
    return manifest


@lru_cache(maxsize=1)
def demo_manifest() -> dict:
    if not (BUNDLE_DIR / "manifest.json").exists():
        return {}
    return validate_bundle(BUNDLE_DIR)


def session_files(local_dir: Path) -> list[Path]:
    """Use the existing data lake when present; otherwise use the complete demo."""
    local = sorted(p for p in local_dir.glob("*.parquet") if SESSION_NAME.fullmatch(p.name))
    if local:
        return local
    return [BUNDLE_DIR / entry["path"] for entry in demo_manifest().get("files", [])]


def driver_metadata(year: int, round_num: int, session_type: str, driver: str) -> dict:
    session = demo_manifest().get("sessions", {}).get(f"{year}_{round_num}_{session_type}", {})
    return session.get("drivers", {}).get(str(driver), {})
