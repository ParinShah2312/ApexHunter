"""Project-wide configuration guards: single FastF1 cache location and complete requirements.txt."""

import ast
import sys
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPTS_DIR))
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
CODE_DIRS = [PROJECT_ROOT / "backend", PROJECT_ROOT / "frontend"]

import utils

# Top-level names that are this project's own modules/packages, not pip packages
LOCAL_MODULES = {
    "utils", "components", "config", "cv", "etl", "bigdata", "lstm", "racing_line",
    "isolation_forest", "detect_mistakes", "train_lstm", "predict_cliff", "optimal_line",
    "conftest", "run_tests", "integration",
}
# import name → pip distribution name, where they differ
IMPORT_TO_PACKAGE = {
    "cv2": "opencv-python", "yt_dlp": "yt-dlp", "sklearn": "scikit-learn", "PIL": "pillow",
}


def _python_files():
    for d in CODE_DIRS:
        yield from (p for p in d.rglob("*.py") if "__pycache__" not in p.parts)


class TestFastF1Cache(unittest.TestCase):

    def test_enable_fastf1_cache_uses_project_cache_dir(self) -> None:
        with mock.patch("fastf1.Cache.enable_cache") as enable:
            path = utils.enable_fastf1_cache()
        self.assertEqual(path, PROJECT_ROOT / "cache")
        enable.assert_called_once_with(str(PROJECT_ROOT / "cache"))

    def test_only_utils_and_frontend_config_configure_the_cache(self) -> None:
        """Any other enable_cache call risks a second cache (data_lake/cache) or none at all."""
        allowed = {SCRIPTS_DIR / "utils.py", PROJECT_ROOT / "frontend" / "config.py"}
        offenders = [str(p.relative_to(PROJECT_ROOT)) for p in _python_files()
                     if "Cache.enable_cache(" in p.read_text(encoding="utf-8") and p not in allowed
                     and "tests" not in p.parts]
        self.assertEqual(offenders, [])

    def test_frontend_cache_matches_backend_cache(self) -> None:
        sys.path.insert(0, str(PROJECT_ROOT / "frontend"))
        import config as frontend_config
        self.assertEqual(frontend_config.CACHE_DIR.resolve(), utils.CACHE_DIR.resolve())

    def test_racing_line_lap_lookup_enables_project_cache(self) -> None:
        import pandas as pd
        from racing_line import racing_line_io

        df = pd.DataFrame({"Year": [2023], "Round": [1], "Session": ["Q"],
                           "SessionTime": pd.to_timedelta([0], unit="s")})
        with mock.patch.object(racing_line_io, "enable_fastf1_cache") as enable, \
             mock.patch("fastf1.get_session", side_effect=RuntimeError("offline")):
            racing_line_io.fetch_fastest_lap_bounds(df, "1", mock.MagicMock())
        enable.assert_called_once()


class TestLoggingEncoding(unittest.TestCase):

    def test_unencodable_characters_do_not_drop_log_lines(self) -> None:
        """A redirected Windows stdout is cp1252; '→' used to raise and lose the message."""
        import io
        import logging

        out = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
        err = io.StringIO()
        with mock.patch("sys.stdout", out), mock.patch("sys.stderr", err):
            lg = utils.setup_logger("test_logging_encoding_cp1252")
            lg.info("contamination=0.005 → mean CV score=0.19")
            out.flush()
        logging.getLogger("test_logging_encoding_cp1252").handlers.clear()
        written = out.buffer.getvalue().decode("cp1252")
        self.assertIn("contamination=0.005 ? mean CV score=0.19", written)
        self.assertNotIn("Logging error", err.getvalue())


class TestRequirementsComplete(unittest.TestCase):

    def test_every_third_party_import_is_in_requirements(self) -> None:
        required = {
            line.strip().split("[")[0].split("=")[0].split(">")[0].lower()
            for line in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        missing = {}
        for path in _python_files():
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    if top in sys.stdlib_module_names or top in LOCAL_MODULES:
                        continue
                    if IMPORT_TO_PACKAGE.get(top, top).lower() not in required:
                        missing.setdefault(top, str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(missing, {}, "Imported but not listed in requirements.txt")


if __name__ == "__main__":
    unittest.main()
