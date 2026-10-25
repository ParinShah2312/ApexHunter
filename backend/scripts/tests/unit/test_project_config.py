"""Project-wide configuration guards: single FastF1 cache location, complete requirements.txt and
explicit UTF-8 for text files."""

import ast
import sys
import unittest
from pathlib import Path

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




def _text_open_calls_without_encoding(tree: ast.AST):
    """Yield line numbers of builtin open() calls in text mode that don't pass encoding=."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "open"):
            continue
        mode = node.args[1] if len(node.args) > 1 else next(
            (k.value for k in node.keywords if k.arg == "mode"), None)
        if isinstance(mode, ast.Constant) and "b" in str(mode.value):
            continue
        if not any(k.arg == "encoding" for k in node.keywords):
            yield node.lineno


class TestUtf8TextFiles(unittest.TestCase):
    """Regression for #23: on Windows open() defaults to cp1252, so config.json's "São Paulo"
    was read as "SÃ£o Paulo"."""

    def test_config_non_ascii_names_read_correctly(self) -> None:
        names = [c["name"] for c in utils.CONFIG["satellite"]["circuits"]]
        self.assertIn("São Paulo Grand Prix", names)

    def test_every_text_open_passes_an_encoding(self) -> None:
        offenders = []
        for path in _python_files():
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            offenders += [f"{path.relative_to(PROJECT_ROOT)}:{n}" for n in _text_open_calls_without_encoding(tree)]
        self.assertEqual(offenders, [])


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
