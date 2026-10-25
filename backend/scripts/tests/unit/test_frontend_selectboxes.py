"""Dashboard dropdowns must not accept typing, and must not rely on the deprecated
st.components.v1.html JavaScript injection that previously enforced this."""

import ast
import os
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def _frontend_files():
    return [p for p in FRONTEND_DIR.rglob("*.py") if "__pycache__" not in p.parts]


class TestSelectboxesNotTypeable(unittest.TestCase):

    def test_every_selectbox_call_disables_typing(self) -> None:
        offenders = []
        for path in _frontend_files():
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                        and node.func.attr == "selectbox":
                    mode = next((k.value for k in node.keywords if k.arg == "filter_mode"), None)
                    if not (isinstance(mode, ast.Constant) and mode.value is None):
                        offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")
        self.assertEqual(offenders, [], "selectbox without filter_mode=None")

    def test_no_deprecated_components_html(self) -> None:
        users = [str(p.relative_to(PROJECT_ROOT)) for p in _frontend_files()
                 if "components.v1" in "".join(
                     line for line in p.read_text(encoding="utf-8").splitlines()
                     if not line.lstrip().startswith("#"))]
        self.assertEqual(users, [])

    def test_rendered_sidebar_dropdowns_have_typing_disabled(self) -> None:
        from streamlit.proto.Selectbox_pb2 import Selectbox
        from streamlit.testing.v1 import AppTest

        filter_mode_none = Selectbox.DESCRIPTOR.fields_by_name["filter_mode"] \
            .enum_type.values_by_name["FILTER_MODE_NONE"].number

        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(FRONTEND_DIR)!r})\n"
            "from components.sidebar import render_sidebar\n"
            "render_sidebar()\n"
        )
        cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        try:
            at = AppTest.from_string(script, default_timeout=120).run()
        finally:
            os.chdir(cwd)
        self.assertFalse(at.exception, [e.value for e in at.exception])
        boxes = at.sidebar.selectbox
        self.assertEqual([b.label for b in boxes], ["Year", "Session", "Driver"])
        for box in boxes:
            self.assertEqual(box.proto.filter_mode, filter_mode_none, box.label)


class TestDropdownKeyLock(unittest.TestCase):
    """filter_mode=None stops typing, but Backspace/Delete would still clear a dropdown's shown
    value, so app.py injects a key lock. It must reach the page with JavaScript enabled."""

    def test_dashboard_injects_backspace_delete_lock(self) -> None:
        from streamlit.testing.v1 import AppTest

        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(FRONTEND_DIR)!r})\n"
            f"exec(open({str(FRONTEND_DIR / 'app.py')!r}, encoding='utf-8').read())\n"
        )
        cwd = os.getcwd()
        os.chdir(PROJECT_ROOT)
        try:
            at = AppTest.from_string(script, default_timeout=300).run()
        finally:
            os.chdir(cwd)
        self.assertFalse(at.exception, [e.value for e in at.exception])
        html_blocks = [el.proto for el in at.main.children.values()
                       if type(el).__name__ == "UnknownElement" and el.type == "html"]
        locks = [p for p in html_blocks if "Backspace" in p.body and "Delete" in p.body
                 and 'data-baseweb="select"' in p.body]
        self.assertEqual(len(locks), 1)
        self.assertTrue(locks[0].unsafe_allow_javascript, "script would be stripped without this flag")


if __name__ == "__main__":
    unittest.main()
