"""Real bundled telemetry, integrity failures, and full dashboard interactions. No mocks."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "frontend"))
from components.demo_assets import BUNDLE_DIR, session_files, validate_bundle


class TestDemoBundle(unittest.TestCase):
    def test_genuine_bundle_schema_and_local_data_precedence(self):
        manifest = validate_bundle(BUNDLE_DIR)
        for entry in manifest["files"]:
            frame = pd.read_parquet(BUNDLE_DIR / entry["path"])
            self.assertEqual(len(frame), entry["rows"])
            self.assertGreater(len(frame), 1000)
            self.assertEqual(set(frame.Driver.astype(str)), {"1", "11"})
            self.assertTrue({"Speed", "Throttle", "Brake", "RPM", "nGear", "X", "Y",
                             "SessionTime"}.issubset(frame.columns))
            self.assertGreater(frame.Speed.max(), 250)
            self.assertTrue(frame.SessionTime.notna().all())
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.assertEqual(len(session_files(folder)), 2)
            local = folder / "2023_1_Q.parquet"
            shutil.copyfile(BUNDLE_DIR / "clean_data/2023_1_Q.parquet", local)
            self.assertEqual(session_files(folder), [local])

    def test_rejects_corrupt_bytes_and_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "bundle"
            shutil.copytree(BUNDLE_DIR, bundle)
            manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
            asset = bundle / manifest["files"][0]["path"]
            with asset.open("r+b") as stream:
                stream.write(b"FAIL")
            with self.assertRaisesRegex(ValueError, "integrity"):
                validate_bundle(bundle)
            manifest["files"][0]["path"] = "../outside.parquet"
            (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "asset path"):
                validate_bundle(bundle)


class TestDemoDashboard(unittest.TestCase):
    def test_sessions_drivers_charts_refresh_and_unavailable_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = (
                "import sys\n"
                f"sys.path.insert(0, {str(ROOT / 'frontend')!r})\n"
                "from pathlib import Path\n"
                "from components import sidebar\n"
                f"sidebar.DATA_LAKE_DIR = Path({tmp!r})\n"
                f"exec(Path({str(ROOT / 'frontend/app.py')!r}).read_text(encoding='utf-8'))\n"
            )
            app = AppTest.from_string(script, default_timeout=90).run()
            self.assertFalse(app.exception, [e.value for e in app.exception])
            self.assertEqual(app.sidebar.selectbox(key="sel_year").options, ["2023"])
            sessions = app.sidebar.selectbox(key="sel_session").options
            self.assertEqual(len(sessions), 2)
            for session in sessions:
                app.sidebar.selectbox(key="sel_session").select(session).run()
                for driver in app.sidebar.selectbox(key="sel_driver").options:
                    app.sidebar.selectbox(key="sel_driver").select(driver).run()
                    self.assertFalse(app.exception, [e.value for e in app.exception])
                    self.assertFalse(app.error, [e.value for e in app.error])
                    self.assertGreaterEqual(len(app.get("plotly_chart")), 3)
                    self.assertTrue(any(m.label == "Top Speed" and "km/h" in m.value
                                        for m in app.metric))
                    self.assertTrue(any("Isolation Forest output" in w.value for w in app.warning))
                    self.assertTrue(any("Onboard video" in i.value for i in app.info))
            app.run()  # A rerun must recover data and keep the selected session/driver.
            self.assertFalse(app.exception)
            cold = AppTest.from_string(script, default_timeout=90).run()
            self.assertFalse(cold.exception)
            self.assertGreaterEqual(len(cold.get("plotly_chart")), 3)


if __name__ == "__main__":
    unittest.main()
