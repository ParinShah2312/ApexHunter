"""
ApexHunter parallel test runner.
Runs all unit tests in parallel, then optionally runs integration tests.

Usage:
  python backend/scripts/tests/run_tests.py              # unit tests only (fast)
  python backend/scripts/tests/run_tests.py --all        # unit + integration (slow)
  python backend/scripts/tests/run_tests.py --file test_geometry_distance.py  # one file
  (--file on an integration test runs it with APEXHUNTER_RUN_SLOW=1, so it is not skipped)
"""

import re
import subprocess
import concurrent.futures
import argparse
import os
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).parent
UNIT_DIR = TESTS_DIR / "unit"
INTEGRATION_DIR = TESTS_DIR / "integration"
PROJECT_ROOT = TESTS_DIR.parent.parent.parent  # ApexHunter 2.0/


def parse_counts(unittest_output: str) -> tuple:
    """Return (tests run, tests skipped) from unittest's summary ("Ran 13 tests", "skipped=13")."""
    ran = re.search(r"^Ran (\d+) tests?", unittest_output, re.MULTILINE)
    skipped = re.search(r"skipped=(\d+)", unittest_output)
    return (int(ran.group(1)) if ran else 0, int(skipped.group(1)) if skipped else 0)


def status_label(result: dict) -> str:
    """FAIL on a non-zero exit; SKIP when every test was skipped (nothing was checked); else PASS."""
    if not result["passed"]:
        return "FAIL"
    if result["ran"] == result["skipped"]:
        return "SKIP"
    return "PASS"


def run_file(test_file: Path, slow: bool = False) -> dict:
    """Run a single test file in a subprocess. Returns result dict."""
    env = dict(os.environ)
    if slow:
        env["APEXHUNTER_RUN_SLOW"] = "1"

    # Use the module path relative to project root
    rel = test_file.relative_to(PROJECT_ROOT)
    module_path = str(rel).replace(os.sep, "/")

    start = time.time()
    result = subprocess.run(
        # sys.executable, not "python": on Windows a bare "python" can resolve to the base
        # interpreter instead of the project venv, silently testing against the wrong packages
        [sys.executable, "-m", "unittest", module_path, "-v"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - start
    ran, skipped = parse_counts(result.stderr)

    return {
        "file": test_file.name,
        "returncode": result.returncode,
        "elapsed": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
        "ran": ran,
        "skipped": skipped,
    }


def run_all(include_slow: bool = False, specific_file: str = None):
    """Run tests in parallel and print consolidated report."""
    if specific_file:
        files = list(UNIT_DIR.glob(specific_file)) + list(INTEGRATION_DIR.glob(specific_file))
        if not files:
            print(f"No file matching '{specific_file}' found.")
            return 1
    else:
        files = sorted(UNIT_DIR.glob("test_*.py"))
        if include_slow:
            files += sorted(INTEGRATION_DIR.glob("test_*.py"))

    print(f"\n  Running {len(files)} test file(s) "
          f"({'unit + integration' if include_slow else 'unit only'})...\n")

    total_start = time.time()
    results = []

    max_workers = min(len(files), 8)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # An integration file named with --file runs for real; it used to be skipped and reported PASS
        futures = {
            executor.submit(run_file, f, include_slow or f.parent == INTEGRATION_DIR): f
            for f in files
        }
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            results.append(r)
            note = f"  ({r['skipped']} of {r['ran']} skipped)" if r["skipped"] else ""
            print(f"  [{status_label(r)}] {r['file']:<45} {r['elapsed']:.2f}s{note}")

    total_elapsed = time.time() - total_start
    passed = sum(1 for r in results if status_label(r) == "PASS")
    skipped_files = [r for r in results if status_label(r) == "SKIP"]
    failed = [r for r in results if not r["passed"]]
    n_tests = sum(r["ran"] for r in results)
    n_skipped = sum(r["skipped"] for r in results)

    print(f"\n  {'-' * 55}")
    print(f"  {passed}/{len(results)} files passed in {total_elapsed:.2f}s "
          f"({n_tests - n_skipped} tests run, {n_skipped} skipped)")
    if skipped_files:
        print(f"  {len(skipped_files)} file(s) had every test skipped: "
              f"{', '.join(r['file'] for r in skipped_files)}")

    if failed:
        print("\n  FAILURES:\n")
        for r in failed:
            print(f"  -- {r['file']} --")
            print(r["stderr"][-2000:])
        return 1
    elif skipped_files:
        print("  No failures, but the skipped file(s) above checked nothing.\n")
        return 0
    else:
        print("  All tests passed.\n")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ApexHunter parallel test runner")
    parser.add_argument("--all", action="store_true", help="Include slow integration tests")
    parser.add_argument("--file", type=str, help="Run a specific test file by name")
    args = parser.parse_args()
    exit(run_all(include_slow=args.all, specific_file=args.file))
