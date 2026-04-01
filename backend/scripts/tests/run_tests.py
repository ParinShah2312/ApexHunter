"""
ApexHunter parallel test runner.
Runs all unit tests in parallel, then optionally runs integration tests.

Usage:
  python backend/scripts/tests/run_tests.py              # unit tests only (fast)
  python backend/scripts/tests/run_tests.py --all        # unit + integration (slow)
  python backend/scripts/tests/run_tests.py --file test_geometry_distance.py  # one file
"""

import subprocess
import concurrent.futures
import argparse
import os
import time
from pathlib import Path

TESTS_DIR = Path(__file__).parent
UNIT_DIR = TESTS_DIR / "unit"
INTEGRATION_DIR = TESTS_DIR / "integration"
PROJECT_ROOT = TESTS_DIR.parent.parent.parent  # ApexHunter 2.0/


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
        ["python", "-m", "unittest", module_path, "-v"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - start

    return {
        "file": test_file.name,
        "returncode": result.returncode,
        "elapsed": elapsed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
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
        futures = {
            executor.submit(run_file, f, include_slow): f
            for f in files
        }
        for future in concurrent.futures.as_completed(futures):
            r = future.result()
            results.append(r)
            status = "PASS" if r["passed"] else "FAIL"
            print(f"  [{status}] {r['file']:<45} {r['elapsed']:.2f}s")

    total_elapsed = time.time() - total_start
    passed = sum(1 for r in results if r["passed"])
    failed = [r for r in results if not r["passed"]]

    print(f"\n  {'-' * 55}")
    print(f"  {passed}/{len(results)} files passed in {total_elapsed:.2f}s")

    if failed:
        print(f"\n  FAILURES:\n")
        for r in failed:
            print(f"  -- {r['file']} --")
            print(r["stderr"][-2000:])
        return 1
    else:
        print(f"  All tests passed.\n")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ApexHunter parallel test runner")
    parser.add_argument("--all", action="store_true", help="Include slow integration tests")
    parser.add_argument("--file", type=str, help="Run a specific test file by name")
    args = parser.parse_args()
    exit(run_all(include_slow=args.all, specific_file=args.file))
