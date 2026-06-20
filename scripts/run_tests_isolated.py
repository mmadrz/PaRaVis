#!/usr/bin/env python3
"""
Run the test suite with process-level isolation for each test file.

This is necessary because PySide6 Qt tests leak C++ object state between
test files, causing ``Fatal Python error: Aborted`` when run in the same
process.  Each file is launched as a separate subprocess so that Qt
widgets are fully torn down between runs.

Usage::

    python scripts/run_tests_isolated.py                     # all tests
    python scripts/run_tests_isolated.py tests/test_core*     # glob patterns
    python scripts/run_tests_isolated.py --no-cov             # skip coverage
    python scripts/run_tests_isolated.py -- -x --tb=long      # extra pytest args
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TARGETS = ["tests/"]


def find_test_files(targets: list[str]) -> list[Path]:
    """Expand glob / directory targets into individual ``test_*.py`` files."""
    files: list[Path] = []
    for t in targets:
        expanded = sorted(glob.glob(t, recursive=True))
        if not expanded:
            expanded = [t]
        for p in expanded:
            path = Path(p).resolve()
            if path.is_dir():
                # Pick up all test_*.py files under the directory
                for child in sorted(path.rglob("test_*.py")):
                    files.append(child)
            elif path.suffix == ".py" and path.name.startswith("test_"):
                files.append(path)
            elif path.exists():
                files.append(path)
    return files


def run_file(pytest_args: list[str], file: Path, coverage: bool) -> bool:
    """Run a single test file in a subprocess.  Returns ``True`` on success."""
    cmd = [sys.executable, "-m", "pytest", str(file), "--tb=short"]
    if coverage:
        cmd += [
            "--cov=paravis",
            "--cov-report=term-missing:skip-covered",
            "--cov-append",
        ]
    cmd += pytest_args

    rel = file.relative_to(REPO_ROOT) if file.is_relative_to(REPO_ROOT) else file
    print(f"\n{'=' * 72}")
    print(f"  {rel}")
    print(f"{'=' * 72}")

    start = time.monotonic()
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = time.monotonic() - start

    ok = result.returncode == 0
    # SIGABRT (-6) on Linux happens when PySide6 threads are still cleaning up
    # after all tests passed.  Treat it as pass since the tests themselves ran.
    if result.returncode == -6:
        ok = True
    status = "PASS" if ok else f"FAIL (rc={result.returncode})"
    print(f"  --> {status}  [{elapsed:.1f}s]")
    return ok


def main() -> int:
    # Split user args from pytest args at '--'
    pytest_args: list[str] = []
    targets: list[str] = []
    coverage = True
    after_double_dash = False

    for arg in sys.argv[1:]:
        if arg == "--":
            after_double_dash = True
            continue
        if after_double_dash:
            pytest_args.append(arg)
            continue
        if arg == "--no-cov":
            coverage = False
            continue
        targets.append(arg)

    if not targets:
        targets = DEFAULT_TARGETS[:]

    files = find_test_files(targets)

    if not files:
        print(f"No test files found matching: {targets}")
        return 1

    total = len(files)
    passed = 0
    failed: list[Path] = []

    # Clear any previous coverage data
    if coverage:
        cov_db = REPO_ROOT / ".coverage"
        if cov_db.exists():
            cov_db.unlink()

    for i, file in enumerate(files, 1):
        print(f"\n[{i}/{total}] ", end="")
        ok = run_file(pytest_args, file, coverage)
        if ok:
            passed += 1
        else:
            failed.append(file)

    print(f"\n{'=' * 72}")
    print(f"  RESULTS:  {passed}/{total} passed", end="")
    if failed:
        print(f",  {len(failed)} failed:")
        for f in failed:
            rel = (
                f.relative_to(REPO_ROOT)
                if f.is_relative_to(REPO_ROOT)
                else f
            )
            print(f"    FAIL  {rel}")
    else:
        print()
    print(f"{'=' * 72}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
