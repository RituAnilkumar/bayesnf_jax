"""
src/logging_utils.py

Lightweight stdout/stderr tee for capturing printed output to a log file.

Usage:
    from src.logging_utils import LogTee

    with LogTee(output_dir / "run.log"):
        run_main_function(cfg)

The log file is opened in append mode so repeated runs accumulate rather than
overwrite.  A timestamped header and footer are written on each entry/exit.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


class _TeeStream:
    """Forwards writes to both an original stream and a log file."""

    def __init__(self, original, log_file):
        self._original = original
        self._log_file = log_file

    def write(self, data: str) -> int:
        self._original.write(data)
        self._log_file.write(data)
        return len(data)

    def flush(self) -> None:
        self._original.flush()
        self._log_file.flush()

    def isatty(self) -> bool:
        return False

    # Forward any other attribute access to the original stream so that
    # libraries that inspect sys.stdout (e.g. tqdm) don't break.
    def __getattr__(self, name):
        return getattr(self._original, name)


class LogTee:
    """
    Context manager that tees stdout and stderr to a log file.

    Opens in append mode — repeated runs accumulate in the same file.
    Each run is delimited by a timestamped header and footer.

    Args:
        log_path: Path to the log file.  Parent directories are created if needed.
    """

    def __init__(self, log_path: str | Path) -> None:
        self._log_path = Path(log_path)

    def __enter__(self) -> "LogTee":
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._log_path, "a", buffering=1, encoding="utf-8")

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._file.write(f"\n{'=' * 70}\n")
        self._file.write(f"Run started : {ts}\n")
        self._file.write(f"{'=' * 70}\n\n")
        self._file.flush()

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeStream(self._orig_stdout, self._file)
        sys.stderr = _TeeStream(self._orig_stderr, self._file)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr

        self._file.write(f"\n{'=' * 70}\n")
        if exc_type is not None:
            self._file.write(f"Run FAILED  : {ts}  ({exc_type.__name__}: {exc_val})\n")
        else:
            self._file.write(f"Run ended   : {ts}\n")
        self._file.write(f"{'=' * 70}\n")
        self._file.close()

        print(f"  Log written → {self._log_path}")
        return False   # do not suppress exceptions
