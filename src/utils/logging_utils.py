"""Logging via the stdlib `logging` module with a consistent format.

A single console handler is configured once on the root project logger; child
loggers (`get_logger("data.jester")` etc.) inherit it. `setup_file_logging`
additionally tees output to a file inside a run directory.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_ROOT_NAME = "ge"  # gesture-efficiency
_CONFIGURED = False
_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _ensure_root_configured(level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    root = logging.getLogger(_ROOT_NAME)
    if not _CONFIGURED:
        root.setLevel(level)
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    return root


def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """Return a project logger. `name` is namespaced under the project root."""
    _ensure_root_configured(level)
    if name is None or name == _ROOT_NAME:
        return logging.getLogger(_ROOT_NAME)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def setup_file_logging(run_dir: str | Path, filename: str = "run.log") -> Path:
    """Tee all project logs into `run_dir/filename`. Returns the log path."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / filename
    root = _ensure_root_configured()
    # Avoid duplicate file handlers if called twice for the same path.
    for h in root.handlers:
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path.resolve():
            return log_path
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
    root.addHandler(fh)
    return log_path
