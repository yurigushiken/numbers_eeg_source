import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Dict

# Track handlers to avoid duplicates when setup_run_logger is called multiple times
_attached_handlers: Dict[Path, RotatingFileHandler] = {}


def setup_run_logger(log_path: Path, *, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 3) -> Path:
    """Attach a rotating file handler to the root logger for this run.

    Parameters
    ----------
    log_path : Path
        Destination file for the log output.
    max_bytes : int, optional
        Maximum size per log file before rotation (default 10 MB).
    backup_count : int, optional
        Number of rotated files to keep (default 3).

    Returns
    -------
    Path
        The resolved log file path that will capture run output.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if log_path in _attached_handlers:
        return log_path

    handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    _attached_handlers[log_path] = handler

    return log_path
