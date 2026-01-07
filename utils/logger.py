import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from utils.paths import LOGS

LOG_DIR = Path(LOGS)
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"


def setup_logging(
    level: int = logging.INFO
) -> None:
    """
    Create and configure a named logger with console and rotating file handlers.

    Parameters:
        level: Logging level for the logger (default: logging.INFO).

    Returns:
        Configured logger instance.
    """

    root = logging.getLogger()

    if getattr(root, "_z_index_logging_configured", False):
        return

    root.setLevel(level)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(fmt)

    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(fmt)

    root.addHandler(console)
    root.addHandler(file_handler)

    root._z_index_logging_configured = True
