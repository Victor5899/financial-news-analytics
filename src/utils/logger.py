"""
Logging configuration for the financial-news-analytics project.

All modules should obtain their logger through ``get_logger(__name__)``
rather than calling ``logging.getLogger`` directly. This guarantees a
consistent format and avoids duplicate handler registration.

Call ``configure_logging(level)`` once at application startup (e.g. in
scripts/ or Streamlit app.py). Individual module loggers inherit the root
level automatically.

Usage
-----
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Fetching articles", extra={"ticker": "AAPL"})
"""

from __future__ import annotations

import logging
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT))
_handler.setLevel(logging.DEBUG)

# Attach handler to the root logger once at module-import time.
_root = logging.getLogger()
if not _root.handlers:
    _root.addHandler(_handler)


def configure_logging(level: str = "INFO") -> None:
    """Set the effective log level for the entire application.

    Call this once at the entry point (script, Streamlit app, etc.)
    before any other imports that might emit log lines.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.getLogger().setLevel(numeric)
    _handler.setLevel(numeric)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger inheriting the application-wide configuration."""
    return logging.getLogger(name)
