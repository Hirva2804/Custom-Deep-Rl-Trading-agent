"""
Centralised logging configuration.
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Configure the root ``deeprl_trade`` logger.

    Parameters
    ----------
    level : str
        Logging level name (``DEBUG``, ``INFO``, ``WARNING``, …).

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    logger = logging.getLogger("deeprl_trade")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s — %(name)s — %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
