"""Logging configuration utility for the project.

Provides a standardized logger with formatted console and optional file
logging across all pipeline components.
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Returns a configured logger instance with standard formatting.

    Args:
        name: The name of the logger (typically __name__).

    Returns:
        logging.Logger: Configured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
