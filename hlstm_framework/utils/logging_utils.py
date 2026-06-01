"""
Logging utilities — logger setup, timer context manager.

Usage:
    >>> from hlstm_framework.utils import setup_logger, Timer
    >>> logger = setup_logger(__name__)
    >>> logger.info("Training started")
    >>> with Timer("epoch"):
    ...     run_epoch()
"""

import logging
import time
from typing import Optional


def setup_logger(
    name: str,
    level: int = logging.INFO,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a logger with console output.

    Args:
        name: Logger name (typically __name__).
        level: Logging level.
        fmt: Log format string.

    Returns:
        Configured Logger instance.
    """
    if fmt is None:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        logger.addHandler(handler)

    return logger


class Timer:
    """Context manager for timing code blocks.

    Args:
        name: Label for this timer (used in the report string).
        logger: Optional logger. If None, prints to stdout.

    Example:
        >>> with Timer("forward pass"):
        ...     out = model(x)
    """

    def __init__(self, name: str = "block", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger
        self.elapsed: float = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        msg = f"[Timer] {self.name}: {self.elapsed:.3f}s"
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)
