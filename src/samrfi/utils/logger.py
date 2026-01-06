"""
Structured logging for SAM-RFI.

This module provides consistent, formatted logging across all SAM-RFI modules
with configurable log levels, console output, and optional file output.

The logger uses a standardized format: [YYYY-MM-DD HH:MM:SS] LEVEL: Message
which makes it easy to track events during training, data generation, and inference.

Functions
---------
setup_logger
    Create and configure a logger with console and optional file handlers.

Module Variables
----------------
logger : logging.Logger
    Global logger instance for SAM-RFI, pre-configured with INFO level.

Examples
--------
>>> from samrfi.utils.logger import logger, setup_logger
>>>
>>> # Use default global logger
>>> logger.info("Starting training...")
[2025-12-30 10:30:45] INFO: Starting training...
>>>
>>> # Create custom logger with DEBUG level
>>> debug_logger = setup_logger(name="debug", level=logging.DEBUG)
>>> debug_logger.debug("Detailed debug information")
[2025-12-30 10:30:46] DEBUG: Detailed debug information
>>>
>>> # Create logger with file output
>>> file_logger = setup_logger(
...     name="training",
...     level=logging.INFO,
...     log_file="training.log"
... )
>>> file_logger.info("Training started")  # Writes to both console and file
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "samrfi",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup structured logger for SAM-RFI.

    Creates a logger with console output and optional file output, using
    a consistent timestamp format across all log messages. If called multiple
    times with the same name, returns the existing logger to avoid duplicate
    handlers.

    Parameters
    ----------
    name : str, optional
        Logger name for identification. Default is 'samrfi'.
    level : int, optional
        Minimum log level to capture. Use logging constants:
        - logging.DEBUG (10): Detailed diagnostic information
        - logging.INFO (20): General informational messages (default)
        - logging.WARNING (30): Warning messages
        - logging.ERROR (40): Error messages
        - logging.CRITICAL (50): Critical error messages
        Default is logging.INFO.
    log_file : str or None, optional
        Path to log file for persistent logging. If None, only console
        output is used. Parent directories are created automatically
        if they don't exist. Default is None.

    Returns
    -------
    logging.Logger
        Configured logger instance with console handler and optional
        file handler.

    Notes
    -----
    - Log format: [YYYY-MM-DD HH:MM:SS] LEVEL: Message
    - Console output goes to stdout (not stderr)
    - File output appends to existing file if it exists
    - Calling multiple times with same name returns existing logger

    Examples
    --------
    >>> import logging
    >>> from samrfi.utils.logger import setup_logger
    >>>
    >>> # Basic logger with INFO level
    >>> logger = setup_logger()
    >>> logger.info("Processing data...")
    [2025-12-30 10:30:45] INFO: Processing data...
    >>>
    >>> # Debug logger
    >>> debug_logger = setup_logger(name="debug", level=logging.DEBUG)
    >>> debug_logger.debug("Variable x = 42")
    [2025-12-30 10:30:46] DEBUG: Variable x = 42
    >>>
    >>> # Logger with file output
    >>> file_logger = setup_logger(
    ...     name="training",
    ...     level=logging.INFO,
    ...     log_file="./logs/training.log"
    ... )
    >>> file_logger.info("Epoch 1/10 complete")
    [2025-12-30 10:30:47] INFO: Epoch 1/10 complete
    >>>
    >>> # Warning and error logging
    >>> logger.warning("GPU memory running low")
    [2025-12-30 10:30:48] WARNING: GPU memory running low
    >>> logger.error("Failed to load checkpoint")
    [2025-12-30 10:30:49] ERROR: Failed to load checkpoint
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Format: [2025-12-26 10:30:45] INFO: Message
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = setup_logger()
