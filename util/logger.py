import logging
from logging.handlers import RotatingFileHandler
import os

# --- Ensure logs directory exists ---
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str, log_file: str = "app.log") -> logging.Logger:
    """
    Creates and returns a logger instance with both console and rotating file handlers.

    Args:
        name (str): The name of the logger (usually __name__).
        log_file (str): The log file name (default: app.log).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if function is called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # --- Format for logs ---
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- Console handler ---
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # --- Rotating file handler (5 MB per file, keep 3 backups) ---
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_file),
        maxBytes=5_000_000,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False
    return logger
