import logging
from logging.handlers import RotatingFileHandler
import os


def get_logger(name: str, log_file: str = "app.log") -> logging.Logger:
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # --- Ensure logs directory exists ---
    LOG_DIR = "logs"
    os.makedirs(LOG_DIR, exist_ok=True)

    # If user passed "logs/xxx.log" → remove leading folder
    log_file_only = os.path.basename(log_file)

    # Final path
    full_path = os.path.join(LOG_DIR, log_file_only)

    # Log format
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler
    file_handler = RotatingFileHandler(
        full_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.propagate = False

    return logger
