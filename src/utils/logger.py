# utils/logger.py

import logging
import os

LOG_DIR = "logs"
LOG_FILE = "app.log"

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)  # log everything, decide output by handler level

        os.makedirs(LOG_DIR, exist_ok=True)

        # File handler - full log
        file_handler = logging.FileHandler(os.path.join(LOG_DIR, LOG_FILE), encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_format = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

        # Console handler - only show key warnings
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_format = logging.Formatter("🔔 %(message)s")
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

    return logger
