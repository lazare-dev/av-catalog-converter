# config/logging_config.py
"""
Logging configuration for the AV Catalog Standardizer
"""

import os
import sys
import json
import logging
import logging.config
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union

from config.settings import LOGS_DIR
from utils.logging.json_formatter import JsonFormatter

# Default logging configuration
DEFAULT_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "detailed": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "json": {
            "()": "utils.logging.json_formatter.JsonFormatter",
            "datefmt": "%Y-%m-%dT%H:%M:%S.%fZ"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",  # Changed from INFO to DEBUG
            "formatter": "detailed",  # Changed from standard to detailed
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": str(LOGS_DIR / "app.log"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        },
        "json_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "json",
            "filename": str(LOGS_DIR / "app.json"),
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        }
    },
    "loggers": {
        "": {  # Root logger
            "handlers": ["console", "file", "json_file"],
            "level": "DEBUG",  # Changed from INFO to DEBUG
            "propagate": True
        },
        "core": {
            "level": "DEBUG",
            "propagate": True
        },
        "services": {
            "level": "DEBUG",
            "propagate": True
        },
        "utils": {
            "level": "DEBUG",
            "propagate": True
        },
        "web": {
            "level": "DEBUG",
            "propagate": True
        },
        # Third-party modules - keep these at WARNING to avoid excessive logs
        "urllib3": {
            "level": "WARNING",
            "propagate": True
        },
        "matplotlib": {
            "level": "WARNING",
            "propagate": True
        },
        "PIL": {
            "level": "WARNING",
            "propagate": True
        }
    }
}


def setup_logging(level: Union[int, str] = logging.INFO, config: Optional[Dict[str, Any]] = None) -> logging.Logger:
    """
    Set up application logging

    Args:
        level: Default logging level
        config: Custom logging configuration

    Returns:
        Root logger
    """
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True, parents=True)

    # Use provided config or default
    log_config = config or DEFAULT_CONFIG

    # Adjust root logger level if specified
    if level != logging.INFO:
        log_config["loggers"][""]["level"] = level if isinstance(level, str) else logging.getLevelName(level)

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_log_file = LOGS_DIR / f"av_catalog_standardizer_{timestamp}.json"

    # Update JSON file handler with timestamped filename
    log_config["handlers"]["json_file"]["filename"] = str(json_log_file)

    # Configure logging
    try:
        logging.config.dictConfig(log_config)
    except Exception as e:
        # Fallback to basic configuration
        print(f"Error configuring logging: {e}")
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    # Get root logger
    root_logger = logging.getLogger()

    # Log startup information
    root_logger.info(f"Logging initialized at {log_config['loggers']['']['level']} level")
    root_logger.info(f"Log file: {json_log_file}")

    return root_logger