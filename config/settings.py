# config/settings.py
"""
Configuration settings for the AV Catalog Standardizer
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Application configuration
APP_CONFIG = {
    "version": "1.0.0",
    "app_name": "AV Catalog Standardizer",
    "max_threads": 4,
    "chunk_size": 100,  # rows per chunk
    "cache_enabled": True,
    "cache_ttl": 86400,  # 24 hours in seconds
}

# LLM configuration
MODEL_CONFIG = {
    "model_id": "google/gemma-2b-it",
    "quantization": "4bit",      # options: "4bit", "8bit", or None
    "max_new_tokens": 512,       # adjust based on response scope
    "temperature": 0.3,          # lower for deterministic outputs
    "top_p": 0.95,               # nucleus sampling threshold
    "cache_enabled": True        # speed benefit from caching
}

# Alternative model configuration
ALT_MODEL_CONFIG = {
    "model_id": "microsoft/phi-2",
    "quantization": "4bit",
    "max_new_tokens": 512,
    "temperature": 0.3,
    "top_p": 0.95,
    "cache_enabled": True
}

# File parsing settings
PARSER_CONFIG = {
    "csv": {
        "encoding_detection": True,
        "delimiter_detection": True,
        "header_detection": True,
    },
    "excel": {
        "multi_sheet": True,
        "header_detection": True,
    },
    "pdf": {
        "table_extraction": True,
        "ocr_enabled": True,
    },
    "json": {
        "flatten_nested": True,
        "normalize_arrays": True
    },
    "xml": {
        "flatten_attributes": True,
        "record_path": None,  # Auto-detect
        "attribute_prefix": "@"
    }
}

# Field mapping confidence thresholds
MAPPING_THRESHOLDS = {
    "high_confidence": 0.85,
    "medium_confidence": 0.65,
    "low_confidence": 0.40,
    "minimum_required": 0.30,
}

# Required fields (must be present in output)
REQUIRED_FIELDS = [
    "SKU",
    "Short Description",
    "Manufacturer",
    "Trade Price",
]

# Add numeric field definitions
NUMERIC_FIELDS = [
    "Buy Cost",
    "Trade Price",
    "MSRP GBP",
    "MSRP USD",
    "MSRP EUR"
]
