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

# LLM configuration - using a smaller model for better compatibility
MODEL_CONFIG = {
    "model_id": "gpt2",          # Use a small GPT-2 model that's compatible with AutoModelForCausalLM
    "model_type": "gpt2",        # Explicitly set model type to ensure correct handling
    "max_new_tokens": 128,       # smaller for faster responses
    "temperature": 0.1,          # lower for deterministic outputs
    "top_p": 0.9,                # nucleus sampling threshold
    "repetition_penalty": 1.0,   # prevent repetitive outputs
    "max_input_length": 512,     # maximum input length in tokens

    # Memory optimization
    "low_cpu_mem_usage": True,   # optimize memory usage during loading
    "device_map": "cpu",         # force CPU for better compatibility

    # Caching configuration
    "cache_enabled": True,       # speed benefit from caching
    "cache_type": "adaptive",    # options: "adaptive", "memory", "disk"
    "cache_ttl": 3600,           # base TTL in seconds (1 hour)
    "cache_max_size": 1000,      # maximum number of cached items

    # Rate limiting configuration
    "rate_limiting_enabled": True,  # prevent overloading LLM
    "requests_per_minute": 60,      # maximum requests per minute
    "burst_size": 1000             # maximum burst size for large prompts
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
    "required_field_minimum": 0.50,  # Minimum confidence for required fields
    "accept_single_mapping": 0.60,   # Minimum confidence to accept a single mapping
    "prefer_direct_mapping": 0.75    # Threshold to prefer direct mapping over semantic
}

# Field definitions are now centralized in config.schema
# See config/schema.py for REQUIRED_FIELDS, NUMERIC_FIELDS, and FIELD_ORDER
