# core/file_parser/json_parser.py
"""
JSON-specific parsing implementation
"""
import logging
import pandas as pd
import json
from typing import Dict, Any, List, Optional

from core.file_parser.base_parser import BaseParser
from config.settings import PARSER_CONFIG

class JSONParser(BaseParser):
    """Parser for JSON files"""

    def __init__(self, file_path):
        """
        Initialize the JSON parser

        Args:
            file_path (str): Path to the JSON file
        """
        super().__init__(file_path)
        self.logger = logging.getLogger(__name__)

        # Add JSON config to PARSER_CONFIG if not present
        if 'json' not in PARSER_CONFIG:
            PARSER_CONFIG['json'] = {
                "flatten_nested": True,
                "normalize_arrays": True
            }

        self.config = PARSER_CONFIG.get('json', {})

    def parse(self) -> pd.DataFrame:
        """
        Parse the JSON file into a pandas DataFrame

        Returns:
            pd.DataFrame: Parsed data
        """
        self.logger.info(f"Parsing JSON file: {self.file_path}")

        try:
            # Detect encoding
            encoding = self.encoding or self.detect_encoding()

            # Read the JSON file
            with open(self.file_path, 'r', encoding=encoding) as f:
                data = json.load(f)

            # Convert to DataFrame
            df = self._json_to_dataframe(data)

            # Apply common preprocessing
            df = self.preprocess_dataframe(df)

            self.logger.info(f"Successfully parsed JSON with {len(df)} rows and {len(df.columns)} columns")
            return df

        except Exception as e:
            self.logger.error(f"Error parsing JSON file: {str(e)}")
            return pd.DataFrame()

    def _json_to_dataframe(self, data: Any) -> pd.DataFrame:
        """
        Convert JSON data to a pandas DataFrame

        Args:
            data: Parsed JSON data

        Returns:
            pd.DataFrame: Converted DataFrame
        """
        # Handle different JSON structures
        if isinstance(data, dict):
            # Single object - convert to a single-row DataFrame
            return self._process_dict(data)

        elif isinstance(data, list):
            # Array of objects - convert to multi-row DataFrame
            return self._process_list(data)

        else:
            # Unexpected structure
            self.logger.warning(f"Unexpected JSON structure: {type(data)}")
            return pd.DataFrame()

    def _process_dict(self, data: Dict) -> pd.DataFrame:
        """
        Process a dictionary into a DataFrame

        Args:
            data (Dict): Dictionary to process

        Returns:
            pd.DataFrame: Single-row DataFrame
        """
        # If it's a nested structure, flatten it
        if self.config.get('flatten_nested', True):
            flattened = self._flatten_dict(data)
            return pd.DataFrame([flattened])
        else:
            return pd.DataFrame([data])

    def _process_list(self, data: List) -> pd.DataFrame:
        """
        Process a list into a DataFrame

        Args:
            data (List): List to process

        Returns:
            pd.DataFrame: Multi-row DataFrame
        """
        if not data:
            return pd.DataFrame()

        # Check if list contains dictionaries
        if all(isinstance(item, dict) for item in data):
            # If all items are dictionaries, convert to DataFrame
            if self.config.get('flatten_nested', True):
                # Flatten each dictionary
                flattened_data = [self._flatten_dict(item) for item in data]
                return pd.DataFrame(flattened_data)
            else:
                return pd.DataFrame(data)

        elif self.config.get('normalize_arrays', True):
            # For non-dictionary lists, try to normalize
            return self._normalize_array(data)

        else:
            # Convert to a single column DataFrame
            return pd.DataFrame({'value': data})

    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """
        Flatten a nested dictionary

        Args:
            d (Dict): Dictionary to flatten
            parent_key (str): Parent key for nested items
            sep (str): Separator for nested keys

        Returns:
            Dict: Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k

            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
                # For lists of dictionaries, add indices
                for i, item in enumerate(v):
                    items.extend(self._flatten_dict(item, f"{new_key}{sep}{i}", sep).items())
            else:
                items.append((new_key, v))

        return dict(items)

    def _normalize_array(self, data: List) -> pd.DataFrame:
        """
        Normalize a list into a DataFrame

        Args:
            data (List): List to normalize

        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        # Try to convert to a DataFrame using pandas normalization
        try:
            # For lists of mixed types, convert to strings
            if not all(isinstance(item, type(data[0])) for item in data):
                data = [str(item) for item in data]

            return pd.DataFrame({'value': data})

        except Exception as e:
            self.logger.error(f"Error normalizing array: {str(e)}")
            return pd.DataFrame({'value': [str(item) for item in data]})
