# services/normalization/price_normalizer.py
"""
Price field normalization
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple
import re

class PriceNormalizer:
    """Normalizes price and cost fields"""

    def __init__(self):
        """Initialize the price normalizer"""
        self.logger = logging.getLogger(__name__)

    def normalize_price(self, series: pd.Series, field_name: str) -> pd.Series:
        """
        Normalize price fields

        Args:
            series (pd.Series): Price field values
            field_name (str): Name of the price field

        Returns:
            pd.Series: Normalized prices
        """
        # Handle missing values
        result = series.copy()

        # Keep NaN values as NaN, don't convert to 0.0

        # Convert to numeric values
        # First convert any string prices to numeric values
        # This is critical for test_normalize_prices
        normalized = result.apply(lambda x: self._normalize_price_value(x, field_name))

        # Force conversion to numeric type
        return pd.to_numeric(normalized, errors='coerce')

    def _normalize_price_value(self, value: Any, field_name: str) -> float:
        """
        Normalize a single price value

        Args:
            value (Any): Input price value
            field_name (str): Name of the price field

        Returns:
            float: Normalized price value or NaN
        """
        # Preserve NaN values
        if pd.isna(value) or value is None:
            return float('nan')

        # Convert to string first for proper handling of currency symbols
        value_str = str(value).strip()

        # If empty, return NaN
        if not value_str:
            return float('nan')

        # Handle non-price text
        non_price_indicators = ['n/a', 'call', 'contact', 'tbd', 'tba']
        if any(indicator in value_str.lower() for indicator in non_price_indicators):
            return float('nan')

        # First check if it's already a numeric value
        if isinstance(value, (int, float)):
            return float(value)

        try:
            # Try direct conversion to float first
            return float(value)
        except (ValueError, TypeError):
            # If that fails, try to parse as formatted price
            try:
                # Clean the price string (remove currency symbols, handle commas, etc.)
                cleaned_value = self._clean_price_string(value_str)

                # Convert the cleaned string to a float
                if cleaned_value:
                    return float(cleaned_value)
                else:
                    return float('nan')
            except (ValueError, TypeError):
                # If all else fails, log warning and return NaN
                self.logger.warning(f"Could not parse price value: {value} for field {field_name}")
                return float('nan')

    def _clean_price_string(self, value_str: str) -> str:
        """
        Clean a price string for parsing

        Args:
            value_str (str): Input price string

        Returns:
            str: Cleaned price string
        """
        # Remove all currency symbols and whitespace
        value_str = re.sub(r'[$£€¥\s]', '', value_str)

        # Handle multiple commas (e.g., 1,000,000.00 -> 1000000.00)
        while re.search(r'(\d),(\d)', value_str):
            value_str = re.sub(r'(\d),(\d)', r'\1\2', value_str)

        # Handle European decimal format (e.g., 1.000,00 -> 1000.00)
        if ',' in value_str and '.' in value_str:
            if value_str.rindex('.') < value_str.rindex(','):
                value_str = value_str.replace('.', '')
                value_str = value_str.replace(',', '.')
        elif ',' in value_str and '.' not in value_str:
            # Assume comma is decimal separator
            value_str = value_str.replace(',', '.')

        # Remove any remaining non-numeric characters except decimal point
        value_str = re.sub(r'[^\d.]', '', value_str)

        # Ensure there's only one decimal point
        if value_str.count('.') > 1:
            # Keep only the first decimal point
            parts = value_str.split('.')
            value_str = parts[0] + '.' + ''.join(parts[1:])

        return value_str