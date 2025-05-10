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
        result = result.fillna(0.0)
        
        # Convert to numeric values
        return result.apply(lambda x: self._normalize_price_value(x, field_name))
    
    def _normalize_price_value(self, value: Any, field_name: str) -> float:
        """
        Normalize a single price value
        
        Args:
            value (Any): Input price value
            field_name (str): Name of the price field
            
        Returns:
            float: Normalized price value
        """
        if pd.isna(value) or value is None:
            return 0.0
            
        # Convert to string first
        value_str = str(value).strip()
        
        # If empty, return 0
        if not value_str:
            return 0.0
            
        try:
            # If already numeric, just convert to float
            return float(value)
        except ValueError:
            # Try to parse as formatted price
            value_str = self._clean_price_string(value_str)
            
            try:
                return float(value_str)
            except ValueError:
                # If all else fails, return 0
                self.logger.warning(f"Could not parse price value: {value}")
                return 0.0
    
    def _clean_price_string(self, value_str: str) -> str:
        """
        Clean a price string for parsing
        
        Args:
            value_str (str): Input price string
            
        Returns:
            str: Cleaned price string
        """
        # Remove currency symbols
        value_str = re.sub(r'[$£€¥]', '', value_str)
        
        # Remove grouping commas (e.g., 1,000.00 -> 1000.00)
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
        
        return value_str