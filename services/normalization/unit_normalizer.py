# services/normalization/unit_normalizer.py
"""
Units standardization
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple
import re

class UnitNormalizer:
    """Normalizes unit of measure fields"""
    
    def __init__(self):
        """Initialize the unit normalizer"""
        self.logger = logging.getLogger(__name__)
        
        # Standard units and their variations
        self.unit_mapping = {
            "Each": ["each", "ea", "piece", "pc", "pcs", "item", "unit", "single"],
            "Pair": ["pair", "pr", "pairs", "set of 2", "couple"],
            "Set": ["set", "kit", "bundle", "package", "system"],
            "Box": ["box", "carton", "pack", "package", "container"],
            "Case": ["case", "carrying case", "cse"],
            "Meter": ["meter", "m", "mtr", "metre"],
            "Foot": ["foot", "ft", "feet"],
            "Roll": ["roll", "rl", "reel"],
            "Lot": ["lot", "batch", "group"],
            "Hour": ["hour", "hr", "hrs", "service hour"]
        }
        
        # Create reverse mapping for lookup
        self.reverse_mapping = {}
        for std_unit, variations in self.unit_mapping.items():
            for var in variations:
                self.reverse_mapping[var] = std_unit
            # Also map the standard unit to itself
            self.reverse_mapping[std_unit.lower()] = std_unit
    
    def normalize_unit(self, series: pd.Series) -> pd.Series:
        """
        Normalize unit of measure values
        
        Args:
            series (pd.Series): Unit field values
            
        Returns:
            pd.Series: Normalized units
        """
        # Handle missing values
        result = series.copy()
        result = result.fillna("Each")
        
        # Apply normalization
        return result.apply(self._normalize_unit_value)
    
    def _normalize_unit_value(self, value: Any) -> str:
        """
        Normalize a single unit value
        
        Args:
            value (Any): Input unit
            
        Returns:
            str: Normalized unit
        """
        if pd.isna(value) or value is None:
            return "Each"
            
        # Convert to string and clean
        unit = str(value).strip().lower()
        
        # Remove common prefixes like "sold by", "per", etc.
        unit = re.sub(r'^(sold by|sold as|per|in|by|unit is|unit of|units of)\s+', '', unit)
        
        # Look up in mapping
        if unit in self.reverse_mapping:
            return self.reverse_mapping[unit]
            
        # Try to match with variations by removing plurals, etc.
        if unit.endswith('s') and unit[:-1] in self.reverse_mapping:
            return self.reverse_mapping[unit[:-1]]
            
        # Default to "Each" if no match found
        return "Each"