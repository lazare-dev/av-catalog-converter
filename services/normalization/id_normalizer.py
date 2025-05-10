# services/normalization/id_normalizer.py
"""
ID/SKU normalization
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple
import re

class IDNormalizer:
    """Normalizes ID, SKU, and code fields"""
    
    def __init__(self):
        """Initialize the ID normalizer"""
        self.logger = logging.getLogger(__name__)
    
    def normalize_id(self, series: pd.Series, field_name: str) -> pd.Series:
        """
        Normalize ID/SKU fields
        
        Args:
            series (pd.Series): ID field values
            field_name (str): Name of the ID field
            
        Returns:
            pd.Series: Normalized IDs
        """
        # Handle missing values
        result = series.copy()
        result = result.fillna("")
        
        # Apply specific normalization based on field type
        if field_name == "SKU":
            return result.apply(self._normalize_sku)
        elif field_name == "Model":
            return result.apply(self._normalize_model)
        elif field_name == "Manufacturer SKU":
            return result.apply(self._normalize_manufacturer_sku)
        else:
            return result.apply(self._normalize_generic_id)
    
    def _normalize_sku(self, value: Any) -> str:
        """
        Normalize a SKU value
        
        Args:
            value (Any): Input SKU
            
        Returns:
            str: Normalized SKU
        """
        if pd.isna(value) or value is None:
            return ""
            
        # Convert to string
        sku = str(value).strip()
        
        # Remove unwanted characters
        sku = re.sub(r'[\s"\']', '', sku)
        
        # Ensure uppercase for standard format
        sku = sku.upper()
        
        return sku
    
    def _normalize_model(self, value: Any) -> str:
        """
        Normalize a model number
        
        Args:
            value (Any): Input model number
            
        Returns:
            str: Normalized model number
        """
        if pd.isna(value) or value is None:
            return ""
            
        # Convert to string
        model = str(value).strip()
        
        # Remove unwanted characters while preserving hyphens
        model = re.sub(r'[\s"\']', '', model)
        
        # Ensure consistent format (typically uppercase for model numbers)
        model = model.upper()
        
        return model
    
    def _normalize_manufacturer_sku(self, value: Any) -> str:
        """
        Normalize a manufacturer SKU
        
        Args:
            value (Any): Input manufacturer SKU
            
        Returns:
            str: Normalized manufacturer SKU
        """
        if pd.isna(value) or value is None:
            return ""
            
        # Convert to string
        mfr_sku = str(value).strip()
        
        # Preserve format but remove unwanted characters
        mfr_sku = re.sub(r'[\s"\']', '', mfr_sku)
        
        # Some manufacturers use specific formats, which we should preserve
        return mfr_sku
    
    def _normalize_generic_id(self, value: Any) -> str:
        """
        Normalize a generic ID value
        
        Args:
            value (Any): Input ID
            
        Returns:
            str: Normalized ID
        """
        if pd.isna(value) or value is None:
            return ""
            
        # Convert to string
        id_value = str(value).strip()
        
        # Remove unwanted characters
        id_value = re.sub(r'[\s"\']', '', id_value)
        
        return id_value
