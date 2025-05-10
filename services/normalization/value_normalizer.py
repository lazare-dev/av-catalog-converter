# services/normalization/value_normalizer.py
"""
Value normalization orchestrator
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple

from services.normalization.text_normalizer import TextNormalizer
from services.normalization.price_normalizer import PriceNormalizer
from services.normalization.id_normalizer import IDNormalizer
from services.normalization.unit_normalizer import UnitNormalizer
from config.schema import FIELD_ORDER, SCHEMA_DICT

class ValueNormalizer:
    """Service for normalizing values across the standardized schema"""
    
    def __init__(self):
        """Initialize the value normalizer"""
        self.logger = logging.getLogger(__name__)
        self.text_normalizer = TextNormalizer()
        self.price_normalizer = PriceNormalizer()
        self.id_normalizer = IDNormalizer()
        self.unit_normalizer = UnitNormalizer()
    
    def normalize(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize values according to field types
        
        Args:
            data (pd.DataFrame): Input data with mapped fields
            
        Returns:
            pd.DataFrame: Data with normalized values
        """
        self.logger.info("Starting value normalization")
        
        # Make a copy to avoid modifying the original
        result_df = data.copy()
        
        # Process each field according to its type
        for field in FIELD_ORDER:
            if field not in result_df.columns:
                continue
                
            field_def = SCHEMA_DICT.get(field)
            if not field_def:
                continue
                
            # Determine field type and apply appropriate normalization
            if field in ["Short Description", "Long Description", "Document Name"]:
                result_df[field] = self.text_normalizer.normalize_text(result_df[field])
                
            elif field in ["Buy Cost", "Trade Price", "MSRP GBP", "MSRP USD", "MSRP EUR"]:
                result_df[field] = self.price_normalizer.normalize_price(result_df[field], field)
                
            elif field in ["SKU", "Model", "Manufacturer SKU"]:
                result_df[field] = self.id_normalizer.normalize_id(result_df[field], field)
                
            elif field == "Unit Of Measure":
                result_df[field] = self.unit_normalizer.normalize_unit(result_df[field])
                
            elif field in ["Image URL", "Document URL"]:
                result_df[field] = self.normalize_urls(result_df[field])
                
            elif field == "Discontinued":
                result_df[field] = self.normalize_boolean(result_df[field])
                
            elif field == "Manufacturer":
                result_df[field] = self.normalize_manufacturer(result_df[field])
        
        self.logger.info("Value normalization completed")
        return result_df
    
    def normalize_urls(self, series: pd.Series) -> pd.Series:
        """
        Normalize URL values
        
        Args:
            series (pd.Series): URL field values
            
        Returns:
            pd.Series: Normalized URLs
        """
        # Handle missing values
        result = series.copy()
        result = result.fillna("")
        
        # Ensure scheme is present
        def add_scheme(url):
            if pd.isna(url) or not url:
                return ""
            
            url = str(url).strip()
            
            if url and not url.startswith(('http://', 'https://')):
                return f"https://{url}"
            return url
            
        return result.apply(add_scheme)
    
    def normalize_boolean(self, series: pd.Series) -> pd.Series:
        """
        Normalize boolean/flag values
        
        Args:
            series (pd.Series): Boolean field values
            
        Returns:
            pd.Series: Normalized values ("Yes"/"No")
        """
        # Handle missing values
        result = series.copy()
        result = result.fillna("No")
        
        # Map various boolean formats to Yes/No
        def to_yes_no(val):
            if pd.isna(val):
                return "No"
                
            val_str = str(val).strip().lower()
            
            if val_str in ['yes', 'true', 'y', 't', '1', 'discontinued']:
                return "Yes"
            else:
                return "No"
                
        return result.apply(to_yes_no)
    
    def normalize_manufacturer(self, series: pd.Series) -> pd.Series:
        """
        Normalize manufacturer names
        
        Args:
            series (pd.Series): Manufacturer field values
            
        Returns:
            pd.Series: Normalized manufacturer names
        """
        # Handle missing values
        result = series.copy()
        result = result.fillna("Unknown")
        
        # Normalize common manufacturer names
        common_manufacturers = {
            'sony': 'Sony',
            'samsung': 'Samsung',
            'lg': 'LG',
            'panasonic': 'Panasonic',
            'bose': 'Bose',
            'jbl': 'JBL',
            'harman': 'Harman',
            'harman kardon': 'Harman Kardon',
            'crestron': 'Crestron',
            'extron': 'Extron',
            'amx': 'AMX',
            'epson': 'Epson',
            'christie': 'Christie',
            'barco': 'Barco',
            'shure': 'Shure',
            'sennheiser': 'Sennheiser',
            'logitech': 'Logitech',
            'polycom': 'Polycom',
            'cisco': 'Cisco',
            'denon': 'Denon',
            'marantz': 'Marantz',
            'yamaha': 'Yamaha'
        }
        
        def normalize_mfr(val):
            if pd.isna(val) or not val:
                return "Unknown"
                
            val_str = str(val).strip()
            val_lower = val_str.lower()
            
            if val_lower in common_manufacturers:
                return common_manufacturers[val_lower]
            else:
                # Capitalize first letter of each word
                return ' '.join(word.capitalize() for word in val_str.split())
                
        return result.apply(normalize_mfr)
