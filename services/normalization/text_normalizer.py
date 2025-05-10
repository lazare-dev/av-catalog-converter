# services/normalization/text_normalizer.py
"""
Text field normalization
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple
import re
import html

class TextNormalizer:
    """Normalizes text fields like descriptions"""
    
    def __init__(self):
        """Initialize the text normalizer"""
        self.logger = logging.getLogger(__name__)
    
    def normalize_text(self, series: pd.Series) -> pd.Series:
        """
        Normalize text fields
        
        Args:
            series (pd.Series): Text field values
            
        Returns:
            pd.Series: Normalized text
        """
        # Handle missing values
        result = series.copy()
        result = result.fillna("")
        
        # Apply text normalization rules
        return result.apply(self._normalize_text_value)
    
    def _normalize_text_value(self, value: Any) -> str:
        """
        Normalize a single text value
        
        Args:
            value (Any): Input value
            
        Returns:
            str: Normalized text
        """
        if pd.isna(value) or value is None:
            return ""
            
        # Convert to string
        text = str(value)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove ASCII control characters
        text = re.sub(r'[\x00-\x1F\x7F]', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Fix common typos and abbreviations
        text = self._fix_common_typos(text)
        
        return text
    
    def _fix_common_typos(self, text: str) -> str:
        """
        Fix common typos and abbreviations in text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Corrected text
        """
        # Common corrections
        corrections = {
            r'\bw/\b': 'with',
            r'\bw/o\b': 'without',
            r'\bhd\b': 'HD',
            r'\bfhd\b': 'FHD',
            r'\buhd\b': 'UHD',
            r'\b4k\b': '4K',
            r'\b8k\b': '8K',
            r'\blcd\b': 'LCD',
            r'\bled\b': 'LED',
            r'\boled\b': 'OLED',
            r'\bavr\b': 'AVR',
            r'\bhdmi\b': 'HDMI',
            r'\busb\b': 'USB',
            r'\bpts\b': 'pts',
            r'\binch\b': '"',
            r'\bft\b': 'ft',
            r'\binfo\b': 'information',
            r'\bincl\b': 'including',
            r'\bincl\.\b': 'including',
            r'\bapprox\b': 'approximately',
            r'\bapprox\.\b': 'approximately',
            r'\btech\b': 'technology',
            r'\btech\.\b': 'technology',
            r'\bspkr\b': 'speaker',
            r'\bmic\b': 'microphone',
            r'\bproj\b': 'projector',
            r'\bproj\.\b': 'projector',
            r'\bdispl\b': 'display',
            r'\bdispl\.\b': 'display'
        }
        
        # Apply corrections
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
        return text
