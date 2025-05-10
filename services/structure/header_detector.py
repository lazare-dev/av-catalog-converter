# services/structure/header_detector.py
"""
Header detection logic
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Optional

class HeaderDetector:
    """Detect and validate header information in data"""
    
    def __init__(self):
        """Initialize the header detector"""
        self.logger = logging.getLogger(__name__)
    
    def detect_headers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze headers in the data
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Header information
        """
        self.logger.info("Detecting header information")
        
        header_info = {
            "original_headers": list(data.columns),
            "cleaned_headers": self._clean_headers(data.columns),
            "potential_issues": self._detect_header_issues(data),
            "duplicate_headers": self._find_duplicate_headers(data.columns),
            "empty_columns": self._find_empty_columns(data)
        }
        
        return header_info
    
    def _clean_headers(self, headers: List) -> List[str]:
        """
        Clean and normalize header names
        
        Args:
            headers (List): Original headers
            
        Returns:
            List[str]: Cleaned headers
        """
        cleaned = []
        
        for header in headers:
            # Convert to string and strip whitespace
            header_str = str(header).strip()
            
            # Remove special characters
            for char in ['\n', '\r', '\t', '#', '*', '@', '%', '^', '&']:
                header_str = header_str.replace(char, '')
                
            # Replace spaces with underscores
            header_str = header_str.replace(' ', '_')
            
            # Ensure it's not empty
            if not header_str:
                header_str = "unnamed_column"
                
            cleaned.append(header_str)
            
        return cleaned
    
    def _detect_header_issues(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Detect potential issues with the headers
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            Dict[str, List[str]]: Header issues by type
        """
        issues = {
            "non_descriptive": [],
            "inconsistent_format": [],
            "likely_data_as_header": []
        }
        
        for col in data.columns:
            col_str = str(col)
            
            # Check for non-descriptive headers
            if col_str in ["", "Unnamed", "Column1", "Field1"] or pd.isna(col):
                issues["non_descriptive"].append(col_str)
                
            # Check for inconsistent format (mixed case, spaces, etc.)
            if ' ' in col_str and '_' in col_str:
                issues["inconsistent_format"].append(col_str)
                
            # Check if header might actually be data
            # Headers are typically not numeric
            try:
                float(col_str)
                issues["likely_data_as_header"].append(col_str)
            except (ValueError, TypeError):
                pass
                
            # Headers are typically not very long
            if len(col_str) > 30:
                issues["likely_data_as_header"].append(col_str)
        
        return issues
    
    def _find_duplicate_headers(self, headers: List) -> Dict[str, List[int]]:
        """
        Find duplicate header names
        
        Args:
            headers (List): Original headers
            
        Returns:
            Dict[str, List[int]]: Duplicate headers with their positions
        """
        duplicates = {}
        seen = {}
        
        for i, header in enumerate(headers):
            header_str = str(header).strip()
            
            if header_str in seen:
                if header_str not in duplicates:
                    duplicates[header_str] = [seen[header_str]]
                duplicates[header_str].append(i)
            else:
                seen[header_str] = i
                
        return duplicates
    
    def _find_empty_columns(self, data: pd.DataFrame) -> List[str]:
        """
        Find completely empty columns
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            List[str]: Empty column names
        """
        empty_cols = []
        
        for col in data.columns:
            if data[col].isna().all():
                empty_cols.append(str(col))
                
        return empty_cols