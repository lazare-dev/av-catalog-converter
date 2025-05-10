# services/structure/data_boundary_detector.py
"""
Data boundaries detection
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Set, Tuple

class DataBoundaryDetector:
    """Detect logical boundaries in the data"""
    
    def __init__(self):
        """Initialize the data boundary detector"""
        self.logger = logging.getLogger(__name__)
    
    def detect_boundaries(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect logical boundaries in the data
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Boundary information
        """
        self.logger.info("Detecting data boundaries")
        
        boundaries = {
            "sections": self._detect_sections(data),
            "empty_rows": self._find_empty_rows(data),
            "header_rows": self._find_header_rows(data),
            "summary_rows": self._find_summary_rows(data)
        }
        
        return boundaries
    
    def _detect_sections(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect logical sections in the data
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            List[Dict[str, Any]]: Detected sections
        """
        sections = []
        current_section = None
        section_start = 0
        
        # Look for pattern changes that might indicate sections
        for i in range(1, len(data)):
            # Check for empty rows as section separators
            if self._is_empty_row(data.iloc[i]):
                if current_section is not None:
                    # End current section
                    sections.append({
                        "start": section_start,
                        "end": i - 1,
                        "length": i - section_start,
                        "type": current_section
                    })
                    current_section = None
                continue
                
            # Check for header-like rows as section beginnings
            if self._is_header_like_row(data.iloc[i]):
                if current_section is not None:
                    # End current section
                    sections.append({
                        "start": section_start,
                        "end": i - 1,
                        "length": i - section_start,
                        "type": current_section
                    })
                
                # Start new section
                current_section = "data"
                section_start = i + 1  # Start after the header row
                continue
                
            # If no section is currently active, start one
            if current_section is None:
                current_section = "data"
                section_start = i
                
        # Add the final section if active
        if current_section is not None:
            sections.append({
                "start": section_start,
                "end": len(data) - 1,
                "length": len(data) - section_start,
                "type": current_section
            })
            
        return sections
    
    def _find_empty_rows(self, data: pd.DataFrame) -> List[int]:
        """
        Find completely empty rows
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            List[int]: Empty row indices
        """
        empty_rows = []
        
        for i in range(len(data)):
            if self._is_empty_row(data.iloc[i]):
                empty_rows.append(i)
                
        return empty_rows
    
    def _is_empty_row(self, row: pd.Series) -> bool:
        """
        Check if a row is completely empty
        
        Args:
            row (pd.Series): Data row
            
        Returns:
            bool: True if the row is empty
        """
        return row.isna().all()
    
    def _find_header_rows(self, data: pd.DataFrame) -> List[int]:
        """
        Find rows that appear to be headers within the data
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            List[int]: Header row indices
        """
        header_rows = []
        
        for i in range(1, len(data)):
            if self._is_header_like_row(data.iloc[i]):
                header_rows.append(i)
                
        return header_rows
    
    def _is_header_like_row(self, row: pd.Series) -> bool:
        """
        Check if a row appears to be a header
        
        Args:
            row (pd.Series): Data row
            
        Returns:
            bool: True if the row appears to be a header
        """
        # Headers typically:
        # 1. Have mostly string values
        # 2. Have shorter text than data rows
        # 3. Have no numeric values
        
        # Count string values
        string_count = sum(1 for val in row if isinstance(val, str))
        
        # Count total non-NaN values
        non_nan_count = sum(1 for val in row if not pd.isna(val))
        
        if non_nan_count == 0:
            return False
            
        # Calculate ratio of string values
        string_ratio = string_count / non_nan_count
        
        # Check average length of string values
        avg_len = 0
        if string_count > 0:
            total_len = sum(len(str(val)) for val in row if isinstance(val, str))
            avg_len = total_len / string_count
            
        # Check for numeric values
        has_numeric = any(isinstance(val, (int, float)) and not pd.isna(val) for val in row)
        
        # Determine if this looks like a header
        return string_ratio > 0.8 and avg_len < 20 and not has_numeric
    
    def _find_summary_rows(self, data: pd.DataFrame) -> List[int]:
        """
        Find rows that appear to be summaries (e.g., totals)
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            List[int]: Summary row indices
        """
        summary_rows = []
        
        # Look for rows that might be summaries (often at the end of sections)
        summary_keywords = ['total', 'sum', 'average', 'subtotal', 'grand total']
        
        for i in range(len(data)):
            row = data.iloc[i]
            
            # Check for summary keywords
            has_keyword = any(
                isinstance(val, str) and any(kw in str(val).lower() for kw in summary_keywords)
                for val in row
            )
            
            # Summary rows often have numeric values
            has_numbers = any(
                isinstance(val, (int, float)) and not pd.isna(val)
                for val in row
            )
            
            if has_keyword and has_numbers:
                summary_rows.append(i)
                
        return summary_rows