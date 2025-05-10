# core/chunking/category_chunker.py
"""
Category-aware chunking implementation
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from core.chunking.chunker import BaseChunker
from config.settings import APP_CONFIG

class CategoryChunker(BaseChunker):
    """Chunking strategy that respects category boundaries"""
    
    def __init__(self, chunk_size=None, category_column=None):
        """
        Initialize the category chunker
        
        Args:
            chunk_size (int, optional): Target number of rows per chunk
            category_column (str, optional): Column name for category grouping
        """
        chunk_size = chunk_size or APP_CONFIG.get('chunk_size', 100)
        super().__init__(chunk_size)
        self.category_column = category_column
        self.logger = logging.getLogger(__name__)
    
    def _detect_category_column(self, data: pd.DataFrame) -> Optional[str]:
        """
        Automatically detect a likely category column
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            Optional[str]: Detected category column name or None
        """
        # Category-related keywords
        category_keywords = ['category', 'group', 'type', 'class', 'department']
        
        # Check for columns with category-like names
        for keyword in category_keywords:
            matches = [col for col in data.columns 
                      if keyword.lower() in str(col).lower()]
            if matches:
                return matches[0]
        
        # Look for columns with repeated values that could be categories
        for col in data.columns:
            if data[col].dtype == object:  # String-type columns
                # Count unique values and check if it's a reasonable number for categories
                unique_count = data[col].nunique()
                total_count = len(data)
                
                # Categories typically have fewer unique values than 25% of total rows
                # and at least 2 rows per category on average
                if 1 < unique_count < (total_count * 0.25) and total_count / unique_count >= 2:
                    return col
        
        return None
    
    def chunk_data(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Split data into chunks while respecting category boundaries
        
        Args:
            data (pd.DataFrame): Input data to chunk
            
        Returns:
            List[pd.DataFrame]: List of data chunks
        """
        if len(data) <= self.chunk_size:
            self.logger.debug(f"Data fits in a single chunk ({len(data)} rows)")
            return [data]
        
        # Detect category column if not specified
        category_col = self.category_column
        if not category_col:
            category_col = self._detect_category_column(data)
            
        if category_col and category_col in data.columns:
            self.logger.info(f"Using category column: {category_col}")
            return self._chunk_by_category(data, category_col)
        else:
            self.logger.info("No suitable category column found, using row-based chunking")
            return self._chunk_by_rows(data)
    
    def _chunk_by_category(self, data: pd.DataFrame, category_col: str) -> List[pd.DataFrame]:
        """
        Create chunks that keep rows of the same category together
        
        Args:
            data (pd.DataFrame): Input data
            category_col (str): Category column name
            
        Returns:
            List[pd.DataFrame]: List of data chunks
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        # Group by category
        grouped = data.groupby(category_col)
        
        for category, group in grouped:
            group_size = len(group)
            
            # If a single category is too large, split it
            if group_size > self.chunk_size * 2:
                self.logger.debug(f"Large category '{category}' with {group_size} rows, splitting")
                category_chunks = self._chunk_by_rows(group)
                chunks.extend(category_chunks)
                continue
                
                            # If adding this category would make the chunk too large, start a new chunk
            if current_size + group_size > self.chunk_size and current_size > 0:
                chunks.append(pd.concat(current_chunk, ignore_index=True))
                current_chunk = []
                current_size = 0
                
            # Add this category group to the current chunk
            current_chunk.append(group)
            current_size += group_size
            self.logger.debug(f"Added category '{category}' with {group_size} rows to chunk")
            
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append(pd.concat(current_chunk, ignore_index=True))
            
        self.logger.info(f"Created {len(chunks)} category-aware chunks")
        return chunks
    
    def _chunk_by_rows(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Fallback to row-based chunking
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            List[pd.DataFrame]: List of data chunks
        """
        num_chunks = int(np.ceil(len(data) / self.chunk_size))
        self.logger.info(f"Splitting {len(data)} rows into {num_chunks} chunks")
        
        chunks = []
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(data))
            
            chunk_df = data.iloc[start_idx:end_idx].copy()
            chunks.append(chunk_df)
            
        return chunks
    
    def merge_results(self, chunked_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from processed chunks
        
        Args:
            chunked_results (List[Dict[str, Any]]): Results from each chunk
            
        Returns:
            Dict[str, Any]: Merged results
        """
        # Same implementation as RowChunker
        if not chunked_results:
            return {}
            
        merged_results = {}
        
        # Collect all keys
        all_keys = set()
        for result in chunked_results:
            all_keys.update(result.keys())
            
        # Process each key
        for key in all_keys:
            # If all chunks have this key and values are lists, concatenate
            values = [r.get(key) for r in chunked_results if key in r]
            
            if all(isinstance(v, list) for v in values):
                # Concatenate lists
                merged_results[key] = [item for sublist in values for item in sublist]
            elif all(isinstance(v, (pd.DataFrame, pd.Series)) for v in values):
                # Concatenate DataFrames/Series
                merged_results[key] = pd.concat(values)
            elif all(isinstance(v, dict) for v in values):
                # Merge dictionaries recursively
                merged_dict = {}
                for v in values:
                    merged_dict.update(v)
                merged_results[key] = merged_dict
            else:
                # For other types, use the last value
                merged_results[key] = values[-1]
                
        return merged_results