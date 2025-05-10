# core/chunking/row_chunker.py
"""
Row-based chunking implementation
"""
import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from core.chunking.chunker import BaseChunker
from config.settings import APP_CONFIG

class RowChunker(BaseChunker):
    """Simple row-based chunking strategy"""
    
    def __init__(self, chunk_size=None):
        """
        Initialize the row chunker
        
        Args:
            chunk_size (int, optional): Number of rows per chunk
        """
        chunk_size = chunk_size or APP_CONFIG.get('chunk_size', 100)
        super().__init__(chunk_size)
        self.logger = logging.getLogger(__name__)
    
    def chunk_data(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Split data into chunks by row count
        
        Args:
            data (pd.DataFrame): Input data to chunk
            
        Returns:
            List[pd.DataFrame]: List of data chunks
        """
        if len(data) <= self.chunk_size:
            self.logger.debug(f"Data fits in a single chunk ({len(data)} rows)")
            return [data]
            
        # Calculate chunks
        num_chunks = int(np.ceil(len(data) / self.chunk_size))
        self.logger.info(f"Splitting {len(data)} rows into {num_chunks} chunks of {self.chunk_size} rows")
        
        chunks = []
        for i in range(num_chunks):
            start_idx = i * self.chunk_size
            end_idx = min((i + 1) * self.chunk_size, len(data))
            
            chunk_df = data.iloc[start_idx:end_idx].copy()
            chunks.append(chunk_df)
            
            self.logger.debug(f"Created chunk {i+1}/{num_chunks}: rows {start_idx}-{end_idx-1}")
            
        return chunks
    
    def merge_results(self, chunked_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from processed chunks
        
        Args:
            chunked_results (List[Dict[str, Any]]): Results from each chunk
            
        Returns:
            Dict[str, Any]: Merged results
        """
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
