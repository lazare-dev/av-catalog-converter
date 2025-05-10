# core/chunking/chunker.py
"""
Base chunking strategy
"""
import logging
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseChunker(ABC):
    """Abstract base class for data chunking strategies"""
    
    def __init__(self, chunk_size=100):
        """
        Initialize the chunker
        
        Args:
            chunk_size (int): Number of rows per chunk
        """
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def chunk_data(self, data: pd.DataFrame) -> List[pd.DataFrame]:
        """
        Split data into manageable chunks
        
        Args:
            data (pd.DataFrame): Input data to chunk
            
        Returns:
            List[pd.DataFrame]: List of data chunks
        """
        pass
    
    def merge_results(self, chunked_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge results from processed chunks
        
        Args:
            chunked_results (List[Dict[str, Any]]): Results from each chunk
            
        Returns:
            Dict[str, Any]: Merged results
        """
        # Default implementation - override as needed
        if not chunked_results:
            return {}
            
        # Merge dictionaries, last one wins for duplicate keys
        merged = {}
        for result in chunked_results:
            merged.update(result)
            
        return merged