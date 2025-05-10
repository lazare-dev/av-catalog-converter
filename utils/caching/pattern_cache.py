# utils/caching/pattern_cache.py
"""
Mapping pattern caching
"""
import logging
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import time

class PatternCache:
    """Cache for field mapping patterns"""
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the pattern cache
        
        Args:
            cache_dir (str, optional): Directory for cache files
        """
        self.logger = logging.getLogger(__name__)
        
        # Use default cache directory if not specified
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'cache', 'patterns')
            
        self.cache_dir = Path(cache_dir)
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_mappings(self, key: str) -> Dict[str, str]:
        """
        Get cached field mappings
        
        Args:
            key (str): Cache key (e.g., manufacturer name)
            
        Returns:
            Dict[str, str]: Cached mappings or empty dict if not found
        """
        # Normalize key
        key = self._normalize_key(key)
        cache_path = self.cache_dir / f"{key}.json"
        
        # Check if cache file exists
        if not cache_path.exists():
            return {}
            
        try:
            # Load cached mappings
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            mappings = data.get('mappings', {})
            self.logger.debug(f"Loaded {len(mappings)} cached mappings for '{key}'")
            return mappings
            
        except (OSError, json.JSONDecodeError) as e:
            self.logger.error(f"Error loading cached mappings: {str(e)}")
            return {}
    
    def store_mappings(self, key: str, mappings: Dict[str, str]):
        """
        Store field mappings in cache
        
        Args:
            key (str): Cache key (e.g., manufacturer name)
            mappings (Dict[str, str]): Field mappings to store
        """
        # Normalize key
        key = self._normalize_key(key)
        cache_path = self.cache_dir / f"{key}.json"
        
        try:
            # Store mappings with metadata
            data = {
                'mappings': mappings,
                'timestamp': time.time(),
                'mapping_count': len(mappings)
            }
            
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
            self.logger.debug(f"Stored {len(mappings)} mappings for '{key}'")
            
        except OSError as e:
            self.logger.error(f"Error storing mappings: {str(e)}")
    
    def get_all_patterns(self) -> Dict[str, Dict[str, str]]:
        """
        Get all cached mapping patterns
        
        Returns:
            Dict[str, Dict[str, str]]: All cached patterns
        """
        patterns = {}
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                key = cache_file.stem
                
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                patterns[key] = data.get('mappings', {})
                
            except (OSError, json.JSONDecodeError) as e:
                self.logger.error(f"Error loading pattern file {cache_file}: {str(e)}")
                
        return patterns
    
    def _normalize_key(self, key: str) -> str:
        """
        Normalize a cache key
        
        Args:
            key (str): Cache key
            
        Returns:
            str: Normalized key
        """
        # Convert to lowercase and remove special characters
        key = key.lower()
        key = ''.join(c if c.isalnum() else '_' for c in key)
        
        # Remove consecutive underscores
        while '__' in key:
            key = key.replace('__', '_')
            
        # Remove leading/trailing underscores
        key = key.strip('_')
        
        return key