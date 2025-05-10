# utils/caching/disk_cache.py
"""
Persistent disk caching
"""
import logging
import time
import os
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import hashlib

class DiskCache:
    """Persistent disk-based caching utility"""
    
    def __init__(self, cache_dir: str, ttl: int = 86400):
        """
        Initialize the disk cache
        
        Args:
            cache_dir (str): Directory for cache files
            ttl (int): Cache time-to-live in seconds
        """
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get a value from the cache
        
        Args:
            key (Any): Cache key
            
        Returns:
            Optional[Any]: Cached value or None if not found/expired
        """
        # Generate key hash
        key_hash = self._hash_key(key)
        cache_path = self.cache_dir / f"{key_hash}.cache"
        
        # Check if cache file exists
        if not cache_path.exists():
            return None
            
        try:
            # Check file age
            mtime = cache_path.stat().st_mtime
            if time.time() - mtime > self.ttl:
                # Remove expired file
                os.unlink(cache_path)
                return None
                
            # Load cached value
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
                
        except (OSError, pickle.PickleError) as e:
            self.logger.error(f"Error reading cache file: {str(e)}")
            return None
    
    def set(self, key: Any, value: Any):
        """
        Store a value in the cache
        
        Args:
            key (Any): Cache key
            value (Any): Value to cache
        """
        # Generate key hash
        key_hash = self._hash_key(key)
        cache_path = self.cache_dir / f"{key_hash}.cache"
        
        try:
            # Store value in cache file
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
                
        except (OSError, pickle.PickleError) as e:
            self.logger.error(f"Error writing cache file: {str(e)}")
    
    def _hash_key(self, key: Any) -> str:
        """
        Hash a complex key into a string
        
        Args:
            key (Any): Key to hash
            
        Returns:
            str: Hashed key
        """
        try:
            # Convert to JSON and hash
            key_json = json.dumps(key, sort_keys=True)
            return hashlib.md5(key_json.encode()).hexdigest()
        except (TypeError, ValueError):
            # Fallback for unhashable types
            return hashlib.md5(str(key).encode()).hexdigest()
    
    def clear(self):
        """Clear all cache entries"""
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                os.unlink(cache_file)
            except OSError as e:
                self.logger.error(f"Error removing cache file {cache_file}: {str(e)}")
    
    def prune(self):
        """Remove expired cache entries"""
        current_time = time.time()
        count = 0
        
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                # Check file age
                mtime = cache_file.stat().st_mtime
                if current_time - mtime > self.ttl:
                    os.unlink(cache_file)
                    count += 1
            except OSError as e:
                self.logger.error(f"Error pruning cache file {cache_file}: {str(e)}")
                
        self.logger.debug(f"Pruned {count} expired cache entries")