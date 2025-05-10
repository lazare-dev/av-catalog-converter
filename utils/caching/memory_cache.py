
# utils/caching/memory_cache.py
"""
In-memory caching
"""
import logging
import time
from typing import Dict, Any, Optional, Tuple
import hashlib
import json

class MemoryCache:
    """Simple in-memory caching utility"""
    
    def __init__(self, ttl: int = 3600):
        """
        Initialize the memory cache
        
        Args:
            ttl (int): Cache time-to-live in seconds
        """
        self.logger = logging.getLogger(__name__)
        self.cache = {}  # key -> (value, timestamp)
        self.ttl = ttl
    
    def get(self, key: Any) -> Optional[Any]:
        """
        Get a value from the cache
        
        Args:
            key (Any): Cache key
            
        Returns:
            Optional[Any]: Cached value or None if not found/expired
        """
        # Generate key hash if not a string
        if not isinstance(key, str):
            key = self._hash_key(key)
            
        # Check if key exists
        if key not in self.cache:
            return None
            
        # Get value and timestamp
        value, timestamp = self.cache[key]
        
        # Check if expired
        if time.time() - timestamp > self.ttl:
            # Remove expired entry
            del self.cache[key]
            return None
            
        return value
    
    def set(self, key: Any, value: Any):
        """
        Store a value in the cache
        
        Args:
            key (Any): Cache key
            value (Any): Value to cache
        """
        # Generate key hash if not a string
        if not isinstance(key, str):
            key = self._hash_key(key)
            
        # Store value with current timestamp
        self.cache[key] = (value, time.time())
        
        # Prune expired entries if cache is getting large
        if len(self.cache) > 1000:
            self._prune()
    
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
    
    def _prune(self):
        """Remove expired entries from the cache"""
        current_time = time.time()
        expired_keys = [
            k for k, (_, timestamp) in self.cache.items() 
            if current_time - timestamp > self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
            
        self.logger.debug(f"Pruned {len(expired_keys)} expired cache entries")
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()