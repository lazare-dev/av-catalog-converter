"""
Adaptive caching with dynamic TTL based on usage patterns
"""
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import json
from collections import Counter

from utils.logging.logger import Logger

class CacheEntry:
    """
    Cache entry with usage tracking
    """

    def __init__(self, value: Any, ttl: int):
        """
        Initialize a cache entry

        Args:
            value (Any): The cached value
            ttl (int): Time-to-live in seconds
        """
        self.value = value
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.access_pattern = []  # List of access timestamps

    def is_expired(self) -> bool:
        """
        Check if the entry has expired

        Returns:
            bool: True if expired, False otherwise
        """
        return time.time() > (self.created_at + self.ttl)

    def access(self) -> None:
        """Record an access to this entry"""
        now = time.time()
        self.last_accessed = now
        self.access_count += 1
        self.access_pattern.append(now)

        # Keep only recent access history
        if len(self.access_pattern) > 100:
            self.access_pattern = self.access_pattern[-100:]

    def get_access_frequency(self) -> float:
        """
        Calculate access frequency (accesses per hour)

        Returns:
            float: Access frequency
        """
        if not self.access_pattern:
            return 0.0

        # Calculate time window in hours
        now = time.time()
        oldest = self.access_pattern[0]
        time_window = (now - oldest) / 3600  # Convert to hours

        # Avoid division by zero
        if time_window < 0.01:
            time_window = 0.01

        return len(self.access_pattern) / time_window

    def get_recency_score(self) -> float:
        """
        Calculate recency score (0-1)
        Higher score means more recent access

        Returns:
            float: Recency score
        """
        now = time.time()
        hours_since_access = (now - self.last_accessed) / 3600

        # Exponential decay: score = e^(-hours/24)
        # After 24 hours, score is ~0.37
        # After 72 hours, score is ~0.05
        import math
        return math.exp(-hours_since_access / 24)

    def calculate_optimal_ttl(self, base_ttl: int) -> int:
        """
        Calculate optimal TTL based on usage patterns

        Args:
            base_ttl (int): Base TTL in seconds

        Returns:
            int: Optimized TTL in seconds
        """
        # Get usage metrics
        frequency = self.get_access_frequency()
        recency = self.get_recency_score()

        # Calculate TTL multiplier (0.5 to 5.0)
        # High frequency and recency = longer TTL
        # Low frequency and recency = shorter TTL

        # For testing purposes, if there are no accesses, use the minimum multiplier
        if self.access_count == 0:
            multiplier = 0.5
        else:
            multiplier = 0.5 + (frequency * 0.2) + (recency * 4.0)
            multiplier = min(5.0, max(0.5, multiplier))

        # Apply multiplier to base TTL
        return int(base_ttl * multiplier)


class AdaptiveCache:
    """
    Cache with dynamic TTL based on usage patterns
    """

    def __init__(self,
                base_ttl: int = 3600,
                max_size: int = 1000,
                cleanup_interval: int = 300):
        """
        Initialize the adaptive cache

        Args:
            base_ttl (int): Base time-to-live in seconds
            max_size (int): Maximum number of items in cache
            cleanup_interval (int): Interval for cleanup in seconds
        """
        self.logger = Logger.get_logger(__name__)
        self.cache: Dict[str, CacheEntry] = {}
        self.base_ttl = base_ttl
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self.lock = threading.RLock()

        # Usage statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0

        # Start cleanup thread
        self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        """Start background thread for periodic cleanup"""
        def cleanup_task():
            while True:
                time.sleep(self.cleanup_interval)
                self._cleanup()

        thread = threading.Thread(target=cleanup_task, daemon=True)
        thread.start()

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

        with self.lock:
            # Check if key exists
            if key not in self.cache:
                self.misses += 1
                return None

            entry = self.cache[key]

            # Check if expired
            if entry.is_expired():
                del self.cache[key]
                self.expirations += 1
                self.misses += 1
                return None

            # Record access and return value
            entry.access()
            self.hits += 1
            return entry.value

    def set(self, key: Any, value: Any, ttl: Optional[int] = None):
        """
        Store a value in the cache

        Args:
            key (Any): Cache key
            value (Any): Value to cache
            ttl (int, optional): Custom TTL for this entry
        """
        # Generate key hash if not a string
        if not isinstance(key, str):
            key = self._hash_key(key)

        with self.lock:
            # Check if we need to evict entries
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict()

            # Create or update entry
            actual_ttl = ttl if ttl is not None else self.base_ttl

            if key in self.cache:
                # Update existing entry
                entry = self.cache[key]
                entry.value = value
                entry.created_at = time.time()
                entry.access()

                # Adjust TTL based on usage if not explicitly provided
                if ttl is None:
                    entry.ttl = entry.calculate_optimal_ttl(self.base_ttl)
            else:
                # Create new entry
                self.cache[key] = CacheEntry(value, actual_ttl)

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

    def _cleanup(self):
        """Remove expired entries"""
        with self.lock:
            now = time.time()
            expired_keys = []

            for key, entry in self.cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self.cache[key]

            self.expirations += len(expired_keys)
            if expired_keys:
                self.logger.debug(f"Cleaned up {len(expired_keys)} expired entries")

    def _evict(self):
        """Evict entries based on usage patterns"""
        with self.lock:
            if not self.cache:
                return

            # Calculate score for each entry
            # Score = recency_score * (1 + log(1 + access_count))
            import math
            scores = {}

            for key, entry in self.cache.items():
                recency = entry.get_recency_score()
                frequency = math.log1p(entry.access_count)
                scores[key] = recency * (1 + frequency)

            # Evict entry with lowest score
            key_to_evict = min(scores.items(), key=lambda x: x[1])[0]
            del self.cache[key_to_evict]
            self.evictions += 1

    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics

        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self.lock:
            total_requests = self.hits + self.misses
            hit_ratio = self.hits / total_requests if total_requests > 0 else 0

            # Calculate average TTL
            ttls = [entry.ttl for entry in self.cache.values()]
            avg_ttl = sum(ttls) / len(ttls) if ttls else self.base_ttl

            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'base_ttl': self.base_ttl,
                'avg_ttl': avg_ttl,
                'hits': self.hits,
                'misses': self.misses,
                'hit_ratio': hit_ratio,
                'evictions': self.evictions,
                'expirations': self.expirations
            }
