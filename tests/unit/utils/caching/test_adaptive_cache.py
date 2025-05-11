"""
Unit tests for the adaptive cache utility
"""
import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor

from utils.caching.adaptive_cache import AdaptiveCache, CacheEntry


class TestCacheEntry:
    """Tests for the CacheEntry class"""

    def test_init(self):
        """Test initialization"""
        entry = CacheEntry("test_value", ttl=3600)
        assert entry.value == "test_value"
        assert entry.ttl == 3600
        assert entry.access_count == 0
        assert len(entry.access_pattern) == 0

        # Check timestamps
        assert entry.created_at <= time.time()
        assert entry.last_accessed <= time.time()

    def test_is_expired(self):
        """Test expiration check"""
        # Create an entry with a short TTL
        entry = CacheEntry("test_value", ttl=0.1)

        # Should not be expired immediately
        assert not entry.is_expired()

        # Wait for expiration
        time.sleep(0.2)
        assert entry.is_expired()

        # Create an entry with a longer TTL
        entry = CacheEntry("test_value", ttl=3600)
        assert not entry.is_expired()

    def test_access(self):
        """Test recording access"""
        entry = CacheEntry("test_value", ttl=3600)

        # Record access
        entry.access()

        assert entry.access_count == 1
        assert len(entry.access_pattern) == 1
        assert entry.last_accessed <= time.time()

        # Record multiple accesses
        for _ in range(5):
            entry.access()

        assert entry.access_count == 6
        assert len(entry.access_pattern) == 6

    def test_access_pattern_limit(self):
        """Test that access pattern is limited to 100 entries"""
        entry = CacheEntry("test_value", ttl=3600)

        # Add more than 100 accesses
        for _ in range(110):
            entry.access()

        # Should be limited to 100
        assert len(entry.access_pattern) == 100
        assert entry.access_count == 110

    def test_get_access_frequency(self):
        """Test calculating access frequency"""
        entry = CacheEntry("test_value", ttl=3600)

        # No accesses
        assert entry.get_access_frequency() == 0.0

        # Add some accesses
        for _ in range(10):
            entry.access()
            time.sleep(0.01)  # Small delay to spread out accesses

        # Calculate frequency (accesses per hour)
        frequency = entry.get_access_frequency()

        # Should be a high frequency since accesses were recent and close together
        assert frequency > 0

    def test_get_recency_score(self):
        """Test calculating recency score"""
        entry = CacheEntry("test_value", ttl=3600)

        # Access the entry
        entry.access()

        # Recency score should be high for recent access
        assert 0.9 < entry.get_recency_score() <= 1.0

        # Simulate older access
        entry.last_accessed = time.time() - (24 * 3600)  # 24 hours ago

        # Recency score should be lower (e^-1 ≈ 0.368)
        assert entry.get_recency_score() < 0.4

    def test_calculate_optimal_ttl(self):
        """Test calculating optimal TTL"""
        entry = CacheEntry("test_value", ttl=3600)

        # No accesses, should use base multiplier
        # Force recency score to be low for consistent testing
        entry.last_accessed = time.time() - (24 * 3600)  # 24 hours ago
        optimal_ttl = entry.calculate_optimal_ttl(3600)
        assert 1800 <= optimal_ttl <= 2000  # Around 0.5 * base_ttl

        # Add some accesses
        for _ in range(10):
            entry.access()

        # Should have a higher multiplier due to recent accesses
        optimal_ttl = entry.calculate_optimal_ttl(3600)
        assert optimal_ttl > 3600  # Greater than base_ttl


class TestAdaptiveCache:
    """Tests for the AdaptiveCache class"""

    def test_init(self):
        """Test initialization"""
        cache = AdaptiveCache()
        assert cache.base_ttl == 3600
        assert cache.max_size == 1000
        assert cache.cleanup_interval == 300
        assert len(cache.cache) == 0

        # Test with custom parameters
        cache = AdaptiveCache(base_ttl=7200, max_size=500, cleanup_interval=600)
        assert cache.base_ttl == 7200
        assert cache.max_size == 500
        assert cache.cleanup_interval == 600

    def test_get_set_basic(self):
        """Test basic get and set operations"""
        cache = AdaptiveCache()

        # Set a value
        cache.set("key1", "value1")

        # Get the value
        assert cache.get("key1") == "value1"

        # Get a non-existent key
        assert cache.get("key2") is None

        # Check stats
        assert cache.hits == 1
        assert cache.misses == 1

    def test_expiration(self):
        """Test that entries expire"""
        cache = AdaptiveCache(base_ttl=0.1)

        # Set a value
        cache.set("key1", "value1")

        # Get the value immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(0.2)

        # Value should be expired
        assert cache.get("key1") is None
        assert cache.expirations == 1

    def test_complex_keys(self):
        """Test using complex objects as keys"""
        cache = AdaptiveCache()

        # Use a dict as key
        key = {"id": 123, "name": "test"}
        cache.set(key, "value1")

        # Get with the same dict structure
        assert cache.get({"id": 123, "name": "test"}) == "value1"

        # Use a list as key
        key = [1, 2, 3]
        cache.set(key, "value2")

        # Get with the same list
        assert cache.get([1, 2, 3]) == "value2"

    def test_eviction(self):
        """Test that entries are evicted when cache is full"""
        cache = AdaptiveCache(max_size=2)

        # Set values
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Cache should have 2 entries
        assert len(cache.cache) == 2

        # Access key1 multiple times to increase its score
        for _ in range(5):
            assert cache.get("key1") == "value1"

        # Add a third entry, should evict one (likely key2)
        cache.set("key3", "value3")

        # Cache should still have 2 entries
        assert len(cache.cache) == 2

        # key1 should still be there due to high access count
        assert cache.get("key1") == "value1"

        # Either key2 or key3 should be evicted
        assert cache.evictions == 1

    def test_adaptive_ttl(self):
        """Test that TTL adapts based on usage patterns"""
        cache = AdaptiveCache(base_ttl=3600)

        # Set a value
        cache.set("key1", "value1")

        # Get the initial TTL
        initial_ttl = cache.cache["key1"].ttl

        # Access the value multiple times
        for _ in range(10):
            assert cache.get("key1") == "value1"
            time.sleep(0.01)  # Small delay to spread out accesses

        # Update the value
        cache.set("key1", "value1-updated")

        # Get the new TTL
        new_ttl = cache.cache["key1"].ttl

        # TTL should have increased due to frequent access
        assert new_ttl > initial_ttl

    def test_clear(self):
        """Test clearing the cache"""
        cache = AdaptiveCache()

        # Set some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Clear the cache
        cache.clear()

        # Cache should be empty
        assert len(cache.cache) == 0
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_get_stats(self):
        """Test getting cache statistics"""
        cache = AdaptiveCache()

        # Set some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Access values
        assert cache.get("key1") == "value1"
        assert cache.get("key2") == "value2"
        assert cache.get("key3") is None

        # Get stats
        stats = cache.get_stats()

        assert stats['size'] == 2
        assert stats['max_size'] == 1000
        assert stats['base_ttl'] == 3600
        assert stats['hits'] == 2
        assert stats['misses'] == 1
        assert stats['hit_ratio'] == 2/3
        assert stats['evictions'] == 0
        assert stats['expirations'] == 0

    def test_concurrent_access(self):
        """Test concurrent access to the cache"""
        cache = AdaptiveCache()

        # Function to access the cache
        def access_cache(key):
            # Try to get the value
            value = cache.get(key)
            if value is None:
                # Set the value if not found
                cache.set(key, f"value-{key}")
                return False
            return True

        # Run concurrent access
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Each key will be accessed by multiple threads
            keys = ["key1", "key2", "key3"] * 10
            results = list(executor.map(access_cache, keys))

        # Check that the cache has the expected entries
        assert cache.get("key1") == "value-key1"
        assert cache.get("key2") == "value-key2"
        assert cache.get("key3") == "value-key3"

        # Some accesses should have been hits
        assert cache.hits > 0
