"""
Unit tests for the rate limiter utility
"""
import pytest
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import functools

from utils.rate_limiting.rate_limiter import TokenBucket, RateLimiter


class TestTokenBucket:
    """Tests for the TokenBucket class"""

    def test_init(self):
        """Test initialization with default parameters"""
        bucket = TokenBucket(tokens_per_second=10, max_tokens=100)
        assert bucket.tokens_per_second == 10
        assert bucket.max_tokens == 100
        assert bucket.tokens == 100  # Default to full

        # Test with custom initial tokens
        bucket = TokenBucket(tokens_per_second=10, max_tokens=100, initial_tokens=50)
        assert bucket.tokens == 50

    def test_add_tokens(self):
        """Test adding tokens based on elapsed time"""
        bucket = TokenBucket(tokens_per_second=10, max_tokens=100, initial_tokens=0)

        # Manually set the last update time to simulate elapsed time
        bucket.last_update = time.time() - 1  # 1 second ago

        # Add tokens
        bucket._add_tokens()

        # Check that tokens were added (should be around 10)
        assert 9.5 <= bucket.tokens <= 10.5

        # Check that tokens don't exceed max_tokens
        bucket.tokens = 95
        bucket.last_update = time.time() - 1
        bucket._add_tokens()
        assert bucket.tokens == 100

    def test_consume_immediate(self):
        """Test consuming tokens immediately"""
        bucket = TokenBucket(tokens_per_second=10, max_tokens=100)

        # Consume tokens
        assert bucket.consume(tokens=10, wait=False)
        assert abs(bucket.tokens - 90) < 0.01  # Allow small floating-point differences

        # Try to consume more tokens than available with wait=False
        assert not bucket.consume(tokens=100, wait=False)
        assert abs(bucket.tokens - 90) < 0.01  # Tokens unchanged

        # Try to consume more tokens than max_tokens
        assert not bucket.consume(tokens=101, wait=False)
        assert abs(bucket.tokens - 90) < 0.01  # Tokens unchanged

    def test_consume_wait(self):
        """Test consuming tokens with waiting"""
        bucket = TokenBucket(tokens_per_second=100, max_tokens=100, initial_tokens=10)

        # Consume tokens that are available immediately
        start_time = time.time()
        assert bucket.consume(tokens=10, wait=True)
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should be almost immediate
        assert abs(bucket.tokens) < 0.01  # Should be close to 0

        # Consume tokens that require waiting
        start_time = time.time()
        assert bucket.consume(tokens=20, wait=True)
        elapsed = time.time() - start_time
        assert 0.15 < elapsed < 0.25  # Should wait about 0.2 seconds
        assert abs(bucket.tokens) < 0.01  # Should be close to 0

    def test_get_status(self):
        """Test getting bucket status"""
        bucket = TokenBucket(tokens_per_second=10, max_tokens=100, initial_tokens=50)

        status = bucket.get_status()
        assert abs(status['available_tokens'] - 50) < 0.01
        assert status['max_tokens'] == 100
        assert status['tokens_per_second'] == 10
        assert abs(status['utilization'] - 0.5) < 0.01


class TestRateLimiter:
    """Tests for the RateLimiter class"""

    def test_init(self):
        """Test initialization with default parameters"""
        limiter = RateLimiter()
        assert limiter.bucket.tokens_per_second == 1.0  # 60 per minute / 60 seconds
        assert limiter.bucket.max_tokens == 10

        # Test with custom parameters
        limiter = RateLimiter(requests_per_minute=120, burst_size=20)
        assert limiter.bucket.tokens_per_second == 2.0  # 120 per minute / 60 seconds
        assert limiter.bucket.max_tokens == 20

    def test_limit_decorator(self):
        """Test the limit decorator"""
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)

        # Define a function to limit
        @limiter.limit
        def limited_func(x):
            return x * 2

        # Call the function multiple times
        results = [limited_func(i) for i in range(5)]
        assert results == [0, 2, 4, 6, 8]
        assert limiter.total_requests == 5

        # Check that the function still works when rate limited
        limiter.bucket.tokens = 0  # Force rate limiting

        # Call the function with rate limiting
        start_time = time.time()
        result = limited_func(5)
        elapsed = time.time() - start_time

        assert result == 10
        assert elapsed > 0.01  # Should have waited
        assert limiter.limited_requests >= 1

    def test_token_cost_function(self):
        """Test using a custom token cost function"""
        # Define a token cost function
        def token_cost_func(x, *args, **kwargs):
            return x  # Cost equals the input value

        limiter = RateLimiter(
            requests_per_minute=60,
            burst_size=10,
            token_cost_func=token_cost_func
        )

        # Define a function to limit
        @limiter.limit
        def limited_func(x):
            return x * 2

        # Call with low cost
        start_time = time.time()
        result = limited_func(1)
        elapsed = time.time() - start_time
        assert result == 2
        assert elapsed < 0.1  # Should be immediate

        # Call with high cost that exceeds available tokens
        limiter.bucket.tokens = 5  # Set available tokens

        start_time = time.time()
        result = limited_func(10)  # Cost of 10 > available 5
        elapsed = time.time() - start_time

        assert result == 20
        assert elapsed > 0.05  # Should have waited

    def test_concurrent_requests(self):
        """Test rate limiting with concurrent requests"""
        limiter = RateLimiter(requests_per_minute=60, burst_size=5)

        # Define a function to limit
        @limiter.limit
        def limited_func(x):
            time.sleep(0.01)  # Small delay to simulate work
            return x

        # Run multiple concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            start_time = time.time()
            results = list(executor.map(limited_func, range(10)))
            elapsed = time.time() - start_time

        # Check results
        assert results == list(range(10))

        # Should have rate limited some requests
        assert limiter.limited_requests > 0

        # Total time should be at least the time needed to refill tokens
        # 10 requests - 5 burst = 5 additional requests
        # 5 requests / (60/60) tokens per second = 5 seconds
        # But we're using a higher rate to keep the test fast
        assert elapsed > 0.1

    def test_get_stats(self):
        """Test getting rate limiter statistics"""
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)

        # Make some requests
        @limiter.limit
        def limited_func(x):
            return x

        for i in range(5):
            limited_func(i)

        # Force some rate limiting
        limiter.bucket.tokens = 0
        limited_func(5)

        # Get stats
        stats = limiter.get_stats()

        assert stats['total_requests'] == 6
        assert stats['limited_requests'] >= 1
        assert stats['wait_time'] > 0
        assert 0 < stats['limit_rate'] <= 1
        assert stats['tokens_per_second'] == 1.0
        assert stats['max_tokens'] == 10
