"""
Rate limiting utilities for API calls
"""
import time
import logging
import threading
from typing import Dict, Any, Optional, Callable, List, Tuple
from collections import deque
import functools

from utils.logging.logger import Logger

class TokenBucket:
    """
    Token bucket rate limiter implementation

    This implementation uses the token bucket algorithm:
    - The bucket has a maximum capacity of tokens
    - Tokens are added to the bucket at a fixed rate
    - Each operation consumes one or more tokens
    - If the bucket doesn't have enough tokens, the operation is blocked
    """

    def __init__(self,
                tokens_per_second: float,
                max_tokens: int,
                initial_tokens: Optional[int] = None):
        """
        Initialize the token bucket

        Args:
            tokens_per_second (float): Rate at which tokens are added to the bucket
            max_tokens (int): Maximum number of tokens the bucket can hold
            initial_tokens (int, optional): Initial number of tokens in the bucket
                If None, the bucket starts full
        """
        self.tokens_per_second = tokens_per_second
        self.max_tokens = max_tokens
        self.tokens = initial_tokens if initial_tokens is not None else max_tokens
        self.last_update = time.time()
        self.lock = threading.RLock()
        self.logger = Logger.get_logger(__name__)

    def _add_tokens(self):
        """Add tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_update
        new_tokens = elapsed * self.tokens_per_second

        self.tokens = min(self.max_tokens, self.tokens + new_tokens)
        self.last_update = now

    def consume(self, tokens: int = 1, wait: bool = True, force: bool = False) -> bool:
        """
        Consume tokens from the bucket

        Args:
            tokens (int): Number of tokens to consume
            wait (bool): Whether to wait for tokens to become available
            force (bool): Whether to force consumption even if tokens exceed capacity

        Returns:
            bool: True if tokens were consumed, False otherwise
        """
        # Handle case where requested tokens exceed bucket capacity
        if tokens > self.max_tokens and not force:
            self.logger.warning(f"Requested tokens ({tokens}) exceeds bucket capacity ({self.max_tokens})")
            if not wait:
                return False  # Fail immediately if not waiting
            # Cap the tokens to the maximum capacity if waiting
            tokens = self.max_tokens
            self.logger.info(f"Capping token request to bucket capacity: {self.max_tokens}")

        with self.lock:
            self._add_tokens()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            if not wait:
                self.logger.debug(f"Rate limited: not enough tokens available ({self.tokens} < {tokens})")
                return False

            # Calculate how long to wait for enough tokens
            deficit = tokens - self.tokens
            wait_time = deficit / self.tokens_per_second

            # Cap wait time to a reasonable maximum (30 seconds)
            if wait_time > 30:
                self.logger.warning(f"Excessive wait time ({wait_time:.2f}s) capped to 30s")
                wait_time = 30

            self.logger.debug(f"Rate limited: waiting {wait_time:.2f}s for {deficit:.2f} tokens")

            # Sleep and then consume
            time.sleep(wait_time)
            self.tokens = 0  # We've consumed all available tokens plus waited for the deficit
            self.last_update = time.time()
            return True

    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the token bucket

        Returns:
            Dict[str, Any]: Status information
        """
        with self.lock:
            self._add_tokens()
            return {
                'available_tokens': self.tokens,
                'max_tokens': self.max_tokens,
                'tokens_per_second': self.tokens_per_second,
                'utilization': 1.0 - (self.tokens / self.max_tokens)
            }


class RateLimiter:
    """
    Rate limiter for API calls

    Provides decorators and context managers for rate limiting
    """

    def __init__(self,
                requests_per_minute: float = 60.0,
                burst_size: int = 10,
                token_cost_func: Optional[Callable] = None):
        """
        Initialize the rate limiter

        Args:
            requests_per_minute (float): Maximum requests per minute
            burst_size (int): Maximum burst size (number of requests that can be made at once)
            token_cost_func (Callable, optional): Function to calculate token cost
                If None, each request costs 1 token
        """
        # Convert requests per minute to tokens per second
        tokens_per_second = requests_per_minute / 60.0

        self.bucket = TokenBucket(tokens_per_second, burst_size)
        self.token_cost_func = token_cost_func or (lambda *args, **kwargs: 1)
        self.logger = Logger.get_logger(__name__)

        # Stats
        self.total_requests = 0
        self.limited_requests = 0
        self.wait_time = 0.0

    def limit(self, func):
        """
        Decorator for rate limiting a function

        Args:
            func (Callable): Function to rate limit

        Returns:
            Callable: Rate-limited function
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Calculate token cost
            token_cost = self.token_cost_func(*args, **kwargs)

            # Track stats
            self.total_requests += 1

            # Try to consume tokens
            start_time = time.time()
            if not self.bucket.consume(token_cost, wait=True):
                self.logger.warning(f"Rate limit exceeded for {func.__name__}")
                self.limited_requests += 1
                raise Exception(f"Rate limit exceeded for {func.__name__}")

            # Track wait time
            wait_time = time.time() - start_time
            if wait_time > 0.01:  # Only count significant waits
                self.wait_time += wait_time
                self.limited_requests += 1

            # Call the function
            return func(*args, **kwargs)

        return wrapper

    def check_limit(self, token_cost: int = 1, wait: bool = False) -> bool:
        """
        Check if a request would be rate limited

        Args:
            token_cost (int): Number of tokens this request would cost
            wait (bool): Whether to wait for tokens to become available

        Returns:
            bool: True if request can proceed, False if it would be rate limited
        """
        # Track stats
        self.total_requests += 1

        # Try to consume tokens
        start_time = time.time()
        result = self.bucket.consume(token_cost, wait=wait)

        # If rate limited, increment counter
        if not result:
            self.limited_requests += 1
            self.logger.debug(f"Rate limited: request with cost {token_cost}")
            return False

        # Track wait time
        wait_time = time.time() - start_time
        if wait_time > 0.01:  # Only count significant waits
            self.wait_time += wait_time
            # Count as limited if we had to wait
            self.limited_requests += 1

        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiting statistics

        Returns:
            Dict[str, Any]: Statistics
        """
        bucket_status = self.bucket.get_status()

        return {
            'total_requests': self.total_requests,
            'limited_requests': self.limited_requests,
            'wait_time': self.wait_time,
            'limit_rate': self.limited_requests / max(1, self.total_requests),
            'tokens_per_second': bucket_status['tokens_per_second'],
            'available_tokens': bucket_status['available_tokens'],
            'max_tokens': bucket_status['max_tokens'],
            'utilization': bucket_status['utilization']
        }
