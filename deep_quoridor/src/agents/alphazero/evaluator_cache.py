"""
LRU Cache implementation for evaluator caching.
"""

import sys
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np


class EvaluatorCache:
    """
    LRU (Least Recently Used) cache for storing numpy array evaluations.

    The cache maps from numpy arrays (converted to bytes as keys) to tuples of two numpy arrays
    representing the evaluation results. When the cache exceeds max_size, the least recently
    used items are removed.
    """

    def __init__(self, max_size: int = 1e5):
        """
        Initialize the LRU cache.

        Args:
            max_size: Maximum number of items to store in the cache
        """
        self.max_size = max_size
        self._cache = OrderedDict()

    def _to_key(self, array: np.ndarray) -> bytes:
        """Convert numpy array or bytes to bytes, which are hashable and can be used as a dictioanry key."""
        return array.tobytes()

    def get(self, key: np.ndarray, default=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Get an item from the cache. Moves the item to the end (most recently used).

        Args:
            key: The cache key (numpy array or bytes)
            default: Default value to return if key is not found

        Returns:
            The cached value or default if not found
        """
        key = self._to_key(key)
        if key in self._cache:
            # Move to end (mark as recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        return default

    def __setitem__(self, key: np.ndarray, value: Tuple[np.ndarray, np.ndarray]):
        """
        Set an item in the cache. Handles LRU eviction if cache is full.

        Args:
            key: The cache key (numpy array or bytes)
            value: Tuple of two numpy arrays (value, policy)
        """
        key = self._to_key(key)
        if key in self._cache:
            # Update existing item and move to end
            self._cache.pop(key)
        elif len(self._cache) >= self.max_size:
            # Remove least recently used item
            self._cache.popitem(last=False)

        self._cache[key] = value

    def __getitem__(self, key: np.ndarray):
        """
        Get an item from the cache with dictionary-style access.

        Args:
            key: The cache key

        Returns:
            The cached value

        Raises:
            KeyError: If key is not in cache
        """
        key = self._to_key(key)
        if key in self._cache:
            # Move to end (mark as recently used)
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        raise KeyError(key)

    def __contains__(self, key: Union[np.ndarray, bytes]) -> bool:
        """
        Check if key is in cache.

        Args:
            key: The cache key

        Returns:
            True if key is in cache, False otherwise
        """
        key = self._to_key(key)
        return key in self._cache

    def clear(self):
        """Clear all items from the cache."""
        self._cache.clear()

    def get_memory_usage(self) -> int:
        """
        Calculate the total memory usage of the cache including all keys and values.

        Returns:
            Total memory usage in bytes
        """
        total_memory = 0

        # Memory for the OrderedDict structure itself
        total_memory += sys.getsizeof(self._cache)

        for key, value in self._cache.items():
            # Memory for the key (bytes)
            total_memory += sys.getsizeof(key)

            # Memory for the tuple wrapper
            total_memory += sys.getsizeof(value)

            # Memory for the numpy arrays in the tuple
            for array in value:
                total_memory += array.nbytes
                total_memory += sys.getsizeof(array)  # Array object overhead

        return total_memory

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        return len(self._cache)
