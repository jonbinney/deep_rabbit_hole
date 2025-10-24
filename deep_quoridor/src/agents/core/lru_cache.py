class LRUCache:
    """An LRU (Least Recently Used) cache implementation.
    It stores key-value pairs and discards the least recently used items when the cache
    exceeds its maximum capacity. Important: __contains__ does not currently considers
    the entry as used."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache = {}

    def get(self, key):
        if key in self.cache:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[key] = value

    def __len__(self):
        return len(self.cache)

    def __contains__(self, key):
        return key in self.cache

    def __iter__(self):
        return iter(self.cache)

    def __getitem__(self, key):
        v = self.get(key)
        if v is None:
            raise KeyError(key)
        return v

    def __setitem__(self, key, value):
        self.put(key, value)

    def clear(self):
        self.cache.clear()
