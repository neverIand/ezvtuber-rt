import numpy as np
from collections import OrderedDict
from typing import Hashable, Optional

"""
Lossless image cache with raw/Brotli storage and LRU eviction.
"""


def _brotli_module():
    # Raw mode is the EasyVtuber fast path and should not require importing the
    # optional legacy compression dependency at runtime.
    import brotli

    return brotli


def array_cache_key(array: np.ndarray) -> Hashable:
    """Create an exact, collision-safe in-memory key for a NumPy array.

    Returning the original metadata and bytes lets Python's dictionary resolve
    hash collisions by equality. This also avoids NumPy's comparatively costly
    and formatting-dependent string conversion.
    """
    contiguous = np.ascontiguousarray(array)
    return contiguous.dtype.str, contiguous.shape, contiguous.tobytes()


class Cacher:
    """Lossless LRU image cache.

    ``brotli`` preserves the legacy high-capacity behavior.  ``raw`` stores an
    owned, read-only contiguous array and removes compression/decompression
    from the frame hot path.  Both modes enforce the configured byte budget.
    
    Attributes:
        cache: OrderedDict storing compressed bytes or raw arrays
        max_kbytes: Maximum cache size in kilobytes
        cached_kbytes: Current cache size in kilobytes
        hits: Total cache hits
        miss: Total cache misses
    """
    
    def __init__(
            self,
            max_volume_giga: float = 2.0,
            width: int = 512,
            height: int = 512,
            storage_mode: str = "brotli",
    ):
        """Initialize cache with specified size.
        
        Args:
            max_volume_giga: Maximum cache size in gigabytes (default 2.0)
            storage_mode: ``brotli`` for compressed entries or ``raw`` for the
                low-latency, lossless in-memory fast path.
        """
        if storage_mode not in ("brotli", "raw"):
            raise ValueError(
                "storage_mode must be either 'brotli' or 'raw', got {!r}".format(
                    storage_mode
                )
            )
        self.cache = OrderedDict()  # LRU cache storage
        self.width = width
        self.height = height
        self.storage_mode = storage_mode
        
        # Cache size management
        self.max_kbytes = max_volume_giga * 1024 * 1024  # Convert GB to KB
        self.cached_kbytes = 0  # Tracks total cached data size

        # Performance tracking
        self.hits = 0  # Total successful cache retrievals
        self.miss = 0  # Total cache misses

    def query(self, hs: Hashable) -> bool:
        """Check if a hash key exists in the cache.
        
        Args:
            hs: Hash key to query
        Returns:
            bool: True if key exists, False otherwise
        """
        is_in = hs in self.cache
        if is_in:
            self.cache.move_to_end(hs)  # Update LRU position
        return is_in
    
    def get(self, hs: Hashable) -> Optional[np.ndarray]:
        """Retrieve cached data by hash key without anti-thrashing.
        
        Args:
            hs: Hash key of requested data
        Returns:
            np.ndarray: Decompressed image data or None if miss
        """
        cached = self.cache.get(hs)
        if cached is not None:
            self.hits += 1
            self.cache.move_to_end(hs)
            if self.storage_mode == "raw":
                return cached
            return np.frombuffer(
                _brotli_module().decompress(cached),
                dtype=np.uint8,
            ).reshape((self.height, self.width, 4))
        else:
            self.miss += 1
            return None

    def put(self, hs: Hashable, data:np.ndarray):
        """Write data to cache with compression.
        
        Args:
            hs: Hash key for the data
            data: Raw image data to cache
        """
        # Skip if already cached
        if hs in self.cache:
            return
            
        if self.storage_mode == "raw":
            payload_kbytes = data.nbytes / 1024
            payload = None
        else:
            payload = _brotli_module().compress(
                np.ascontiguousarray(data).tobytes(),
                quality=0,
            )
            payload_kbytes = len(payload) / 1024

        # An entry larger than the complete budget can never be retained.  Do
        # not evict useful entries or transiently exceed the configured size.
        if payload_kbytes > self.max_kbytes:
            return

        # Evict before insertion so the cache remains within its budget even
        # at the instant a new raw frame is added.
        while self.cache and self.cached_kbytes + payload_kbytes > self.max_kbytes:
            _, evicted = self.cache.popitem(last=False)
            self.cached_kbytes -= self._payload_kbytes(evicted)

        if self.storage_mode == "raw":
            # Copy only after making room so retained cache memory never
            # exceeds the configured budget, even transiently.
            payload = np.array(data, copy=True, order="C")
            # Compressed entries were already effectively read-only because
            # ``np.frombuffer(bytes)`` is read-only.  Match that contract and
            # protect cached frames from debug overlays or external mutation.
            payload.flags.writeable = False

        self.cache[hs] = payload
        self.cached_kbytes += payload_kbytes

    @staticmethod
    def _payload_kbytes(payload) -> float:
        if isinstance(payload, np.ndarray):
            return payload.nbytes / 1024
        return len(payload) / 1024
