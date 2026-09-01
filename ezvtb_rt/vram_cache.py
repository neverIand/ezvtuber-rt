import pycuda.driver as cuda
from collections import OrderedDict
from typing import Dict, Hashable, List, Optional
from ezvtb_rt.trt_engine import HostDeviceMem

#memory management
class VRAMMem(object):
    def __init__(self, nbytes:int):
        self.nbytes: int = nbytes
        self.device: cuda.DeviceAllocation = cuda.mem_alloc(nbytes)
        self._freed = False

    def __str__(self):
        return "Size:\n" + str(self.nbytes) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def free(self) -> None:
        if self._freed:
            return
        self.device.free()
        self._freed = True

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class VRAMCacher:
    """LRU cache for compressed VRAM buffers with configurable memory limit.
    
    Stores VRAMMem data in compressed form using VRAMCodeC to reduce VRAM usage.
    """
    
    def __init__(self, max_size_gb: float, stream: cuda.Stream = None):
        """
        Initialize the VRAM cacher.
        
        Args:
            max_size_gb: Maximum cache size in gigabytes.
            stream: Optional CUDA stream for operations.
        """
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.current_size_bytes = 0
        stream = stream if stream is not None else cuda.Stream()
        self.stream = stream
        self._cache: OrderedDict[Hashable, List[VRAMMem]] = OrderedDict()
        self._free_buffers: Dict[int, List[VRAMMem]] = {}
        self.reserved_size_bytes = 0
        self.hits = 0
        self.miss = 0
        self.allocation_count = 0
        self.reuse_count = 0
        self.eviction_count = 0
        self.pool_release_count = 0
    
    @staticmethod
    def _calculate_entry_size(buffers: List[VRAMMem]) -> int:
        """Calculate total size of a cache entry in bytes."""
        return sum(buf.nbytes for buf in buffers)
    
    def _evict_until_fit(self, required_bytes: int) -> None:
        """Evict LRU entries until there's enough space for required_bytes."""
        while self._cache and (self.current_size_bytes + required_bytes > self.max_size_bytes):
            _, oldest_buffers = self._cache.popitem(last=False)
            evicted_size = self._calculate_entry_size(oldest_buffers)
            self.current_size_bytes -= evicted_size
            self.eviction_count += 1
            self._recycle_buffers(oldest_buffers)

    def _recycle_buffers(self, buffers: List[VRAMMem]) -> None:
        """Return inactive buffers to exact-size free lists."""
        for buffer in buffers:
            self._free_buffers.setdefault(buffer.nbytes, []).append(buffer)

    def _release_pool_until_fit(self, required_new_bytes: int) -> None:
        """Release pooled blocks until new allocations stay within budget."""
        while self.reserved_size_bytes + required_new_bytes > self.max_size_bytes:
            available_sizes = [
                size for size, buffers in self._free_buffers.items() if buffers
            ]
            if not available_sizes:
                raise MemoryError("VRAM cache pool cannot satisfy its memory budget")
            size = max(available_sizes)
            buffer = self._free_buffers[size].pop()
            if not self._free_buffers[size]:
                del self._free_buffers[size]
            buffer.free()
            self.reserved_size_bytes -= size
            self.pool_release_count += 1

    def _acquire_buffers(self, sizes: List[int]) -> List[VRAMMem]:
        """Acquire exact-size pooled blocks, allocating only cache misses."""
        acquired: List[Optional[VRAMMem]] = [None] * len(sizes)
        missing_indices = []
        for index, size in enumerate(sizes):
            free_list = self._free_buffers.get(size)
            if free_list:
                acquired[index] = free_list.pop()
                if not free_list:
                    del self._free_buffers[size]
                self.reuse_count += 1
            else:
                missing_indices.append(index)

        required_new_bytes = sum(sizes[index] for index in missing_indices)
        try:
            self._release_pool_until_fit(required_new_bytes)
        except Exception:
            self._recycle_buffers(
                [buffer for buffer in acquired if buffer is not None]
            )
            raise
        try:
            for index in missing_indices:
                buffer = VRAMMem(sizes[index])
                acquired[index] = buffer
                self.reserved_size_bytes += buffer.nbytes
                self.allocation_count += 1
        except Exception:
            self._recycle_buffers(
                [buffer for buffer in acquired if buffer is not None]
            )
            raise

        return [buffer for buffer in acquired if buffer is not None]
    
    def put(self, key: Hashable, buffers: List[HostDeviceMem]) -> None:
        """
        Store a list of HostDeviceMem buffers in the cache (compressed).
        
        Args:
            key: Integer key for the cache entry.
            buffers: List of HostDeviceMem objects to compress and cache.
        """
        if key in self._cache:
            self._cache.move_to_end(key)
            return  # Already cached

        # Enforce the configured limit before allocating.  Allocating first
        # caused a full cache to transiently exceed its limit by one complete
        # entry, which can trigger avoidable OOMs on tightly sized GPUs.
        entry_size = sum(buf.host.nbytes for buf in buffers)
        if entry_size > self.max_size_bytes:
            return
        self._evict_until_fit(entry_size)

        sizes = [buf.host.nbytes for buf in buffers]
        saved_mems = self._acquire_buffers(sizes)
        try:
            for buf, saved_mem in zip(buffers, saved_mems):
                cuda.memcpy_dtod_async(
                    saved_mem.device,
                    buf.device,
                    buf.host.nbytes,
                    self.stream,
                )
        except Exception:
            self._recycle_buffers(saved_mems)
            raise
        
        # Add new entry at the end (most recently used)
        self._cache[key] = saved_mems
        self.current_size_bytes += entry_size
    
    def get(self, key: Hashable) -> Optional[List[VRAMMem]]:
        """
        Retrieve and decode a cached entry by key.
        
        Args:
            key: Integer key to look up.
            
        Returns:
            List of decoded NVCompDeviceBuffer if found, None otherwise.
            Use the .device pointer to access decompressed data on GPU.
        """
        if key not in self._cache:
            self.miss += 1
            return None
        
        self.hits += 1
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        
        # Decode all buffers
        return self._cache[key]
    
    def __contains__(self, key: Hashable) -> bool:
        """Check if a key exists in the cache."""
        return key in self._cache
    
    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._cache)
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        synchronize = getattr(self.stream, "synchronize", None)
        if synchronize is not None:
            synchronize()
        buffers = [
            buffer
            for entry in self._cache.values()
            for buffer in entry
        ]
        buffers.extend(
            buffer
            for free_list in self._free_buffers.values()
            for buffer in free_list
        )
        self._cache.clear()
        self._free_buffers.clear()
        for buffer in buffers:
            buffer.free()
        self.current_size_bytes = 0
        self.reserved_size_bytes = 0
        self.hits = 0
        self.miss = 0
        self.allocation_count = 0
        self.reuse_count = 0
        self.eviction_count = 0
        self.pool_release_count = 0
    
    @property
    def size_gb(self) -> float:
        """Return current cache size in gigabytes."""
        return self.current_size_bytes / (1024 * 1024 * 1024)

    @property
    def pool_size_bytes(self) -> int:
        """Return bytes reserved by inactive reusable buffers."""
        return self.reserved_size_bytes - self.current_size_bytes

    @property
    def pool_buffer_count(self) -> int:
        """Return the number of inactive reusable allocations."""
        return sum(len(buffers) for buffers in self._free_buffers.values())
    
    @property
    def hit_rate(self) -> float:
        """Return cache hit rate as a fraction."""
        total = self.hits + self.miss
        return self.hits / total if total > 0 else 0.0
