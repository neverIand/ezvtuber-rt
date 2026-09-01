import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


PACKAGE_DIR = Path(__file__).parents[1] / "ezvtb_rt"


class AllocationTracker:
    def __init__(self):
        self.current = 0
        self.peak = 0
        self.allocations = 0
        self.frees = 0

    def allocate(self, size):
        self.allocations += 1
        self.current += size
        self.peak = max(self.peak, self.current)
        return FakeAllocation(self, size)


class FakeAllocation:
    def __init__(self, tracker, size):
        self.tracker = tracker
        self.size = size
        self.freed = False

    def free(self):
        if not self.freed:
            self.tracker.current -= self.size
            self.tracker.frees += 1
            self.freed = True


class FakeStream:
    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


class FakeHost:
    def __init__(self, nbytes):
        self.nbytes = nbytes


class FakeHostDeviceMem:
    def __init__(self, nbytes):
        self.host = FakeHost(nbytes)
        self.device = object()


def load_vram_cache_module(tracker):
    fake_package = types.ModuleType("ezvtb_rt")
    fake_package.__path__ = [str(PACKAGE_DIR)]

    fake_pycuda = types.ModuleType("pycuda")
    fake_pycuda.__path__ = []
    fake_cuda = types.ModuleType("pycuda.driver")
    fake_cuda.DeviceAllocation = FakeAllocation
    fake_cuda.Stream = FakeStream
    fake_cuda.mem_alloc = tracker.allocate
    fake_cuda.memcpy_dtod_async = mock.Mock()
    fake_pycuda.driver = fake_cuda

    fake_engine = types.ModuleType("ezvtb_rt.trt_engine")
    fake_engine.HostDeviceMem = FakeHostDeviceMem

    replacements = {
        "ezvtb_rt": fake_package,
        "ezvtb_rt.trt_engine": fake_engine,
        "pycuda": fake_pycuda,
        "pycuda.driver": fake_cuda,
    }
    with mock.patch.dict(sys.modules, replacements):
        spec = importlib.util.spec_from_file_location(
            "ezvtb_rt.vram_cache_under_test",
            PACKAGE_DIR / "vram_cache.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class VRAMCacheTests(unittest.TestCase):
    def test_eviction_happens_before_replacement_allocation(self):
        tracker = AllocationTracker()
        module = load_vram_cache_module(tracker)
        cache = module.VRAMCacher(max_size_gb=8 / (1024 ** 3), stream=FakeStream())

        cache.put("first", [FakeHostDeviceMem(6)])
        cache.put("second", [FakeHostDeviceMem(6)])

        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.current_size_bytes, 6)
        self.assertLessEqual(tracker.peak, cache.max_size_bytes)
        self.assertEqual(tracker.allocations, 1)
        self.assertEqual(cache.reuse_count, 1)
        self.assertEqual(cache.eviction_count, 1)

    def test_entry_larger_than_limit_is_not_allocated(self):
        tracker = AllocationTracker()
        module = load_vram_cache_module(tracker)
        cache = module.VRAMCacher(max_size_gb=8 / (1024 ** 3), stream=FakeStream())

        cache.put("oversized", [FakeHostDeviceMem(9)])

        self.assertEqual(len(cache), 0)
        self.assertEqual(tracker.peak, 0)

    def test_exact_size_pool_reuses_each_buffer_in_multi_buffer_entry(self):
        tracker = AllocationTracker()
        module = load_vram_cache_module(tracker)
        cache = module.VRAMCacher(max_size_gb=8 / (1024 ** 3), stream=FakeStream())

        cache.put("first", [FakeHostDeviceMem(3), FakeHostDeviceMem(3)])
        cache.put("second", [FakeHostDeviceMem(3), FakeHostDeviceMem(3)])

        self.assertEqual(tracker.allocations, 2)
        self.assertEqual(cache.reuse_count, 2)
        self.assertEqual(cache.eviction_count, 1)
        self.assertEqual(cache.current_size_bytes, 6)
        self.assertEqual(cache.reserved_size_bytes, 6)
        self.assertEqual(cache.pool_size_bytes, 0)

    def test_mismatched_pool_block_is_released_before_allocation(self):
        tracker = AllocationTracker()
        module = load_vram_cache_module(tracker)
        cache = module.VRAMCacher(max_size_gb=8 / (1024 ** 3), stream=FakeStream())

        cache.put("large", [FakeHostDeviceMem(6)])
        cache.put("small", [FakeHostDeviceMem(4)])

        self.assertEqual(tracker.allocations, 2)
        self.assertEqual(tracker.frees, 1)
        self.assertEqual(tracker.current, 4)
        self.assertLessEqual(tracker.peak, cache.max_size_bytes)
        self.assertEqual(cache.pool_release_count, 1)
        self.assertEqual(cache.reserved_size_bytes, 4)

    def test_unused_evicted_blocks_remain_within_total_budget(self):
        tracker = AllocationTracker()
        module = load_vram_cache_module(tracker)
        cache = module.VRAMCacher(max_size_gb=8 / (1024 ** 3), stream=FakeStream())

        cache.put("first", [FakeHostDeviceMem(3), FakeHostDeviceMem(3)])
        cache.put("second", [FakeHostDeviceMem(3)])

        self.assertEqual(cache.current_size_bytes, 3)
        self.assertEqual(cache.pool_size_bytes, 3)
        self.assertEqual(cache.pool_buffer_count, 1)
        self.assertEqual(cache.reserved_size_bytes, 6)
        self.assertLessEqual(cache.reserved_size_bytes, cache.max_size_bytes)

    def test_clear_frees_active_and_pooled_allocations(self):
        tracker = AllocationTracker()
        module = load_vram_cache_module(tracker)
        stream = FakeStream()
        cache = module.VRAMCacher(max_size_gb=8 / (1024 ** 3), stream=stream)
        cache.put("first", [FakeHostDeviceMem(3), FakeHostDeviceMem(3)])
        cache.put("second", [FakeHostDeviceMem(3)])

        cache.clear()

        self.assertEqual(stream.synchronize_calls, 1)
        self.assertEqual(tracker.current, 0)
        self.assertEqual(cache.current_size_bytes, 0)
        self.assertEqual(cache.reserved_size_bytes, 0)
        self.assertEqual(cache.pool_buffer_count, 0)
        self.assertEqual(len(cache), 0)


if __name__ == "__main__":
    unittest.main()
