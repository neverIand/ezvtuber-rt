import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "ezvtb_rt" / "cache.py"
SPEC = importlib.util.spec_from_file_location("ezvtb_rt_cache_test", MODULE_PATH)
cache_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cache_module)


class ArrayCacheKeyTests(unittest.TestCase):
    def test_key_uses_values_omitted_by_numpy_string_formatting(self):
        first = np.zeros(10000, dtype=np.float32)
        second = first.copy()
        second[5000] = 1.0

        self.assertEqual(str(first), str(second))
        self.assertNotEqual(
            cache_module.array_cache_key(first),
            cache_module.array_cache_key(second),
        )

    def test_key_includes_shape_and_dtype(self):
        flat = np.arange(8, dtype=np.uint8)
        reshaped = flat.reshape(2, 4)
        wider_type = flat.view(np.uint16)

        self.assertNotEqual(
            cache_module.array_cache_key(flat),
            cache_module.array_cache_key(reshaped),
        )
        self.assertNotEqual(
            cache_module.array_cache_key(flat),
            cache_module.array_cache_key(wider_type),
        )

    def test_noncontiguous_view_matches_equivalent_contiguous_array(self):
        view = np.arange(12, dtype=np.float32).reshape(3, 4)[:, ::2]
        contiguous = np.ascontiguousarray(view)
        self.assertEqual(
            cache_module.array_cache_key(view),
            cache_module.array_cache_key(contiguous),
        )


class CacheBudgetTests(unittest.TestCase):
    def test_without_super_resolution_the_base_cache_gets_the_full_budget(self):
        self.assertEqual(
            cache_module.split_ram_cache_budget(2.0, False),
            (2.0, 0.0),
        )

    def test_super_resolution_budget_is_split_by_frame_byte_size(self):
        base_giga, sr_giga = cache_module.split_ram_cache_budget(2.0, True)

        self.assertAlmostEqual(base_giga, 0.4)
        self.assertAlmostEqual(sr_giga, 1.6)
        self.assertAlmostEqual(base_giga + sr_giga, 2.0)
        self.assertAlmostEqual(sr_giga / base_giga, 4.0)

    def test_nonpositive_budget_cannot_create_a_cache(self):
        self.assertEqual(
            cache_module.split_ram_cache_budget(-2.0, True),
            (0.0, 0.0),
        )


class RawCacheTests(unittest.TestCase):
    def test_raw_mode_owns_exact_read_only_frame(self):
        frame = np.arange(8 * 8 * 4, dtype=np.uint8).reshape(8, 8, 4)
        cacher = cache_module.Cacher(
            max_volume_giga=frame.nbytes * 2 / (1024 ** 3),
            width=8,
            height=8,
            storage_mode="raw",
        )

        cacher.put("frame", frame)
        frame.fill(0)
        cached = cacher.get("frame")

        np.testing.assert_array_equal(
            cached,
            np.arange(8 * 8 * 4, dtype=np.uint8).reshape(8, 8, 4),
        )
        self.assertFalse(cached.flags.writeable)
        self.assertFalse(np.shares_memory(cached, frame))

    def test_raw_mode_evicts_before_exceeding_budget(self):
        frame_bytes = 4 * 4 * 4
        cacher = cache_module.Cacher(
            max_volume_giga=frame_bytes * 2 / (1024 ** 3),
            width=4,
            height=4,
            storage_mode="raw",
        )
        frames = [np.full((4, 4, 4), value, dtype=np.uint8) for value in range(3)]

        for index, frame in enumerate(frames):
            cacher.put(index, frame)
            self.assertLessEqual(cacher.cached_kbytes, cacher.max_kbytes)

        self.assertNotIn(0, cacher.cache)
        self.assertIn(1, cacher.cache)
        self.assertIn(2, cacher.cache)
        self.assertEqual(cacher.cached_kbytes, frame_bytes * 2 / 1024)

    def test_oversized_raw_entry_does_not_evict_existing_data(self):
        small = np.zeros((2, 2, 4), dtype=np.uint8)
        large = np.zeros((8, 8, 4), dtype=np.uint8)
        cacher = cache_module.Cacher(
            max_volume_giga=small.nbytes / (1024 ** 3),
            width=2,
            height=2,
            storage_mode="raw",
        )
        cacher.put("small", small)

        cacher.put("large", large)

        self.assertIn("small", cacher.cache)
        self.assertNotIn("large", cacher.cache)
        self.assertEqual(cacher.cached_kbytes, small.nbytes / 1024)

    def test_brotli_remains_default_and_lossless(self):
        frame = np.arange(4 * 4 * 4, dtype=np.uint8).reshape(4, 4, 4)
        cacher = cache_module.Cacher(
            max_volume_giga=1 / 1024,
            width=4,
            height=4,
        )

        cacher.put("frame", frame)
        cached = cacher.get("frame")

        self.assertEqual(cacher.storage_mode, "brotli")
        np.testing.assert_array_equal(cached, frame)

    def test_unknown_storage_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "storage_mode"):
            cache_module.Cacher(storage_mode="lossy")


if __name__ == "__main__":
    unittest.main()
