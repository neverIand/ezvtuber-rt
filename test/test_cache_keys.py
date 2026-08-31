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


if __name__ == "__main__":
    unittest.main()
