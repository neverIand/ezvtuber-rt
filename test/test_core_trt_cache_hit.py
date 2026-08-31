import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import numpy as np


PACKAGE_DIR = Path(__file__).parents[1] / "ezvtb_rt"


class FakeStream:
    def __init__(self):
        self.sync_calls = 0

    def synchronize(self):
        self.sync_calls += 1


class FakeOutputMemory:
    def __init__(self):
        self.host = np.zeros((512, 512, 4), dtype=np.uint8)
        self.htod_calls = 0

    def htod(self, stream):
        self.htod_calls += 1


class FakeTHA:
    def __init__(self):
        self.output_memory = FakeOutputMemory()
        self.cachestream = FakeStream()
        self.infer_calls = 0

    def getOutputMem(self):
        return self.output_memory

    def asyncInfer(self, pose, stream):
        self.infer_calls += 1


class FakeCacher:
    def __init__(self, cached):
        self.cached = cached

    def get(self, key):
        return self.cached


def load_core_trt_module():
    fake_package = types.ModuleType("ezvtb_rt")
    fake_package.__path__ = [str(PACKAGE_DIR)]
    fake_package.EZVTB_DATA = "unused"

    fake_utils = types.ModuleType("ezvtb_rt.trt_utils")
    fake_utils.get_gpu_duty_limit_percent = lambda: 90.0

    fake_engine_module = types.ModuleType("ezvtb_rt.trt_engine")
    fake_engine_module.TRTEngine = object
    fake_engine_module.HostDeviceMem = FakeOutputMemory

    fake_cache_module = types.ModuleType("ezvtb_rt.cache")
    fake_cache_module.Cacher = FakeCacher
    fake_cache_module.array_cache_key = lambda array: (
        array.dtype.str,
        array.shape,
        np.ascontiguousarray(array).tobytes(),
    )

    replacements = {
        "ezvtb_rt": fake_package,
        "ezvtb_rt.trt_utils": fake_utils,
        "ezvtb_rt.trt_engine": fake_engine_module,
        "ezvtb_rt.tha3": types.SimpleNamespace(THA3Engines=object),
        "ezvtb_rt.tha4": types.SimpleNamespace(THA4Engines=object),
        "ezvtb_rt.tha4_student": types.SimpleNamespace(THA4StudentEngines=object),
        "ezvtb_rt.cache": fake_cache_module,
        "pyanime4k": types.SimpleNamespace(Anime4K=object),
    }

    with mock.patch.dict(sys.modules, replacements):
        spec = importlib.util.spec_from_file_location(
            "ezvtb_rt.core_trt_under_test",
            PACKAGE_DIR / "core_trt.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class CoreTRTCacheHitTests(unittest.TestCase):
    def test_plain_tha_cache_hit_returns_without_gpu_upload(self):
        module = load_core_trt_module()
        cached = np.full((512, 512, 4), 17, dtype=np.uint8)
        core = object.__new__(module.CoreTRT)
        core.cacher = FakeCacher(cached)
        core.cache_stream = FakeStream()
        core.main_stream = FakeStream()
        core.tha = FakeTHA()
        core.tha_model_fp16 = False
        core.v3 = True
        core.rife = None
        core.sr = None
        core.sr_a4k = None

        result = core.inference([np.zeros(45, dtype=np.float32)])

        np.testing.assert_array_equal(result[0], cached)
        self.assertEqual(core.tha.infer_calls, 0)
        self.assertEqual(core.tha.output_memory.htod_calls, 0)
        self.assertEqual(core.tha.cachestream.sync_calls, 1)


if __name__ == "__main__":
    unittest.main()
