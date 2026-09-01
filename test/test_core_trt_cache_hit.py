import importlib.util
import os
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
        self.dtoh_calls = 0

    def htod(self, stream):
        self.htod_calls += 1

    def dtoh(self, stream):
        self.dtoh_calls += 1


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
        self.puts = []

    def get(self, key):
        return self.cached

    def put(self, key, value):
        self.puts.append((key, np.copy(value)))


class FakeStudent:
    last_init = None

    def __init__(self, model_dir, vram_cache_size):
        type(self).last_init = (model_dir, vram_cache_size)


def load_core_trt_module():
    fake_package = types.ModuleType("ezvtb_rt")
    fake_package.__path__ = [str(PACKAGE_DIR)]
    fake_package.EZVTB_DATA = "unused"

    fake_utils = types.ModuleType("ezvtb_rt.trt_utils")
    fake_utils.get_gpu_duty_limit_percent = lambda: 90.0
    fake_utils.cuda = types.SimpleNamespace(Stream=FakeStream)

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
    fake_cache_module.split_ram_cache_budget = lambda total, sr_enabled: (
        (total / 5.0, total * 4.0 / 5.0)
        if sr_enabled else (total, 0.0)
    )

    replacements = {
        "ezvtb_rt": fake_package,
        "ezvtb_rt.trt_utils": fake_utils,
        "ezvtb_rt.trt_engine": fake_engine_module,
        "ezvtb_rt.tha3": types.SimpleNamespace(THA3Engines=object),
        "ezvtb_rt.tha4": types.SimpleNamespace(THA4Engines=object),
        "ezvtb_rt.tha4_student": types.SimpleNamespace(
            THA4StudentEngines=FakeStudent,
        ),
        "ezvtb_rt.cache": fake_cache_module,
        "cv2": types.ModuleType("cv2"),
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
    def test_student_model_uses_configured_root_and_vram_budget(self):
        module = load_core_trt_module()
        FakeStudent.last_init = None

        module.CoreTRT(
            tha_model_version="v4_student",
            tha_model_name="demo",
            vram_cache_size=0.375,
            cache_max_giga=0.0,
        )

        expected_path = os.path.normpath(
            os.path.join("unused", "custom_tha4_models", "demo")
        )
        self.assertEqual(FakeStudent.last_init, (expected_path, 0.375))

    def test_plain_tha_cache_hit_returns_without_gpu_upload(self):
        module = load_core_trt_module()
        cached = np.full((512, 512, 4), 17, dtype=np.uint8)
        core = object.__new__(module.CoreTRT)
        core.cacher = FakeCacher(cached)
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

    def test_copy_output_false_exposes_reusable_host_buffer(self):
        module = load_core_trt_module()
        core = object.__new__(module.CoreTRT)
        core.cacher = None
        core.main_stream = FakeStream()
        core.tha = FakeTHA()
        core.tha_model_fp16 = False
        core.v3 = True
        core.rife = None
        core.sr = None
        core.sr_a4k = None

        result = core.inference(
            [np.zeros(45, dtype=np.float32)],
            copy_output=False,
        )

        self.assertTrue(np.shares_memory(result[0], core.tha.output_memory.host))
        self.assertEqual(core.tha.output_memory.dtoh_calls, 1)

    def test_rife_cache_records_every_intermediate_pose_for_all_scales(self):
        module = load_core_trt_module()
        for scale in (2, 3, 4):
            with self.subTest(scale=scale):
                poses = [
                    np.full(45, index, dtype=np.float32)
                    for index in range(scale)
                ]
                frames = np.stack(
                    [
                        np.full((2, 2, 4), index + 10, dtype=np.uint8)
                        for index in range(scale)
                    ],
                    axis=0,
                )
                cacher = FakeCacher(None)

                module._cache_rife_intermediate_frames(cacher, poses, frames)

                self.assertEqual(len(cacher.puts), scale - 1)
                for index, (key, cached_frame) in enumerate(cacher.puts):
                    self.assertEqual(
                        key,
                        (poses[index].dtype.str, poses[index].shape, poses[index].tobytes()),
                    )
                    np.testing.assert_array_equal(cached_frame, frames[index])


if __name__ == "__main__":
    unittest.main()
