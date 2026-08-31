import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
from typing import Dict, List, Tuple

import numpy as np


PACKAGE_DIR = Path(__file__).parents[1] / "ezvtb_rt"


class FakeLogger:
    INFO = 1
    WARNING = 2

    def log(self, severity, message):
        pass


class FakeRuntimeConfig:
    cuda_graph_strategy = None


class FakeContext:
    def __init__(self):
        self.execute_calls = 0

    def execute_async_v3(self, handle):
        self.execute_calls += 1
        return True


class FakeEngine:
    num_io_tensors = 0

    def __init__(self):
        self.runtime_config = FakeRuntimeConfig()
        self.context = FakeContext()

    def create_runtime_config(self):
        return self.runtime_config

    def create_execution_context(self, runtime_config):
        self.used_runtime_config = runtime_config
        return self.context


class FakeStream:
    handle = 123

    def synchronize(self):
        pass


class FakeEvent:
    def record(self, stream):
        pass

    def time_till(self, other):
        return 0.0


def load_trt_engine_module(save_calls):
    fake_package = types.ModuleType("ezvtb_rt")
    fake_package.__path__ = [str(PACKAGE_DIR)]

    fake_trt = types.SimpleNamespace(
        ICudaEngine=FakeEngine,
        IExecutionContext=FakeContext,
        CudaGraphStrategy=types.SimpleNamespace(WHOLE_GRAPH_CAPTURE=1),
    )
    fake_cuda = types.SimpleNamespace(
        DeviceAllocation=object,
        Stream=FakeStream,
        Event=FakeEvent,
    )
    fake_engine = FakeEngine()
    fake_runtime_cache = object()

    fake_utils = types.ModuleType("ezvtb_rt.trt_utils")
    fake_utils.np = np
    fake_utils.numpy = np
    fake_utils.trt = fake_trt
    fake_utils.cuda = fake_cuda
    fake_utils.List = List
    fake_utils.Dict = Dict
    fake_utils.Tuple = Tuple
    fake_utils.TRT_LOGGER = FakeLogger()
    fake_utils.load_engine = lambda path: fake_engine
    fake_utils.get_runtime_cache_path = lambda path: Path(f"{path}.runtime.cache")
    fake_utils.load_runtime_cache = lambda config, path: (fake_runtime_cache, True)
    fake_utils.save_runtime_cache = lambda cache, path: save_calls.append((cache, path))
    fake_utils.pace_gpu_startup = lambda started, operation: 0.0

    replacements = {
        "ezvtb_rt": fake_package,
        "ezvtb_rt.trt_utils": fake_utils,
    }
    patcher = mock.patch.dict(sys.modules, replacements)
    patcher.start()
    try:
        module_name = "ezvtb_rt.trt_engine_under_test"
        spec = importlib.util.spec_from_file_location(
            module_name,
            PACKAGE_DIR / "trt_engine.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        patcher.stop()
    return module, fake_engine, fake_runtime_cache


class TensorRTEngineTests(unittest.TestCase):
    def test_runtime_cache_is_kept_and_saved_after_first_enqueue(self):
        save_calls = []
        module, fake_engine, fake_runtime_cache = load_trt_engine_module(save_calls)

        engine = module.TRTEngine("model.onnx", n_input=0)
        self.assertIs(engine.runtime_config, fake_engine.runtime_config)
        self.assertIs(engine.runtime_cache, fake_runtime_cache)
        self.assertIs(fake_engine.used_runtime_config, engine.runtime_config)
        self.assertEqual(len(save_calls), 1)

        engine.kickoff()
        self.assertEqual(fake_engine.context.execute_calls, 1)
        self.assertEqual(len(save_calls), 2)

        engine.kickoff()
        self.assertEqual(fake_engine.context.execute_calls, 2)
        self.assertEqual(len(save_calls), 2)


if __name__ == "__main__":
    unittest.main()
