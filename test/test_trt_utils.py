from contextlib import contextmanager
import importlib.util
from pathlib import Path
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


PACKAGE_DIR = Path(__file__).parents[1] / "ezvtb_rt"


class FakeLogger:
    INFO = 1
    WARNING = 2
    ERROR = 3

    def __init__(self, severity):
        self.severity = severity
        self.messages = []

    def log(self, severity, message):
        self.messages.append((severity, message))


class FakeEngineValidity:
    VALID = "valid"
    INVALID = "invalid"


class FakeRuntime:
    engine_header_size = 4

    def __init__(self, logger):
        self.logger = logger

    def get_engine_validity(self, header):
        if bytes(header) == b"BAD!":
            return FakeEngineValidity.INVALID, 1
        return FakeEngineValidity.VALID, 0

    def deserialize_cuda_engine(self, payload):
        if payload.startswith(b"FAIL"):
            return None
        return {"payload": payload}


class FakeCudaDevice:
    def __init__(self, device_id):
        self.device_id = device_id

    def name(self):
        return f"Fake GPU {self.device_id}"

    def compute_capability(self):
        return (9, 9)


@contextmanager
def loaded_trt_utils():
    fake_package = types.ModuleType("ezvtb_rt")
    fake_package.__path__ = [str(PACKAGE_DIR)]

    fake_trt = types.ModuleType("tensorrt_rtx")
    fake_trt.__version__ = "1.3-test"
    fake_trt.Logger = FakeLogger
    fake_trt.Runtime = FakeRuntime
    fake_trt.EngineValidity = FakeEngineValidity

    fake_pycuda = types.ModuleType("pycuda")
    fake_pycuda.__path__ = []
    fake_cuda = types.ModuleType("pycuda.driver")
    fake_cuda.init = mock.Mock()
    fake_cuda.Device = FakeCudaDevice
    fake_pycuda.driver = fake_cuda

    replacements = {
        "ezvtb_rt": fake_package,
        "tensorrt_rtx": fake_trt,
        "pycuda": fake_pycuda,
        "pycuda.driver": fake_cuda,
    }
    with mock.patch.dict(sys.modules, replacements):
        sys.modules.pop("ezvtb_rt.trt_cache", None)
        module_name = "ezvtb_rt.trt_utils_under_test"
        spec = importlib.util.spec_from_file_location(
            module_name,
            PACKAGE_DIR / "trt_utils.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
        sys.modules.pop("ezvtb_rt.trt_cache", None)


class TensorRTUtilsTests(unittest.TestCase):
    def test_cache_identity_does_not_require_an_active_context(self):
        with loaded_trt_utils() as trt_utils, mock.patch.dict(
            os.environ,
            {"EZVTB_DEVICE_ID": "1"},
        ):
            identity = trt_utils._trt_cache_identity()

            self.assertEqual(
                identity,
                "1.3-test|device-id=1|name=Fake GPU 1|cc=(9, 9)",
            )
            trt_utils.cuda.init.assert_called_once_with()

    def test_onnx_engine_is_built_once_then_reused(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "model.onnx"
            source.write_bytes(b"onnx-model")
            with mock.patch.dict(
                os.environ,
                {"EZVTB_TRT_CACHE_DIR": str(root / "cache")},
            ), loaded_trt_utils() as trt_utils:
                trt_utils.build_engine = mock.Mock(return_value=b"GOOD-engine")

                first = trt_utils.load_engine(str(source))
                second = trt_utils.load_engine(str(source))

                self.assertEqual(first, second)
                trt_utils.build_engine.assert_called_once_with(str(source))
                self.assertEqual(len(list((root / "cache").glob("*.trt"))), 1)

    def test_invalid_cached_engine_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "model.onnx"
            source.write_bytes(b"onnx-model")
            with mock.patch.dict(
                os.environ,
                {"EZVTB_TRT_CACHE_DIR": str(root / "cache")},
            ), loaded_trt_utils() as trt_utils:
                engine_path = trt_utils.get_engine_cache_path(str(source))
                engine_path.write_bytes(b"BAD!-old-engine")
                trt_utils.build_engine = mock.Mock(return_value=b"GOOD-new-engine")

                engine = trt_utils.load_engine(str(source))

                self.assertEqual(engine["payload"], b"GOOD-new-engine")
                self.assertEqual(engine_path.read_bytes(), b"GOOD-new-engine")
                trt_utils.build_engine.assert_called_once_with(str(source))

    def test_startup_pacing_uses_configured_duty_cycle(self):
        with loaded_trt_utils() as trt_utils, mock.patch.dict(
            os.environ,
            {"EZVTB_GPU_DUTY_LIMIT": "90"},
        ), mock.patch.object(
            trt_utils.time,
            "perf_counter",
            return_value=10.0,
        ), mock.patch.object(
            trt_utils.time,
            "sleep",
        ) as sleep:
            cooldown = trt_utils.pace_gpu_startup(1.0, "test")

            self.assertAlmostEqual(cooldown, 1.0)
            sleep.assert_called_once_with(cooldown)


if __name__ == "__main__":
    unittest.main()
