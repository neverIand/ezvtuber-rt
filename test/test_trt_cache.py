import importlib.util
from pathlib import Path
import tempfile
import unittest


MODULE_PATH = Path(__file__).parents[1] / "ezvtb_rt" / "trt_cache.py"
SPEC = importlib.util.spec_from_file_location("ezvtb_rt_trt_cache_test", MODULE_PATH)
trt_cache = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trt_cache)


class FakeRuntimeCache:
    def __init__(self, serialized=b"new-cache"):
        self.serialized = serialized
        self.deserialized = None

    def deserialize(self, data):
        self.deserialized = data
        return True

    def serialize(self):
        return self.serialized


class FakeRuntimeConfig:
    def __init__(self, runtime_cache):
        self.runtime_cache = runtime_cache
        self.attached = None

    def create_runtime_cache(self):
        return self.runtime_cache

    def set_runtime_cache(self, runtime_cache):
        self.attached = runtime_cache
        return True


class TensorRTCacheTests(unittest.TestCase):
    def test_engine_cache_key_distinguishes_content_and_trt_version(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_dir = root / "first"
            second_dir = root / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            first = first_dir / "model.onnx"
            same_content = second_dir / "model.onnx"
            different_content = root / "model.onnx"
            first.write_bytes(b"model-a")
            same_content.write_bytes(b"model-a")
            different_content.write_bytes(b"model-b")

            first_path = trt_cache.get_engine_cache_path(first, "1.3", root / "cache")
            same_path = trt_cache.get_engine_cache_path(same_content, "1.3", root / "cache")
            different_path = trt_cache.get_engine_cache_path(
                different_content,
                "1.3",
                root / "cache",
            )
            new_version_path = trt_cache.get_engine_cache_path(
                first,
                "1.4",
                root / "cache",
            )

            self.assertEqual(first_path, same_path)
            self.assertNotEqual(first_path, different_path)
            self.assertNotEqual(first_path, new_version_path)

    def test_atomic_write_replaces_complete_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            destination = Path(temporary_directory) / "cache.bin"
            destination.write_bytes(b"old")
            trt_cache.atomic_write(destination, b"replacement")

            self.assertEqual(destination.read_bytes(), b"replacement")
            self.assertEqual(list(destination.parent.glob("*.tmp")), [])

    def test_runtime_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            cache_path = Path(temporary_directory) / "runtime.cache"
            cache_path.write_bytes(b"existing-cache")
            runtime_cache = FakeRuntimeCache()
            runtime_config = FakeRuntimeConfig(runtime_cache)

            loaded_cache, loaded = trt_cache.load_runtime_cache(
                runtime_config,
                cache_path,
            )
            self.assertTrue(loaded)
            self.assertIs(loaded_cache, runtime_cache)
            self.assertIs(runtime_config.attached, runtime_cache)
            self.assertEqual(runtime_cache.deserialized, b"existing-cache")

            trt_cache.save_runtime_cache(runtime_cache, cache_path)
            self.assertEqual(cache_path.read_bytes(), b"new-cache")

    def test_engine_build_lock_removes_lock_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            engine_path = Path(temporary_directory) / "model.trt"
            lock_path = Path(f"{engine_path}.lock")
            with trt_cache.engine_build_lock(engine_path):
                self.assertTrue(lock_path.is_file())
            self.assertFalse(lock_path.exists())


if __name__ == "__main__":
    unittest.main()
